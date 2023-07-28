# This file is based on:
# https://github.com/bmaltais/kohya_ss/blob/397bf51a8cd36104e52055358e4ffd066c5858df/networks/lora.py
#
# Reference:
# https://github.com/microsoft/LoRA/blob/main/loralib/layers.py
# https://github.com/cloneofsimo/lora/blob/master/lora_diffusion/lora.py

import math
import os
from typing import Dict, List, Optional, Tuple, Type, Union
from diffusers import AutoencoderKL
from transformers import CLIPTextModel
import torch
import re


RE_UPDOWN = re.compile(
    r"(up|down)_blocks_(\d+)_(resnets|upsamplers|downsamplers|attentions)_(\d+)_"
)


class LoRAModule(torch.nn.Module):
    """
    replaces forward method of the original Linear, instead of replacing the original Linear module.
    """

    def __init__(
        self,
        lora_name,
        org_module: torch.nn.Module,
        multiplier=1.0,
        lora_dim=4,
        alpha=1,
        dropout=None,
        rank_dropout=None,
        module_dropout=None,
    ):
        """if alpha == 0 or None, alpha is rank (no scaling)."""
        super().__init__()
        self.lora_name = lora_name

        if org_module.__class__.__name__ == "Conv2d":
            in_dim = org_module.in_channels
            out_dim = org_module.out_channels
        else:
            in_dim = org_module.in_features
            out_dim = org_module.out_features

        self.lora_dim = lora_dim

        if org_module.__class__.__name__ == "Conv2d":
            kernel_size = org_module.kernel_size
            stride = org_module.stride
            padding = org_module.padding
            self.lora_down = torch.nn.Conv2d(
                in_dim, self.lora_dim, kernel_size, stride, padding, bias=False
            )
            self.lora_up = torch.nn.Conv2d(
                self.lora_dim, out_dim, (1, 1), (1, 1), bias=False
            )
        else:
            self.lora_down = torch.nn.Linear(in_dim, self.lora_dim, bias=False)
            self.lora_up = torch.nn.Linear(self.lora_dim, out_dim, bias=False)

        if type(alpha) == torch.Tensor:
            alpha = (
                alpha.detach().float().numpy()
            )  # without casting, bf16 causes error
        alpha = self.lora_dim if alpha is None or alpha == 0 else alpha
        self.scale = alpha / self.lora_dim
        self.register_buffer("alpha", torch.tensor(alpha))  # 定数として扱える

        # This initialization is based on Microsoft's implementation:
        # https://github.com/microsoft/LoRA/blob/998cfe4d351f4d6b4a47f0921dec2397aa0b9dfe/loralib/layers.py#L123
        torch.nn.init.kaiming_uniform_(self.lora_down.weight, a=math.sqrt(5))
        torch.nn.init.zeros_(self.lora_up.weight)

        self.multiplier = multiplier
        self.org_module = org_module
        self.dropout = dropout
        self.rank_dropout = rank_dropout
        self.module_dropout = module_dropout

    def apply_to(self):
        self.org_forward = self.org_module.forward
        self.org_module.forward = self.forward
        del self.org_module

    def forward(self, x):
        org_forwarded = self.org_forward(x)

        # module dropout
        if self.module_dropout is not None and self.training:
            if torch.rand(1) < self.module_dropout:
                return org_forwarded

        lx = self.lora_down(x)

        # normal dropout
        if self.dropout is not None and self.training:
            lx = torch.nn.functional.dropout(lx, p=self.dropout)

        # rank dropout
        if self.rank_dropout is not None and self.training:
            mask = (
                torch.rand((lx.size(0), self.lora_dim), device=lx.device)
                > self.rank_dropout
            )
            if len(lx.size()) == 3:
                mask = mask.unsqueeze(1)  # for Text Encoder
            elif len(lx.size()) == 4:
                mask = mask.unsqueeze(-1).unsqueeze(-1)  # for Conv2d
            lx = lx * mask

            # scaling for rank dropout: treat as if the rank is changed
            # maskから計算することも考えられるが、augmentation的な効果を期待してrank_dropoutを用いる
            scale = self.scale * (
                1.0 / (1.0 - self.rank_dropout)
            )  # redundant for readability
        else:
            scale = self.scale

        lx = self.lora_up(lx)

        return org_forwarded + lx * self.multiplier * scale


def parse_block_lr_kwargs(nw_kwargs):
    down_lr_weight = nw_kwargs.get("down_lr_weight", None)
    mid_lr_weight = nw_kwargs.get("mid_lr_weight", None)
    up_lr_weight = nw_kwargs.get("up_lr_weight", None)

    # 以上のいずれにも設定がない場合は無効としてNoneを返す
    if (
        down_lr_weight is None
        and mid_lr_weight is None
        and up_lr_weight is None
    ):
        return None, None, None

    # extract learning rate weight for each block
    if down_lr_weight is not None:
        # if some parameters are not set, use zero
        if "," in down_lr_weight:
            down_lr_weight = [
                (float(s) if s else 0.0) for s in down_lr_weight.split(",")
            ]

    if mid_lr_weight is not None:
        mid_lr_weight = float(mid_lr_weight)

    if up_lr_weight is not None:
        if "," in up_lr_weight:
            up_lr_weight = [
                (float(s) if s else 0.0) for s in up_lr_weight.split(",")
            ]

    down_lr_weight, mid_lr_weight, up_lr_weight = get_block_lr_weight(
        down_lr_weight,
        mid_lr_weight,
        up_lr_weight,
        float(nw_kwargs.get("block_lr_zero_threshold", 0.0)),
    )

    return down_lr_weight, mid_lr_weight, up_lr_weight


def create_network(
    multiplier: float,
    network_dim: Optional[int],
    network_alpha: Optional[float],
    vae: AutoencoderKL,
    text_encoder: Union[CLIPTextModel, List[CLIPTextModel]],
    unet,
    neuron_dropout: Optional[float] = None,
    **kwargs,
):
    if network_dim is None:
        network_dim = 4  # default
    if network_alpha is None:
        network_alpha = 1.0

    # extract dim/alpha for conv2d, and block dim
    conv_dim = kwargs.get("conv_dim", None)
    conv_alpha = kwargs.get("conv_alpha", None)
    if conv_dim is not None:
        conv_dim = int(conv_dim)
        if conv_alpha is None:
            conv_alpha = 1.0
        else:
            conv_alpha = float(conv_alpha)

    # block dim/alpha/lr
    block_dims = kwargs.get("block_dims", None)
    down_lr_weight, mid_lr_weight, up_lr_weight = parse_block_lr_kwargs(kwargs)

    # 以上のいずれかに指定があればblockごとのdim(rank)を有効にする
    if (
        block_dims is not None
        or down_lr_weight is not None
        or mid_lr_weight is not None
        or up_lr_weight is not None
    ):
        block_alphas = kwargs.get("block_alphas", None)
        conv_block_dims = kwargs.get("conv_block_dims", None)
        conv_block_alphas = kwargs.get("conv_block_alphas", None)

        block_dims, block_alphas, conv_block_dims, conv_block_alphas = (
            get_block_dims_and_alphas(
                block_dims,
                block_alphas,
                network_dim,
                network_alpha,
                conv_block_dims,
                conv_block_alphas,
                conv_dim,
                conv_alpha,
            )
        )

        # remove block dim/alpha without learning rate
        block_dims, block_alphas, conv_block_dims, conv_block_alphas = (
            remove_block_dims_and_alphas(
                block_dims,
                block_alphas,
                conv_block_dims,
                conv_block_alphas,
                down_lr_weight,
                mid_lr_weight,
                up_lr_weight,
            )
        )

    else:
        block_alphas = None
        conv_block_dims = None
        conv_block_alphas = None

    # rank/module dropout
    rank_dropout = kwargs.get("rank_dropout", None)
    if rank_dropout is not None:
        rank_dropout = float(rank_dropout)
    module_dropout = kwargs.get("module_dropout", None)
    if module_dropout is not None:
        module_dropout = float(module_dropout)

    # すごく引数が多いな ( ^ω^)･･･
    network = LoRANetwork(
        text_encoder,
        unet,
        multiplier=multiplier,
        lora_dim=network_dim,
        alpha=network_alpha,
        dropout=neuron_dropout,
        rank_dropout=rank_dropout,
        module_dropout=module_dropout,
        conv_lora_dim=conv_dim,
        conv_alpha=conv_alpha,
        block_dims=block_dims,
        block_alphas=block_alphas,
        conv_block_dims=conv_block_dims,
        conv_block_alphas=conv_block_alphas,
        varbose=True,
    )

    if (
        up_lr_weight is not None
        or mid_lr_weight is not None
        or down_lr_weight is not None
    ):
        network.set_block_lr_weight(up_lr_weight, mid_lr_weight, down_lr_weight)

    return network


# このメソッドは外部から呼び出される可能性を考慮しておく
# network_dim, network_alpha にはデフォルト値が入っている。
# block_dims, block_alphas は両方ともNoneまたは両方とも値が入っている
# conv_dim, conv_alpha は両方ともNoneまたは両方とも値が入っている
def get_block_dims_and_alphas(
    block_dims,
    block_alphas,
    network_dim,
    network_alpha,
    conv_block_dims,
    conv_block_alphas,
    conv_dim,
    conv_alpha,
):
    num_total_blocks = LoRANetwork.NUM_OF_BLOCKS * 2 + 1

    def parse_ints(s):
        return [int(i) for i in s.split(",")]

    def parse_floats(s):
        return [float(i) for i in s.split(",")]

    # block_dimsとblock_alphasをパースする。必ず値が入る
    if block_dims is not None:
        block_dims = parse_ints(block_dims)
        assert len(block_dims) == num_total_blocks, (
            f"block_dims must have {num_total_blocks} elements /"
            f" block_dimsは{num_total_blocks}個指定してください"
        )
    else:
        print(
            f"block_dims is not specified. all dims are set to {network_dim} /"
            " block_dimsが指定されていません。"
            f"すべてのdimは{network_dim}になります"
        )
        block_dims = [network_dim] * num_total_blocks

    if block_alphas is not None:
        block_alphas = parse_floats(block_alphas)
        assert len(block_alphas) == num_total_blocks, (
            f"block_alphas must have {num_total_blocks} elements /"
            f" block_alphasは{num_total_blocks}個指定してください"
        )
    else:
        print(
            "block_alphas is not specified. all alphas are set to"
            f" {network_alpha} / block_alphasが指定されていません。"
            f"すべてのalphaは{network_alpha}になります"
        )
        block_alphas = [network_alpha] * num_total_blocks

    # conv_block_dimsとconv_block_alphasを、指定がある場合のみパースする。指定がなければconv_dimとconv_alphaを使う
    if conv_block_dims is not None:
        conv_block_dims = parse_ints(conv_block_dims)
        assert len(conv_block_dims) == num_total_blocks, (
            f"conv_block_dims must have {num_total_blocks} elements /"
            f" conv_block_dimsは{num_total_blocks}個指定してください"
        )

        if conv_block_alphas is not None:
            conv_block_alphas = parse_floats(conv_block_alphas)
            assert len(conv_block_alphas) == num_total_blocks, (
                f"conv_block_alphas must have {num_total_blocks} elements /"
                f" conv_block_alphasは{num_total_blocks}個指定してください"
            )
        else:
            if conv_alpha is None:
                conv_alpha = 1.0
            print(
                "conv_block_alphas is not specified. all alphas are set to"
                f" {conv_alpha} / conv_block_alphasが指定されていません。"
                f"すべてのalphaは{conv_alpha}になります"
            )
            conv_block_alphas = [conv_alpha] * num_total_blocks
    else:
        if conv_dim is not None:
            print(
                f"conv_dim/alpha for all blocks are set to {conv_dim} and"
                f" {conv_alpha} /"
                f" すべてのブロックのconv_dimとalphaは{conv_dim}および{conv_alpha}になります"
            )
            conv_block_dims = [conv_dim] * num_total_blocks
            conv_block_alphas = [conv_alpha] * num_total_blocks
        else:
            conv_block_dims = None
            conv_block_alphas = None

    return block_dims, block_alphas, conv_block_dims, conv_block_alphas


# 層別学習率用に層ごとの学習率に対する倍率を定義する、外部から呼び出される可能性を考慮しておく
def get_block_lr_weight(
    down_lr_weight, mid_lr_weight, up_lr_weight, zero_threshold
) -> Tuple[List[float], List[float], List[float]]:
    # パラメータ未指定時は何もせず、今までと同じ動作とする
    if (
        up_lr_weight is None
        and mid_lr_weight is None
        and down_lr_weight is None
    ):
        return None, None, None

    max_len = LoRANetwork.NUM_OF_BLOCKS  # フルモデル相当でのup,downの層の数

    def get_list(name_with_suffix) -> List[float]:
        import math

        tokens = name_with_suffix.split("+")
        name = tokens[0]
        base_lr = float(tokens[1]) if len(tokens) > 1 else 0.0

        if name == "cosine":
            return [
                math.sin(math.pi * (i / (max_len - 1)) / 2) + base_lr
                for i in reversed(range(max_len))
            ]
        elif name == "sine":
            return [
                math.sin(math.pi * (i / (max_len - 1)) / 2) + base_lr
                for i in range(max_len)
            ]
        elif name == "linear":
            return [i / (max_len - 1) + base_lr for i in range(max_len)]
        elif name == "reverse_linear":
            return [
                i / (max_len - 1) + base_lr for i in reversed(range(max_len))
            ]
        elif name == "zeros":
            return [0.0 + base_lr] * max_len
        else:
            print(
                "Unknown lr_weight argument %s is used. Valid arguments:  /"
                " 不明なlr_weightの引数 %s が使われました。"
                "有効な引数:\n\tcosine, sine, linear, reverse_linear, zeros"
                % (name)
            )
            return None

    if type(down_lr_weight) == str:
        down_lr_weight = get_list(down_lr_weight)
    if type(up_lr_weight) == str:
        up_lr_weight = get_list(up_lr_weight)

    if (up_lr_weight != None and len(up_lr_weight) > max_len) or (
        down_lr_weight != None and len(down_lr_weight) > max_len
    ):
        print(
            "down_weight or up_weight is too long. Parameters after %d-th are"
            " ignored." % max_len
        )
        print(
            "down_weightもしくはup_weightが長すぎます。"
            "%d個目以降のパラメータは無視されます。" % max_len
        )
        up_lr_weight = up_lr_weight[:max_len]
        down_lr_weight = down_lr_weight[:max_len]

    if (up_lr_weight != None and len(up_lr_weight) < max_len) or (
        down_lr_weight != None and len(down_lr_weight) < max_len
    ):
        print(
            "down_weight or up_weight is too short. Parameters after %d-th are"
            " filled with 1." % max_len
        )
        print(
            "down_weightもしくはup_weightが短すぎます。"
            "%d個目までの不足したパラメータは1で補われます。" % max_len
        )

        if down_lr_weight != None and len(down_lr_weight) < max_len:
            down_lr_weight = down_lr_weight + [1.0] * (
                max_len - len(down_lr_weight)
            )
        if up_lr_weight != None and len(up_lr_weight) < max_len:
            up_lr_weight = up_lr_weight + [1.0] * (max_len - len(up_lr_weight))

    if (
        (up_lr_weight != None)
        or (mid_lr_weight != None)
        or (down_lr_weight != None)
    ):
        print("apply block learning rate / 階層別学習率を適用します。")
        if down_lr_weight != None:
            down_lr_weight = [
                w if w > zero_threshold else 0 for w in down_lr_weight
            ]
            print(
                "down_lr_weight (shallower -> deeper, 浅い層->深い層):",
                down_lr_weight,
            )
        else:
            print("down_lr_weight: all 1.0, すべて1.0")

        if mid_lr_weight != None:
            mid_lr_weight = (
                mid_lr_weight if mid_lr_weight > zero_threshold else 0
            )
            print("mid_lr_weight:", mid_lr_weight)
        else:
            print("mid_lr_weight: 1.0")

        if up_lr_weight != None:
            up_lr_weight = [
                w if w > zero_threshold else 0 for w in up_lr_weight
            ]
            print(
                "up_lr_weight (deeper -> shallower, 深い層->浅い層):",
                up_lr_weight,
            )
        else:
            print("up_lr_weight: all 1.0, すべて1.0")

    return down_lr_weight, mid_lr_weight, up_lr_weight


# lr_weightが0のblockをblock_dimsから除外する、外部から呼び出す可能性を考慮しておく
def remove_block_dims_and_alphas(
    block_dims,
    block_alphas,
    conv_block_dims,
    conv_block_alphas,
    down_lr_weight,
    mid_lr_weight,
    up_lr_weight,
):
    # set 0 to block dim without learning rate to remove the block
    if down_lr_weight != None:
        for i, lr in enumerate(down_lr_weight):
            if lr == 0:
                block_dims[i] = 0
                if conv_block_dims is not None:
                    conv_block_dims[i] = 0
    if mid_lr_weight != None:
        if mid_lr_weight == 0:
            block_dims[LoRANetwork.NUM_OF_BLOCKS] = 0
            if conv_block_dims is not None:
                conv_block_dims[LoRANetwork.NUM_OF_BLOCKS] = 0
    if up_lr_weight != None:
        for i, lr in enumerate(up_lr_weight):
            if lr == 0:
                block_dims[LoRANetwork.NUM_OF_BLOCKS + 1 + i] = 0
                if conv_block_dims is not None:
                    conv_block_dims[LoRANetwork.NUM_OF_BLOCKS + 1 + i] = 0

    return block_dims, block_alphas, conv_block_dims, conv_block_alphas


# 外部から呼び出す可能性を考慮しておく
def get_block_index(lora_name: str) -> int:
    block_idx = -1  # invalid lora name

    m = RE_UPDOWN.search(lora_name)
    if m:
        g = m.groups()
        i = int(g[1])
        j = int(g[3])
        if g[2] == "resnets":
            idx = 3 * i + j
        elif g[2] == "attentions":
            idx = 3 * i + j
        elif g[2] == "upsamplers" or g[2] == "downsamplers":
            idx = 3 * i + 2

        if g[0] == "down":
            block_idx = 1 + idx  # 0に該当するLoRAは存在しない
        elif g[0] == "up":
            block_idx = LoRANetwork.NUM_OF_BLOCKS + 1 + idx

    elif "mid_block_" in lora_name:
        block_idx = LoRANetwork.NUM_OF_BLOCKS  # idx=12

    return block_idx


class LoRANetwork(torch.nn.Module):
    NUM_OF_BLOCKS = 12  # フルモデル相当でのup,downの層の数

    UNET_TARGET_REPLACE_MODULE = ["Transformer2DModel"]
    UNET_TARGET_REPLACE_MODULE_CONV2D_3X3 = [
        "ResnetBlock2D",
        "Downsample2D",
        "Upsample2D",
    ]
    TEXT_ENCODER_TARGET_REPLACE_MODULE = ["CLIPAttention", "CLIPMLP"]
    LORA_PREFIX_UNET = "lora_unet"
    LORA_PREFIX_TEXT_ENCODER = "lora_te"

    # SDXL: must starts with LORA_PREFIX_TEXT_ENCODER
    LORA_PREFIX_TEXT_ENCODER1 = "lora_te1"
    LORA_PREFIX_TEXT_ENCODER2 = "lora_te2"

    def __init__(
        self,
        text_encoder: Union[List[CLIPTextModel], CLIPTextModel],
        unet,
        multiplier: float = 1.0,
        lora_dim: int = 4,
        alpha: float = 1,
        dropout: Optional[float] = None,
        rank_dropout: Optional[float] = None,
        module_dropout: Optional[float] = None,
        conv_lora_dim: Optional[int] = None,
        conv_alpha: Optional[float] = None,
        block_dims: Optional[List[int]] = None,
        block_alphas: Optional[List[float]] = None,
        conv_block_dims: Optional[List[int]] = None,
        conv_block_alphas: Optional[List[float]] = None,
        modules_dim: Optional[Dict[str, int]] = None,
        modules_alpha: Optional[Dict[str, int]] = None,
        module_class: Type[object] = LoRAModule,
        varbose: Optional[bool] = False,
    ) -> None:
        """
        LoRA network: すごく引数が多いが、パターンは以下の通り
        1. lora_dimとalphaを指定
        2. lora_dim、alpha、conv_lora_dim、conv_alphaを指定
        3. block_dimsとblock_alphasを指定 :  Conv2d3x3には適用しない
        4. block_dims、block_alphas、conv_block_dims、conv_block_alphasを指定 : Conv2d3x3にも適用する
        5. modules_dimとmodules_alphaを指定 (推論用)
        """
        super().__init__()
        self.multiplier = multiplier

        self.lora_dim = lora_dim
        self.alpha = alpha
        self.conv_lora_dim = conv_lora_dim
        self.conv_alpha = conv_alpha
        self.dropout = dropout
        self.rank_dropout = rank_dropout
        self.module_dropout = module_dropout

        if modules_dim is not None:
            print(f"create LoRA network from weights")
        elif block_dims is not None:
            print(f"create LoRA network from block_dims")
            print(
                f"neuron dropout: p={self.dropout}, rank dropout:"
                f" p={self.rank_dropout}, module dropout:"
                f" p={self.module_dropout}"
            )
            print(f"block_dims: {block_dims}")
            print(f"block_alphas: {block_alphas}")
            if conv_block_dims is not None:
                print(f"conv_block_dims: {conv_block_dims}")
                print(f"conv_block_alphas: {conv_block_alphas}")
        else:
            print(
                f"create LoRA network. base dim (rank): {lora_dim}, alpha:"
                f" {alpha}"
            )
            print(
                f"neuron dropout: p={self.dropout}, rank dropout:"
                f" p={self.rank_dropout}, module dropout:"
                f" p={self.module_dropout}"
            )
            if self.conv_lora_dim is not None:
                print(
                    "apply LoRA to Conv2d with kernel size (3,3). dim (rank):"
                    f" {self.conv_lora_dim}, alpha: {self.conv_alpha}"
                )

        # create module instances
        def create_modules(
            is_unet: bool,
            text_encoder_idx: Optional[int],  # None, 1, 2
            root_module: torch.nn.Module,
            target_replace_modules: List[torch.nn.Module],
        ) -> List[LoRAModule]:
            prefix = (
                self.LORA_PREFIX_UNET
                if is_unet
                else (
                    self.LORA_PREFIX_TEXT_ENCODER
                    if text_encoder_idx is None
                    else (
                        self.LORA_PREFIX_TEXT_ENCODER1
                        if text_encoder_idx == 1
                        else self.LORA_PREFIX_TEXT_ENCODER2
                    )
                )
            )
            loras = []
            skipped = []
            for name, module in root_module.named_modules():
                if module.__class__.__name__ in target_replace_modules:
                    for child_name, child_module in module.named_modules():
                        is_linear = child_module.__class__.__name__ == "Linear"
                        is_conv2d = child_module.__class__.__name__ == "Conv2d"
                        is_conv2d_1x1 = (
                            is_conv2d and child_module.kernel_size == (1, 1)
                        )

                        if is_linear or is_conv2d:
                            lora_name = prefix + "." + name + "." + child_name
                            lora_name = lora_name.replace(".", "_")

                            dim = None
                            alpha = None

                            if modules_dim is not None:
                                # モジュール指定あり
                                if lora_name in modules_dim:
                                    dim = modules_dim[lora_name]
                                    alpha = modules_alpha[lora_name]
                            elif is_unet and block_dims is not None:
                                # U-Netでblock_dims指定あり
                                block_idx = get_block_index(lora_name)
                                if is_linear or is_conv2d_1x1:
                                    dim = block_dims[block_idx]
                                    alpha = block_alphas[block_idx]
                                elif conv_block_dims is not None:
                                    dim = conv_block_dims[block_idx]
                                    alpha = conv_block_alphas[block_idx]
                            else:
                                # 通常、すべて対象とする
                                if is_linear or is_conv2d_1x1:
                                    dim = self.lora_dim
                                    alpha = self.alpha
                                elif self.conv_lora_dim is not None:
                                    dim = self.conv_lora_dim
                                    alpha = self.conv_alpha

                            if dim is None or dim == 0:
                                # skipした情報を出力
                                if (
                                    is_linear
                                    or is_conv2d_1x1
                                    or (
                                        self.conv_lora_dim is not None
                                        or conv_block_dims is not None
                                    )
                                ):
                                    skipped.append(lora_name)
                                continue

                            lora = module_class(
                                lora_name,
                                child_module,
                                self.multiplier,
                                dim,
                                alpha,
                                dropout=dropout,
                                rank_dropout=rank_dropout,
                                module_dropout=module_dropout,
                            )
                            loras.append(lora)
            return loras, skipped

        text_encoders = (
            text_encoder if type(text_encoder) == list else [text_encoder]
        )

        # create LoRA for text encoder
        # 毎回すべてのモジュールを作るのは無駄なので要検討
        self.text_encoder_loras = []
        skipped_te = []
        for i, text_encoder in enumerate(text_encoders):
            if len(text_encoders) > 1:
                index = i + 1
                print(f"create LoRA for Text Encoder {index}:")
            else:
                index = None
                print(f"create LoRA for Text Encoder:")

            text_encoder_loras, skipped = create_modules(
                False,
                index,
                text_encoder,
                LoRANetwork.TEXT_ENCODER_TARGET_REPLACE_MODULE,
            )
            self.text_encoder_loras.extend(text_encoder_loras)
            skipped_te += skipped
        print(
            "create LoRA for Text Encoder:"
            f" {len(self.text_encoder_loras)} modules."
        )

        # extend U-Net target modules if conv2d 3x3 is enabled, or load from weights
        target_modules = LoRANetwork.UNET_TARGET_REPLACE_MODULE
        if (
            modules_dim is not None
            or self.conv_lora_dim is not None
            or conv_block_dims is not None
        ):
            target_modules += LoRANetwork.UNET_TARGET_REPLACE_MODULE_CONV2D_3X3

        self.unet_loras, skipped_un = create_modules(
            True, None, unet, target_modules
        )
        print(f"create LoRA for U-Net: {len(self.unet_loras)} modules.")

        skipped = skipped_te + skipped_un
        if varbose and len(skipped) > 0:
            print(
                "because block_lr_weight is 0 or dim (rank) is 0,"
                f" {len(skipped)} LoRA modules are skipped /"
                " block_lr_weightまたはdim (rank)が0の為、"
                f"次の{len(skipped)}個のLoRAモジュールはスキップされます:"
            )
            for name in skipped:
                print(f"\t{name}")

        self.up_lr_weight: List[float] = None
        self.down_lr_weight: List[float] = None
        self.mid_lr_weight: float = None
        self.block_lr = False

        # assertion
        names = set()
        for lora in self.text_encoder_loras + self.unet_loras:
            assert (
                lora.lora_name not in names
            ), f"duplicated lora name: {lora.lora_name}"
            names.add(lora.lora_name)

    def set_multiplier(self, multiplier):
        self.multiplier = multiplier
        for lora in self.text_encoder_loras + self.unet_loras:
            lora.multiplier = self.multiplier

    def load_weights(self, file):
        if os.path.splitext(file)[1] == ".safetensors":
            from safetensors.torch import load_file

            weights_sd = load_file(file)
        else:
            weights_sd = torch.load(file, map_location="cpu")
        info = self.load_state_dict(weights_sd, False)
        return info

    def apply_to(
        self, text_encoder, unet, apply_text_encoder=True, apply_unet=True
    ):
        if apply_text_encoder:
            print("enable LoRA for text encoder")
        else:
            self.text_encoder_loras = []

        if apply_unet:
            print("enable LoRA for U-Net")
        else:
            self.unet_loras = []

        for lora in self.text_encoder_loras + self.unet_loras:
            lora.apply_to()
            self.add_module(lora.lora_name, lora)

    # マージできるかどうかを返す
    def is_mergeable(self):
        return True

    # TODO refactor to common function with apply_to
    def merge_to(self, text_encoder, unet, weights_sd, dtype, device):
        apply_text_encoder = apply_unet = False
        for key in weights_sd.keys():
            if key.startswith(LoRANetwork.LORA_PREFIX_TEXT_ENCODER):
                apply_text_encoder = True
            elif key.startswith(LoRANetwork.LORA_PREFIX_UNET):
                apply_unet = True

        if apply_text_encoder:
            print("enable LoRA for text encoder")
        else:
            self.text_encoder_loras = []

        if apply_unet:
            print("enable LoRA for U-Net")
        else:
            self.unet_loras = []

        for lora in self.text_encoder_loras + self.unet_loras:
            sd_for_lora = {}
            for key in weights_sd.keys():
                if key.startswith(lora.lora_name):
                    sd_for_lora[key[len(lora.lora_name) + 1 :]] = weights_sd[
                        key
                    ]
            lora.merge_to(sd_for_lora, dtype, device)

        print(f"weights are merged")

    # 層別学習率用に層ごとの学習率に対する倍率を定義する　引数の順番が逆だがとりあえず気にしない
    def set_block_lr_weight(
        self,
        up_lr_weight: List[float] = None,
        mid_lr_weight: float = None,
        down_lr_weight: List[float] = None,
    ):
        self.block_lr = True
        self.down_lr_weight = down_lr_weight
        self.mid_lr_weight = mid_lr_weight
        self.up_lr_weight = up_lr_weight

    def get_lr_weight(self, lora: LoRAModule) -> float:
        lr_weight = 1.0
        block_idx = get_block_index(lora.lora_name)
        if block_idx < 0:
            return lr_weight

        if block_idx < LoRANetwork.NUM_OF_BLOCKS:
            if self.down_lr_weight != None:
                lr_weight = self.down_lr_weight[block_idx]
        elif block_idx == LoRANetwork.NUM_OF_BLOCKS:
            if self.mid_lr_weight != None:
                lr_weight = self.mid_lr_weight
        elif block_idx > LoRANetwork.NUM_OF_BLOCKS:
            if self.up_lr_weight != None:
                lr_weight = self.up_lr_weight[
                    block_idx - LoRANetwork.NUM_OF_BLOCKS - 1
                ]

        return lr_weight

    # 二つのText Encoderに別々の学習率を設定できるようにするといいかも
    def prepare_optimizer_params(self, text_encoder_lr, unet_lr, default_lr):
        self.requires_grad_(True)
        all_params = []

        def enumerate_params(loras):
            params = []
            for lora in loras:
                params.extend(lora.parameters())
            return params

        if self.text_encoder_loras:
            param_data = {"params": enumerate_params(self.text_encoder_loras)}
            if text_encoder_lr is not None:
                param_data["lr"] = text_encoder_lr
            all_params.append(param_data)

        if self.unet_loras:
            if self.block_lr:
                # 学習率のグラフをblockごとにしたいので、blockごとにloraを分類
                block_idx_to_lora = {}
                for lora in self.unet_loras:
                    idx = get_block_index(lora.lora_name)
                    if idx not in block_idx_to_lora:
                        block_idx_to_lora[idx] = []
                    block_idx_to_lora[idx].append(lora)

                # blockごとにパラメータを設定する
                for idx, block_loras in block_idx_to_lora.items():
                    param_data = {"params": enumerate_params(block_loras)}

                    if unet_lr is not None:
                        param_data["lr"] = unet_lr * self.get_lr_weight(
                            block_loras[0]
                        )
                    elif default_lr is not None:
                        param_data["lr"] = default_lr * self.get_lr_weight(
                            block_loras[0]
                        )
                    if ("lr" in param_data) and (param_data["lr"] == 0):
                        continue
                    all_params.append(param_data)

            else:
                param_data = {"params": enumerate_params(self.unet_loras)}
                if unet_lr is not None:
                    param_data["lr"] = unet_lr
                all_params.append(param_data)

        return all_params

    def enable_gradient_checkpointing(self):
        # not supported
        pass

    def prepare_grad_etc(self, text_encoder, unet):
        self.requires_grad_(True)

    def on_epoch_start(self, text_encoder, unet):
        self.train()

    def get_trainable_params(self):
        return self.parameters()

    def save_weights(self, file, dtype, metadata):
        if metadata is not None and len(metadata) == 0:
            metadata = None

        state_dict = self.state_dict()

        if dtype is not None:
            for key in list(state_dict.keys()):
                v = state_dict[key]
                v = v.detach().clone().to("cpu").to(dtype)
                state_dict[key] = v

        if os.path.splitext(file)[1] == ".safetensors":
            from safetensors.torch import save_file

            # TODO(ryand): Pre-calculate the safetensors model hash to save time
            # on indexing the model later. See this implementation for details:
            # https://github.com/bmaltais/kohya_ss/blob/397bf51a8cd36104e52055358e4ffd066c5858df/networks/lora.py#L1092

            save_file(state_dict, file, metadata)
        else:
            torch.save(state_dict, file)

    # mask is a tensor with values from 0 to 1
    def set_region(self, sub_prompt_index, is_last_network, mask):
        if mask.max() == 0:
            mask = torch.ones_like(mask)

        self.mask = mask
        self.sub_prompt_index = sub_prompt_index
        self.is_last_network = is_last_network

        for lora in self.text_encoder_loras + self.unet_loras:
            lora.set_network(self)

    def set_current_generation(
        self, batch_size, num_sub_prompts, width, height, shared
    ):
        self.batch_size = batch_size
        self.num_sub_prompts = num_sub_prompts
        self.current_size = (height, width)
        self.shared = shared

        # create masks
        mask = self.mask
        mask_dic = {}
        mask = mask.unsqueeze(0).unsqueeze(1)  # b(1),c(1),h,w
        ref_weight = (
            self.text_encoder_loras[0].lora_down.weight
            if self.text_encoder_loras
            else self.unet_loras[0].lora_down.weight
        )
        dtype = ref_weight.dtype
        device = ref_weight.device

        def resize_add(mh, mw):
            # print(mh, mw, mh * mw)
            m = torch.nn.functional.interpolate(
                mask, (mh, mw), mode="bilinear"
            )  # doesn't work in bf16
            m = m.to(device, dtype=dtype)
            mask_dic[mh * mw] = m

        h = height // 8
        w = width // 8
        for _ in range(4):
            resize_add(h, w)
            if (
                h % 2 == 1 or w % 2 == 1
            ):  # add extra shape if h/w is not divisible by 2
                resize_add(h + h % 2, w + w % 2)
            h = (h + 1) // 2
            w = (w + 1) // 2

        self.mask_dic = mask_dic

    def apply_max_norm_regularization(self, max_norm_value, device):
        downkeys = []
        upkeys = []
        alphakeys = []
        norms = []
        keys_scaled = 0

        state_dict = self.state_dict()
        for key in state_dict.keys():
            if "lora_down" in key and "weight" in key:
                downkeys.append(key)
                upkeys.append(key.replace("lora_down", "lora_up"))
                alphakeys.append(key.replace("lora_down.weight", "alpha"))

        for i in range(len(downkeys)):
            down = state_dict[downkeys[i]].to(device)
            up = state_dict[upkeys[i]].to(device)
            alpha = state_dict[alphakeys[i]].to(device)
            dim = down.shape[0]
            scale = alpha / dim

            if up.shape[2:] == (1, 1) and down.shape[2:] == (1, 1):
                updown = (
                    (up.squeeze(2).squeeze(2) @ down.squeeze(2).squeeze(2))
                    .unsqueeze(2)
                    .unsqueeze(3)
                )
            elif up.shape[2:] == (3, 3) or down.shape[2:] == (3, 3):
                updown = torch.nn.functional.conv2d(
                    down.permute(1, 0, 2, 3), up
                ).permute(1, 0, 2, 3)
            else:
                updown = up @ down

            updown *= scale

            norm = updown.norm().clamp(min=max_norm_value / 2)
            desired = torch.clamp(norm, max=max_norm_value)
            ratio = desired.cpu() / norm.cpu()
            sqrt_ratio = ratio**0.5
            if ratio != 1:
                keys_scaled += 1
                state_dict[upkeys[i]] *= sqrt_ratio
                state_dict[downkeys[i]] *= sqrt_ratio
            scalednorm = updown.norm() * ratio
            norms.append(scalednorm.item())

        return keys_scaled, sum(norms) / len(norms), max(norms)
