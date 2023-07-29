import math

import torch


class LoRALinearLayer(torch.nn.Module):
    """An implementation of a linear LoRA layer based on the paper 'LoRA: Low-Rank Adaptation of Large Language Models'.
    (https://arxiv.org/pdf/2106.09685.pdf)
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 4,
        alpha: float = 1.0,
        device: torch.device = None,
        dtype: torch.dtype = None,
    ):
        """Initialize a LoRALinearLayer.

        Args:
            in_features (int): Inputs to this layer will be expected to have shape (..., in_features).
            out_features (int): This layer will produce outputs with shape (..., out_features).
            rank (int, optional): The internal rank of the layer (see the paper for details).
            alpha (float, optional): A scaling factor that enables tuning the rank without having to adjust the learning
                rate. The recommendation from the paper is to set alpha equal to the first rank that you try and then do
                not tune it further. See the paper for more details.
            device (torch.device, optional): Device where weights will be initialized.
            dtype (torch.dtype, optional): Weight dtype.

        Raises:
            ValueError: If the rank is greater than either in_features or out_features.
        """
        super().__init__()

        if rank > min(in_features, out_features):
            raise ValueError(f"LoRA rank {rank} must be less than or equal to {min(in_features, out_features)}")

        self.down = torch.nn.Linear(in_features, rank, bias=False, device=device, dtype=dtype)
        self.up = torch.nn.Linear(rank, out_features, bias=False, device=device, dtype=dtype)

        self.alpha = alpha
        self.rank = rank

        self.reset_parameters()

    def reset_parameters(self):
        # This initialization is based on Microsoft's implementation:
        # https://github.com/microsoft/LoRA/blob/998cfe4d351f4d6b4a47f0921dec2397aa0b9dfe/loralib/layers.py#L123
        torch.nn.init.kaiming_uniform_(self.down.weight, a=math.sqrt(5))
        torch.nn.init.zeros_(self.up.weight)

    def forward(self, input: torch.Tensor):
        down_hidden = self.down(input)
        up_hidden = self.up(down_hidden)

        up_hidden *= self.alpha / self.rank

        return up_hidden
