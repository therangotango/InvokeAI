from pathlib import Path
from pydantic import BaseModel
import typing


class LoraTrainingConfig(BaseModel):
    """The configuration for a LoRA training run."""

    # The output directory where the training outputs (model checkpoints, logs,
    # intermediate predictions) will be written.
    output_dir: Path

    # The name of the diffusers model to train against, as defined in
    # 'configs/models.yaml'.
    model: str = "sd-1/main/stable-diffusion-v1-5"

    # The number of gradient steps to accumulate before each weight update. This
    # value is passed to Hugging Face Accelerate. This is an alternative to
    # increasing the batch size when training with limited VRAM.
    gradient_accumulation_steps: int = 1

    # The mixed precision mode to use ('no','fp16','bf16 or 'fp8'). This value
    # is passed to Hugging Face Accelerate. See accelerate.Accelerator for more
    # details.
    mixed_precision: typing.Optional[
        typing.Literal["no", "fp16", "bf16", "fp8"]
    ] = None

    # The integration to report results and logs to ('all', 'tensorboard',
    # 'wandb', or 'comet_ml'). This value is passed to Hugging Face Accelerate.
    # See accelerate.Accelerator.log_with for more details.
    report_to: typing.Optional[
        typing.Literal["all", "tensorboard", "wandb", "comet_ml"]
    ] = "tensorboard"
