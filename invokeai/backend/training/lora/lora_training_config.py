import typing
from pathlib import Path

from pydantic import BaseModel


class LoraTrainingConfig(BaseModel):
    """The configuration for a LoRA training run."""

    ##################
    # Output Configs
    ##################

    # The output directory where the training outputs (model checkpoints, logs,
    # intermediate predictions) will be written.
    output_dir: str

    # The integration to report results and logs to ('all', 'tensorboard',
    # 'wandb', or 'comet_ml'). This value is passed to Hugging Face Accelerate.
    # See accelerate.Accelerator.log_with for more details.
    report_to: typing.Optional[
        typing.Literal["all", "tensorboard", "wandb", "comet_ml"]
    ] = "tensorboard"

    ##################
    # General Configs
    ##################

    # The name of the diffusers model to train against, as defined in
    # 'configs/models.yaml'.
    model: str = "sd-1/main/stable-diffusion-v1-5"

    # A seed for reproducible training.
    seed: typing.Optional[int] = None

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

    # If true, use xformers for more efficient attention blocks.
    xformers: bool = False

    # Whether or not to use gradient checkpointing to save memory at the expense
    # of a slower backward pass.
    gradient_checkpointing: bool = False

    # Total number of training steps to perform.
    max_train_steps: int = 5000

    #####################
    # Optimizer Configs
    #####################

    # Initial learning rate (after the potential warmup period) to use.
    learning_rate: float = 1e-4

    # Adam optimizer params.
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_weight_decay: float = 1e-2
    adam_epsilon: float = 1e-8

    # The number of warmup steps in the learning rate scheduler. Only applied to
    # schedulers that support warmup. See lr_scheduler.
    lr_warmup_steps: int = 0

    # The learning rate scheduler to use.
    lr_scheduler: typing.Literal[
        "linear",
        "cosine",
        "cosine_with_restarts",
        "polynomial",
        "constant",
        "constant_with_warmup",
    ] = "constant"

    ##################
    # Dataset Configs
    ##################

    # The name of a Hugging Face dataset.
    # One of dataset_name and dataset_dir should be set (dataset_name takes
    # precedence).
    # See also: dataset_config_name.
    dataset_name: typing.Optional[str] = None

    # The directory to load a dataset from. The dataset is expected to be in
    # Hugging Face imagefolder format
    # (https://huggingface.co/docs/datasets/v2.4.0/en/image_load#imagefolder).
    # One of dataset_name and dataset_dir should be set (dataset_name takes
    # precedence).
    dataset_dir: typing.Optional[str] = None

    # The Hugging Face dataset config name. Leave as None if there's only one
    # config.
    # This parameter is only used if dataset_name is set.
    dataset_config_name: typing.Optional[str] = None

    # The Hugging Face cache directory to use for dataset downloads.
    # If None, the default value will be used (usually
    # '~/.cache/huggingface/datasets').
    hf_cache_dir: typing.Optional[str] = None

    # The name of the dataset column that contains image paths.
    dataset_image_column: str = "image"

    # The name of the dataset column that contains captions.
    dataset_caption_column: str = "text"

    # The resolution for input images. All of the images in the dataset will be
    # resized to this (square) resolution.
    resolution: int = 512

    # If True, input images will be center-cropped to resolution.
    # If False, input images will be randomly cropped to resolution.
    center_crop: bool = False

    # Whether random flip augmentations should be applied to input images.
    random_flip: bool = False

    # The training batch size.
    train_batch_size: int = 4

    # Number of subprocesses to use for data loading. 0 means that the data will
    # be loaded in the main process.
    dataloader_num_workers: int = 0
