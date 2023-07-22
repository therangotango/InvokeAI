from pathlib import Path

from invokeai.app.services.config import InvokeAIAppConfig, PagingArgumentParser
from invokeai.backend.training.lora.lora_training import run_lora_training
from invokeai.backend.training.lora.lora_training_config import (
    LoraTrainingConfig,
)


def parse_args():
    config = InvokeAIAppConfig.get_config()

    parser = PagingArgumentParser(description="LoRA model training.")

    # General configs
    general_group = parser.add_argument_group("General")
    general_group.add_argument(
        "--output_dir",
        type=Path,
        # TODO(ryand): Decide on a training directory structure and update for
        # consistency with TI training.
        default=config.root / "training/lora/output",
    )

    # Base model configs
    model_group = parser.add_argument_group("Model")
    model_group.add_argument(
        "--model",
        type=str,
        required=True,
        help="Name of the diffusers model to train against, as defined in "
        "'configs/models.yaml' (e.g. 'sd-1/main/stable-diffusion-v1-5').",
    )

    # Training Group
    training_group = parser.add_argument_group("Training")
    training_group.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="The number of gradient steps to accumulate before each weight "
        "update. This value is passed to Hugging Face Accelerate. This is an "
        "alternative to increasing the batch size when training with limited "
        "VRAM.",
    )
    training_group.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        help="The mixed precision mode to use ('no','fp16','bf16 or 'fp8'). "
        "This value is passed to Hugging Face Accelerate. See "
        "accelerate.Accelerator for more details.",
    )
    training_group.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help="The integration to report results and logs to ('all', "
        "'tensorboard', 'wandb', or 'comet_ml'). This value is passed to "
        "Hugging Face Accelerate. See accelerate.Accelerator.log_with for more "
        "details.",
    )

    return parser.parse_args()


def main():
    app_config = InvokeAIAppConfig.get_config()
    args = parse_args()

    train_config = LoraTrainingConfig(**vars(args))
    run_lora_training(app_config, train_config)


if __name__ == "__main__":
    main()
