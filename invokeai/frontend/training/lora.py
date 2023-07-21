from invokeai.app.services.config import InvokeAIAppConfig, PagingArgumentParser
from invokeai.backend.training.lora.lora_training import run_lora_training
from invokeai.backend.training.lora.lora_training_config import LoraTrainingConfig


def parse_args():
    parser = PagingArgumentParser(description="LoRA model training.")

    # Base model configs
    model_group = parser.add_argument_group("Model")
    model_group.add_argument(
        "--model",
        type=str,
        required=True,
        help="Name of the diffusers model to train against, as defined in "
        "'configs/models.yaml' (e.g. 'sd-1/main/stable-diffusion-v1-5').",
    )

    return parser.parse_args()


def main():
    app_config = InvokeAIAppConfig.get_config()
    args = parse_args()

    train_config = LoraTrainingConfig(**vars(args))
    run_lora_training(app_config, train_config)


if __name__ == "__main__":
    main()
