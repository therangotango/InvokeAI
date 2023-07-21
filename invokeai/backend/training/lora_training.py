from invokeai.app.services.config import InvokeAIAppConfig, PagingArgumentParser
import invokeai.backend.util.logging as logger


def run_lora_training(config: InvokeAIAppConfig, model: str):
    logger.info("Starting LoRA Training.")
    logger.info(f"Model version: {model}")


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
