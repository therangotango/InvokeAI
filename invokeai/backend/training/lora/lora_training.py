from invokeai.app.services.config import InvokeAIAppConfig
from invokeai.backend.training.lora.lora_training_config import LoraTrainingConfig
import invokeai.backend.util.logging as logger


def run_lora_training(app_config: InvokeAIAppConfig, train_config: LoraTrainingConfig):
    logger.info("Starting LoRA Training.")
    logger.info(f"Model version: {train_config.model}")
