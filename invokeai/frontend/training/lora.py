from invokeai.app.services.config import InvokeAIAppConfig
from invokeai.backend.training.lora_training import parse_args, run_lora_training


def main():
    config = InvokeAIAppConfig.get_config()
    args = parse_args()

    run_lora_training(config, **vars(args))


if __name__ == "__main__":
    main()
