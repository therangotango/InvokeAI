from pydantic import BaseModel


class LoraTrainingConfig(BaseModel):
    """The configuration for a LoRA training run."""

    # The name of the diffusers model to train against, as defined in
    # 'configs/models.yaml'.
    model: str
