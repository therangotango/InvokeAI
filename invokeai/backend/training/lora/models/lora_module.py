import torch


class LoRAModule(torch.nn.Module):
    """A wrapper that combines the outputs of an 'original' module and a parallel 'LoRA' layer.

    Note: This class may be renamed in the future. It is being introduced with the goal of supporting LoRAs, but can
    more generally be used to merge the output of any 2 modules.
    """

    def __init__(self, original_module: torch.nn.Module, lora_layer: torch.nn.Module, lora_multiplier: float = 1.0):
        """Initialize a LoRAModule.

        Args:
            original_module (torch.nn.Module): The original module.
            lora_layer (torch.nn.Module): The LoRA layer.
            lora_multiplier (float, optional): A multiplier applied to the LoRA layer output before adding it to the
            original module output. Defaults to 1.0.
        """
        super().__init__()

        self.original_module = original_module
        self.lora_layer = lora_layer
        self.lora_multiplier = lora_multiplier

    def forward(self, input):
        return self.original_module.forward(input) + self.lora_multiplier * self.lora_layer.forward(input)
