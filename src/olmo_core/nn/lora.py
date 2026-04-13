"""
LoRA (Low-Rank Adaptation) for OLMo-core models.

Provides a minimal LoRA implementation compatible with OLMo-core's transformer
architecture and FSDP training. Replaces target nn.Linear modules with LoRALinear
wrappers that freeze base weights and add trainable low-rank A/B matrices.

Usage:
    model = config.build(init_device="meta")
    apply_lora(model, r=64, alpha=128, target_modules=["q_proj", "k_proj", ...])
    # Then wrap with FSDP, load checkpoint, train as usual
"""

import math
from dataclasses import dataclass, field
from typing import List, Set

import torch
import torch.nn as nn

from olmo_core.config import Config


@dataclass
class LoRAConfig(Config):
    """Configuration for LoRA adaptation."""

    r: int = 64
    """LoRA rank."""

    alpha: float = 128.0
    """LoRA scaling factor. Effective scaling = alpha / r."""

    dropout: float = 0.0
    """Dropout applied to input before LoRA path."""

    target_modules: List[str] = field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    )
    """Names of linear modules to apply LoRA to (matched against the final component of the parameter name)."""


class LoRALinear(nn.Module):
    """
    A linear layer with frozen base weights and trainable low-rank A/B matrices.

    output = base(x) + lora_B(lora_A(dropout(x))) * scaling
    """

    def __init__(self, base_layer: nn.Linear, r: int, alpha: float, dropout: float = 0.0):
        super().__init__()
        self.base_layer = base_layer
        self.r = r
        self.scaling = alpha / r

        # Freeze base weights
        self.base_layer.weight.requires_grad_(False)
        if self.base_layer.bias is not None:
            self.base_layer.bias.requires_grad_(False)

        # Detect if base layer is on meta device
        device = base_layer.weight.device

        # Low-rank matrices (created on same device as base, including meta)
        self.lora_A = nn.Linear(base_layer.in_features, r, bias=False, device=device)
        self.lora_B = nn.Linear(r, base_layer.out_features, bias=False, device=device)

        # Dropout
        self.lora_dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

        # Initialize if not on meta device (meta tensors are initialized later during materialization)
        if device.type != "meta":
            self.reset_lora_parameters()

    def reset_lora_parameters(self):
        """Initialize LoRA parameters: Kaiming for A, zeros for B."""
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base_layer(x)
        lora_out = self.lora_B(self.lora_A(self.lora_dropout(x)))
        return base_out + lora_out * self.scaling

    def merge(self) -> nn.Linear:
        """Merge LoRA weights into base layer and return a plain nn.Linear."""
        with torch.no_grad():
            delta = (self.lora_B.weight @ self.lora_A.weight) * self.scaling
            self.base_layer.weight.add_(delta.to(self.base_layer.weight.dtype))
        return self.base_layer


def apply_lora(model: nn.Module, config: LoRAConfig) -> Set[str]:
    """
    Apply LoRA to target modules in the model.

    Walks the module tree, replaces matching nn.Linear layers with LoRALinear,
    and freezes all non-LoRA parameters.

    Returns the set of module names that were replaced.
    """
    replaced = set()

    # First freeze everything
    for param in model.parameters():
        param.requires_grad_(False)

    # Then replace target modules with LoRA wrappers
    for name, module in list(model.named_modules()):
        # Check if this module's name ends with any target module name
        module_name = name.split(".")[-1]
        if module_name not in config.target_modules:
            continue
        if not isinstance(module, nn.Linear):
            continue

        # Create LoRA wrapper
        lora_layer = LoRALinear(
            base_layer=module,
            r=config.r,
            alpha=config.alpha,
            dropout=config.dropout,
        )

        # Replace in parent
        parent_name = ".".join(name.split(".")[:-1])
        parent = model.get_submodule(parent_name) if parent_name else model
        setattr(parent, module_name, lora_layer)
        replaced.add(name)

    return replaced


def merge_lora(model: nn.Module) -> None:
    """Merge all LoRA weights into base layers (for inference/saving)."""
    for name, module in list(model.named_modules()):
        if isinstance(module, LoRALinear):
            merged = module.merge()
            parent_name = ".".join(name.split(".")[:-1])
            module_name = name.split(".")[-1]
            parent = model.get_submodule(parent_name) if parent_name else model
            setattr(parent, module_name, merged)


def get_lora_state_dict(model: nn.Module) -> dict:
    """Extract only LoRA parameters from the model state dict."""
    lora_state = {}
    for name, param in model.named_parameters():
        if "lora_A" in name or "lora_B" in name:
            lora_state[name] = param.detach().cpu()
    return lora_state


def print_trainable_parameters(model: nn.Module) -> None:
    """Print the number of trainable vs total parameters."""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    pct = trainable / total * 100 if total > 0 else 0
    print(f"Trainable params: {trainable:,} / {total:,} ({pct:.2f}%)")
