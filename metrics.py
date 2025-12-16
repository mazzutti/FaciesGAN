"""Typed dataclasses for transporting per-scale training metrics.

The dataclasses in this module are used to return tensor-valued losses and
metrics from model optimization routines. Fields intentionally remain as
``torch.Tensor`` scalars during the forward/backward pass so autograd is
preserved; callers should convert to Python floats (``.item()``) when logging
or writing to external sinks.
"""

from dataclasses import dataclass
import torch


@dataclass
class DiscriminatorMetrics:
    """Per-scale discriminator metric container.

    Fields are scalar ``torch.Tensor`` values representing the total loss,
    real/fake component losses and gradient penalty (``gp``).
    """

    total: torch.Tensor
    real: torch.Tensor
    fake: torch.Tensor
    gp: torch.Tensor


@dataclass
class GeneratorMetrics:
    """Per-scale generator metric container.

    Fields are scalar ``torch.Tensor`` values for adversarial (``fake``),
    reconstruction (``rec``), well-constraint (``well``), diversity (``div``)
    and aggregated total loss (``total``).
    """

    total: torch.Tensor
    fake: torch.Tensor
    rec: torch.Tensor
    well: torch.Tensor
    div: torch.Tensor


@dataclass
class ScaleMetrics:
    """Mapping of scale index to per-scale metric dataclasses.

    ``generator`` and ``discriminator`` map integer scale indices to the
    corresponding metric dataclasses defined above.
    """

    generator: dict[int, GeneratorMetrics]
    discriminator: dict[int, DiscriminatorMetrics]
