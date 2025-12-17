"""Typed dataclasses for transporting per-scale training metrics.

The dataclasses in this module are used to return tensor-valued losses and
metrics from model optimization routines. Fields intentionally remain as
``torch.Tensor`` scalars during the forward/backward pass so autograd is
preserved; callers should convert to Python floats (``.item()``) when logging
or writing to external sinks.
"""

from dataclasses import dataclass


@dataclass
class DiscriminatorMetrics[TTensor]:
    """Per-scale discriminator metric container.

    Fields are scalar ``torch.Tensor`` values representing the total loss,
    real/fake component losses and gradient penalty (``gp``).
    """

    total: TTensor
    real: TTensor
    fake: TTensor
    gp: TTensor


@dataclass
class GeneratorMetrics[TTensor]:
    """Per-scale generator metric container.

    Fields are scalar ``torch.Tensor`` values for adversarial (``fake``),
    reconstruction (``rec``), well-constraint (``well``), diversity (``div``)
    and aggregated total loss (``total``).
    """

    total: TTensor
    fake: TTensor
    rec: TTensor
    well: TTensor
    div: TTensor


@dataclass
class ScaleMetrics[TTensor]:
    """Mapping of scale index to per-scale metric dataclasses.

    ``generator`` and ``discriminator`` map integer scale indices to the
    corresponding metric dataclasses defined above.
    """

    generator: dict[int, GeneratorMetrics[TTensor]]
    discriminator: dict[int, DiscriminatorMetrics[TTensor]]
