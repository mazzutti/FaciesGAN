"""Typed dataclasses for transporting per-scale training metrics.

The dataclasses in this module are used to return tensor-valued losses and
metrics from model optimization routines. Fields intentionally remain as
``torch.Tensor`` scalars during the forward/backward pass so autograd is
preserved; callers should convert to Python floats (``.item()``) when logging
or writing to external sinks.
"""

from __future__ import annotations
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

    def as_dict(self) -> dict[str, float]:
        """Return the metric values as a dictionary.

        Returns
        -------
        dict[str, TTensor]
            Dictionary mapping metric names to their tensor values.
        """
        return {
            "d_total": self.total.item(),  # type: ignore
            "d_real": self.real.item(),  # type: ignore
            "d_fake": self.fake.item(),  # type: ignore
            "d_gp": self.gp.item(),  # type: ignore
        }

    def as_tuple(self) -> tuple[TTensor, ...]:
        """Return the metric values as a tuple in a fixed order.

        Returns
        -------
        tuple[TTensor, ...]
            Tuple of metric values in the order:
            (total, real, fake, gp).
        """
        return (self.total, self.real, self.fake, self.gp)


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

    def as_dict(self) -> dict[str, float]:
        """Return the metric values as a dictionary.

        Returns
        -------
        dict[str, TTensor]
            Dictionary mapping metric names to their tensor values.
        """
        return {
            "g_total": self.total.item(),  # type: ignore
            "g_fake": self.fake.item(),  # type: ignore
            "g_rec": self.rec.item(),  # type: ignore
            "g_well": self.well.item(),  # type: ignore
            "g_div": self.div.item(),  # type: ignore
        }

    def as_tuple(self) -> tuple[TTensor, ...]:
        """Return the metric values as a list in a fixed order.

        Returns
        -------
        tuple[TTensor, ...]
            Tuple of metric values in the order:
            (total, fake, rec, well, div).
        """
        return (self.total, self.fake, self.rec, self.well, self.div)


@dataclass
class ScaleMetrics[TTensor]:
    """Mapping of scale index to per-scale metric dataclasses.

    ``generator`` and ``discriminator`` map integer scale indices to the
    corresponding metric dataclasses defined above.
    """

    generator: tuple[GeneratorMetrics[TTensor], ...]
    discriminator: tuple[DiscriminatorMetrics[TTensor], ...]

    @staticmethod
    def from_dicts(
        gen_dict: dict[str, list[TTensor]],
        disc_dict: dict[str, list[TTensor]],
    ) -> ScaleMetrics[TTensor]:
        """Construct a ``ScaleMetrics`` instance from per-scale metric dicts.

        Parameters
        ----------
        gen_dict : dict[int, GeneratorMetrics[TTensor]]
            Mapping of scale index to generator metrics.
        disc_dict : dict[int, DiscriminatorMetrics[TTensor]]
            Mapping of scale index to discriminator metrics.

        Returns
        -------
        ScaleMetrics[TTensor]
            Constructed ``ScaleMetrics`` instance.
        """
        return ScaleMetrics(
            generator=tuple(
                GeneratorMetrics(*metrics) for _, metrics in gen_dict.items()
            ),
            discriminator=tuple(
                DiscriminatorMetrics(*metrics) for _, metrics in disc_dict.items()
            ),
        )

    def as_tuple_of_dicts(self) -> tuple[dict[str, float], ...]:
        """Return all metric values as a tuple of dictionaries in a fixed order.

        The order is by scale index (ascending), with generator metrics
        preceding discriminator metrics at each scale.

        Returns
        -------
        tuple[dict[str, float], ...]
            Tuple of dictionaries mapping metric names to their values in the order:
            (gen_scale0..., disc_scale0..., gen_scale1..., disc_scale1..., ...).
        """
        return tuple(
            x
            for scale in range(len(self.generator))
            for x in (
                self.generator[scale].as_dict(),
                self.discriminator[scale].as_dict(),
            )
        )

    def as_tuple(self) -> tuple[TTensor, ...]:
        """Return all metric values as a flat list in a fixed order.

        The order is by scale index (ascending), with generator metrics
        preceding discriminator metrics at each scale.

        Returns
        -------
        tuple[TTensor, ...  ]
            Flat tuple of all metric values in the order:
            (gen_scale0..., disc_scale0..., gen_scale1..., disc_scale1..., ...).
        #"""
        return tuple(
            x
            for scale in range(len(self.generator))
            for x in (
                *self.generator[scale].as_tuple(),
                *self.discriminator[scale].as_tuple(),
            )
        )
