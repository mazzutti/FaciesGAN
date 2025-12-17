from typing import Generic, NamedTuple, Tuple

from typedefs import TTensor


class Batch(NamedTuple, Generic[TTensor]):
    """Container for a training batch of multi-scale pyramids.

    Fields are per-scale tuples of tensors. `Batch` is a NamedTuple so it
    subclasses `tuple` and is compatible with PyTorch's default collate.


    Parameters
    ----------
    facies : Tuple[TTensor, ...]
        Per-scale facies tensors.
    wells : Tuple[TTensor, ...] | tuple[()]
        Per-scale well-conditioning tensors (may be empty when unused).
    seismic : Tuple[TTensor, ...] | tuple[()]
        Per-scale seismic-conditioning tensors (may be empty when unused).
    """

    facies: Tuple[TTensor, ...]
    wells: Tuple[TTensor, ...] | tuple[()]
    seismic: Tuple[TTensor, ...] | tuple[()]
