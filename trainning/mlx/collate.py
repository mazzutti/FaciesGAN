# from typing import Iterator

from typing import Iterator
from typedefs import Batch
import mlx.core as mx


def collate(samples: list[Batch[mx.array]]) -> Batch[mx.array]:
    """
    Fast collate function for MLX arrays with full type annotations.
    """
    if not samples:
        return Batch(facies=(), wells=(), seismic=())

    # 1. Transpose: list of Batch objects -> Batch object of tuples
    # 'batch_struct' holds the data grouped by sample.
    # e.g., batch_struct.facies is a tuple of size `batch_size`, containing
    # the facies tuples from each sample.
    batch_struct: Batch[mx.array] = Batch(*zip(*samples))

    # 2. Process Facies
    # Transpose from (Batch, Scales) -> (Scales, Batch)
    facies_by_scale: Iterator[tuple[mx.array, ...]] = zip(*batch_struct.facies)

    batched_facies: tuple[mx.array, ...] = tuple(
        mx.stack(list(scale_group)) for scale_group in facies_by_scale
    )

    # 3. Process Wells
    batched_wells: tuple[mx.array, ...]

    # Check if wells exist and if the first sample has well data
    if batch_struct.wells and batch_struct.wells[0]:
        wells_by_scale: Iterator[tuple[mx.array, ...]] = zip(*batch_struct.wells)
        batched_wells = tuple(
            mx.stack(list(scale_group)) for scale_group in wells_by_scale
        )
    else:
        batched_wells = ()

    # 4. Process Seismic
    batched_seismic: tuple[mx.array, ...]

    if batch_struct.seismic and batch_struct.seismic[0]:
        seismic_by_scale: Iterator[tuple[mx.array, ...]] = zip(*batch_struct.seismic)
        batched_seismic = tuple(
            mx.stack(list(scale_group)) for scale_group in seismic_by_scale
        )
    else:
        batched_seismic = ()

    return Batch(facies=batched_facies, wells=batched_wells, seismic=batched_seismic)
