import mlx.core as mx

from typedefs import Batch


def collate(
    samples: list[Batch[mx.array]],
) -> Batch[mx.array]:
    """Collate function for MLX arrays returned by the dataset.

    Stacks per-scale MLX arrays from a list of `Batch` samples into
    per-scale batched MLX arrays with a leading sample dimension.
    """
    if len(samples) == 0:
        return Batch(facies=(), wells=(), seismic=())

    # Facies: tuple of per-scale arrays -> produce tuple of stacked arrays
    num_scales = len(samples[0].facies)
    facies_scales: list[mx.array] = []
    for s in range(num_scales):
        items = [sample.facies[s] for sample in samples]
        facies_scales.append(mx.stack(items, axis=0))
    # Wells (may be empty tuple)
    if len(samples[0].wells) == 0:
        wells_scales: tuple[mx.array, ...] = ()
    else:
        num_w_scales = len(samples[0].wells)
        wells_list: list[mx.array] = []
        for s in range(num_w_scales):
            items = [sample.wells[s] for sample in samples]
            wells_list.append(mx.stack(items, axis=0))
        wells_scales: tuple[mx.array, ...] = tuple(wells_list)
    # Seismic (may be empty tuple)
    if len(samples[0].seismic) == 0:
        seismic_scales = ()
    else:
        num_s_scales = len(samples[0].seismic)
        seismic_list: list[mx.array] = []
        for s in range(num_s_scales):
            items = [sample.seismic[s] for sample in samples]
            seismic_list.append(mx.stack(items, axis=0))
        seismic_scales = tuple(seismic_list)

    return Batch(
        facies=tuple(facies_scales), wells=wells_scales, seismic=seismic_scales
    )
