from dataclasses import dataclass


@dataclass
class InterpolatorConfig:
    """Configuration for neural interpolator inference.

    This dataclass provides parameters for configuring a NeuralSmoother
    instance during inference (model loading and rendering).

    Attributes
    ----------
    num_classes : int
        Number of output classes for the neural model. Defaults to 4.
        Must match the class count used during training.
    scale : float
        Scale parameter (sigma) for Fourier feature encoding in the model.
        Higher values produce sharper/noisier features; lower values yield
        smoother representations. Defaults to 1.0. Must match the training
        configuration.
    upsample : int
        Upsampling factor applied to input image dimensions during rendering.
        For example, ``upsample=4`` renders at 4× the native resolution.
        Defaults to 4.
    chunk_size : int
        Maximum number of coordinate points to process in a single forward
        pass during inference. Large images or high-resolution outputs are
        evaluated in chunks to limit peak memory usage. Increasing this value
        may improve throughput but will use more memory; decreasing it reduces
        memory at the cost of additional overhead. Defaults to 65536.
    """

    num_classes: int = 4
    scale: float = 1.0
    upsample: int = 4
    chunk_size: int = 65536
    geometry: tuple[int, int] = (150, 120)

