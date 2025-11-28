import argparse
from dataclasses import dataclass


@dataclass
class TrainningOptions(argparse.Namespace):
    """Namespace-like object holding training options with explicit defaults.

    This class mirrors the command-line arguments and provides defaults when
    instantiated without parameters. It is safe to pass an instance to
    `argparse.ArgumentParser.parse_args(namespace=...)`.
    """

    def __init__(
        self,
        alpha: int = 10,
        batch_size: int = 1,
        beta1: float = 0.5,
        crop_size: int = 256,
        discriminator_steps: int = 3,
        facie_num_channels: int = 3,
        gamma: float = 0.9,
        generator_steps: int = 3,
        gpu_device: int = 0,
        img_color_range: tuple[int, int] = (0, 255),
        input_path: str = "data/",
        kernel_size: int = 3,
        lambda_grad: float = 0.1,
        lr_d: float = 5e-05,
        lr_decay: int = 1000,
        lr_g: float = 5e-05,
        manual_seed: int | None = None,
        max_size: int = 1024,
        min_num_feature: int = 32,
        min_size: int = 8,
        noise_amp: float = 0.1,
        num_feature: int = 32,
        num_generated_per_real: int = 5,
        num_iter: int = 2000,
        num_layer: int = 5,
        num_real_facies: int = 5,
        num_train_facies: int = 200,
        output_path: str = "results/",
        padding_size: int = 0,
        regen_npy_gz: bool = False,
        save_interval: int = 100,
        start_scale: int = 0,
        stride: int = 1,
        stop_scale: int = 6,
        use_cpu: bool = False,
        wells: tuple[int, ...] = (),
    ) -> None:
        """Create a TrainningOptions namespace with defaults for training.

        Parameters
        ----------
        alpha : int, optional
            Weight for reconstruction loss (L1/L2) used by the model. Default
            is 10.
        batch_size : int, optional
            Number of samples per batch. Default is 1.
        beta1 : float, optional
            Beta1 parameter for the Adam optimizer. Default is 0.5.
        crop_size : int, optional
            Size to crop input facies for training. Default is 256.
        discriminator_steps : int, optional
            Number of discriminator steps per training iteration. Default is 3.
        facie_num_channels : int, optional
            Number of channels in the input facie. Default is 3.
        gamma : float, optional
            Learning-rate scheduler multiplicative factor. Default is 0.9.
        generator_steps : int, optional
            Number of generator steps per training iteration. Default is 3.
        gpu_device : int, optional
            GPU device id to use when CUDA is available. Default is 0.
        img_color_range : tuple[int, int], optional
            Range of input values for facies (min, max). Default is (0, 255).
        input_path : str, optional
            Path to the dataset root directory. Default is "data/."
        kernel_size : int, optional
            Convolution kernel size used across the networks. Default is 3.
        lambda_grad : float, optional
            Gradient penalty weight for discriminator regularization. Default
            is 0.1.
        lr_d : float, optional
            Learning rate for the discriminator optimizer. Default is 5e-05.
        lr_decay : int, optional
            Number of epochs before the learning rate scheduler decays. Default
            is 1000.
        lr_g : float, optional
            Learning rate for the generator optimizer. Default is 5e-05.
        manual_seed : int or None, optional
            Optional random seed for reproducibility. Default is None.
        max_size : int, optional
            Maximum image size used in scale generation. Default is 1024.
        min_num_feature : int, optional
            Minimum number of features in network layers. Default is 32.
        min_size : int, optional
            Minimum size at the coarsest pyramid scale. Default is 8.
        noise_amp : float, optional
            Base amplitude used to scale adaptive noise. Default is 0.1.
        num_feature : int, optional
            Base number of features in the first network layer. Default is 32.
        num_generated_per_real : int, optional
            How many generated facies to produce per real example. Default is 5.
        num_iter : int, optional
            Number of training iterations (epochs) per scale. Default is 2000.
        num_layer : int, optional
            Number of layers per block/scale. Default is 5.
        num_real_facies : int, optional
            Number of real facies used when composing result grids. Default is 5.
        num_train_facies : int, optional
            Limit on how many training facies to use from the dataset. Default is
            200.
        output_path : str, optional
            Output directory for checkpoints and results. Default is "results/."
        padding_size : int, optional
            Padding size applied in network layers. Default is 0.
        regen_npy_gz : bool, optional
            If True, regenerate the npy.gz files from the input data. Default
            is False.
        save_interval : int, optional
            Interval (in epochs) between saving generated outputs. Default is
            100.
        start_scale : int, optional
            Starting scale index for training. Default is 0.
        stride : int, optional
            Convolution stride used across the networks. Default is 1.
        stop_scale : int, optional
            Final scale index (number of pyramid levels - 1). Default is 6.
        use_cpu : bool, optional
            Force CPU even if CUDA/MPS is available. Default is False.
        wells : tuple of int, optional
            Optional list/tuple of well indices to filter dataset. Default is
            an empty tuple.

        Notes
        -----
        All parameters set here are attached as attributes on the resulting
        `TrainningOptions` instance so `argparse` can populate them when used
        as the `namespace=` for `ArgumentParser.parse_args`.
        """
        # Assign attributes (alphabetical by attribute name)
        self.alpha = alpha
        self.batch_size = batch_size
        self.beta1 = beta1
        self.crop_size = crop_size
        self.discriminator_steps = discriminator_steps
        self.facie_num_channels = facie_num_channels
        self.gamma = gamma
        self.generator_steps = generator_steps
        self.gpu_device = gpu_device
        self.img_color_range = img_color_range
        self.input_path = input_path
        self.kernel_size = kernel_size
        self.lambda_grad = lambda_grad
        self.lr_d = lr_d
        self.lr_decay = lr_decay
        self.lr_g = lr_g
        self.manual_seed = manual_seed
        self.max_size = max_size
        self.min_num_feature = min_num_feature
        self.min_size = min_size
        self.noise_amp = noise_amp
        self.num_feature = num_feature
        self.num_generated_per_real = num_generated_per_real
        self.num_iter = num_iter
        self.num_layer = num_layer
        self.num_real_facies = num_real_facies
        self.num_train_facies = num_train_facies
        self.output_path = output_path
        self.padding_size = padding_size
        self.regen_npy_gz = regen_npy_gz
        self.save_interval = save_interval
        self.start_scale = start_scale
        self.stride = stride
        self.stop_scale = stop_scale
        self.use_cpu = use_cpu
        self.wells = wells


class ResumeOptions(argparse.Namespace):
    """Namespace-like object holding resume script options.

    This mirrors the command-line arguments used by `resume.py` and provides
    explicit defaults. An instance is safe to pass to
    `argparse.ArgumentParser.parse_args(namespace=...)`.
    """

    def __init__(
        self,
        fine_tuning: bool = False,
        checkpoint_path: str = "",
        num_iter: int | None = None,
        start_scale: int = 0,
    ) -> None:
        """Initialize ResumeOptions.

        Parameters
        ----------
        fine_tuning : bool
            If True, resume script will perform fine-tuning of the models.
        checkpoint_path : str
            Path to the checkpoint directory to resume training from.
            This is typically required when resuming.
        num_iter : int or None
            Number of iterations (epochs) to run when fine-tuning. If
            `fine_tuning` is True, this should be provided by the caller.
        start_scale : int
            Starting scale index for resuming/fine-tuning (default 0).
        """

        self.fine_tuning = fine_tuning
        self.checkpoint_path = checkpoint_path
        self.num_iter = num_iter
        self.start_scale = start_scale