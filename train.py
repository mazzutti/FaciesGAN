from __future__ import annotations

import math
import os
import time
from collections.abc import Mapping
from typing import Any, cast

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tensorboardX import SummaryWriter  # type: ignore
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import (
    D_FILE,
    G_FILE,
    OPT_D_FILE,
    OPT_G_FILE,
    RESULT_FACIES_PATH,
    SCH_D_FILE,
    SCH_G_FILE,
)
from dataset import PyramidsDataset
from log import format_time
from models.discriminator import Discriminator
from models.facies_gan import FaciesGAN
from ops import create_dirs,  generate_noise, interpolate, load
from options import TrainningOptions
from utils import plot_generated_facies


class Trainer:
    """Trainer for multi-scale progressive FaciesGAN training.

    Manages the complete training pipeline including dataset loading, model
    initialization, progressive scale training, and checkpoint saving. Supports
    both fresh training and fine-tuning from checkpoints.

    Parameters
    ----------
    device : torch.device
        Device for training (CPU, CUDA, or MPS).
    options : TrainningOptions
        Training configuration containing hyperparameters and paths.
    fine_tuning : bool, optional
        Whether to load and fine-tune from existing checkpoints. Defaults to False.
    checkpoint_path : str, optional
        Path to load checkpoints from when fine-tuning. Defaults to ".checkpoints/".

    Attributes
    ----------
    device : torch.device
        Training device.
    start_scale : int
        Starting pyramid scale index.
    stop_scale : int
        Final pyramid scale index.
    batch_size : int
        Effective batch size for training.
    model : FaciesGAN
        The FaciesGAN model instance.
    scales_list : tuple[tuple[int, ...], ...]
        Pyramid resolutions for each scale.
    data_loader : DataLoader
        PyTorch DataLoader for training data.
    facies : list[torch.Tensor]
        Current batch of facies data at all scales.
    wells : list[torch.Tensor]
        Current batch of well data at all scales.
    seismic : list[torch.Tensor]
        Current batch of seismic data at all scales.
    stacked_data : list[torch.Tensor]
        Element-wise product of wells and facies for conditioning.
    """
    def __init__(
        self,
        device: torch.device,
        options: TrainningOptions,
        fine_tuning: bool = False,
        checkpoint_path: str = ".checkpoints/",
    ) -> None:
        self.device: torch.device = device

        # Training parameters
        self.start_scale: int = options.start_scale
        self.stop_scale: int = options.stop_scale
        self.output_path: str = options.output_path
        self.num_iter: int = options.num_iter
        self.save_interval: int = options.save_interval
        self.batch_size: int = (
            options.batch_size
            if (options.batch_size < options.num_train_facies)
            else options.num_train_facies
        )
        self.batch_size: int = (
            self.batch_size
            if not (len(options.wells) > 0 
                    and options.batch_size < len(options.wells))
            else len(options.wells)
        )
        self.fine_tuning: bool = fine_tuning
        self.checkpoint_path: str = checkpoint_path

        self.num_real_facies: int = options.num_real_facies
        self.num_generated_per_real: int = options.num_generated_per_real
        self.wells_colums: tuple[int, ...]= options.wells

        # Optimizer configuration
        self.lr_g: float = options.lr_g
        self.lr_d: float = options.lr_d
        self.beta1: float = options.beta1
        self.lr_decay: int = options.lr_decay
        self.gamma: float = options.gamma

        # Model parameters
        self.zero_padding: int = options.num_layer * math.floor(options.kernel_size / 2)
        self.img_num_channel: int = options.facie_num_channels + 1
        self.noise_amp: float = options.noise_amp
        self.facies: list[torch.Tensor] = []
        self.wells: list[torch.Tensor] = []
        self.seismic: list[torch.Tensor] = []
        self.stacked_data: list[torch.Tensor] = []

        dataset: PyramidsDataset = PyramidsDataset(options)
        self.scales_list: tuple[tuple[int, ...], ...] = dataset.scales_list # type: ignore

        if len(options.wells) > 0:
            for i in range(len(self.scales_list)):
                dataset.facies_pyramids[i] = dataset.facies_pyramids[i][options.wells]
                dataset.wells_pyramids[i] = dataset.wells_pyramids[i][options.wells]
                dataset.seismic_pyramids[i] = dataset.seismic_pyramids[i][options.wells]
        elif options.num_train_facies < len(dataset):
            idxs = torch.randperm(len(dataset))[: options.num_train_facies]
            for i in range(len(self.scales_list)):
                dataset.facies_pyramids[i] = dataset.facies_pyramids[i][idxs]
                dataset.wells_pyramids[i] = dataset.wells_pyramids[i][idxs]
                dataset.seismic_pyramids[i] = dataset.seismic_pyramids[i][idxs]

        self.data_loader = DataLoader(
            dataset, batch_size=options.batch_size, shuffle=False)

        self.num_of_batchs: int = len(dataset) // self.batch_size

        self.model: FaciesGAN = FaciesGAN(device, options, self.stacked_data)
        self.model.shapes = list(self.scales_list)

        print("Generated facie shapes:")
        print("╔══════════╦══════════╦══════════╦══════════╗")
        print("║ {:^8} ║ {:^8} ║ {:^8} ║ {:^8} ║".format("Batch", "Channels", "Height", "Width"))
        print("╠══════════╬══════════╬══════════╬══════════╣")
        for shape in self.scales_list:
            print(
                "║ {:^8} ║ {:^8} ║ {:^8} ║ {:^8} ║".format(shape[0], shape[1], shape[2], shape[3])
            )
        print("╚══════════╩══════════╩══════════╩══════════╝")

    def train(self) -> None:
        """Train the FaciesGAN model across all pyramid scales.

        Performs progressive training from coarse to fine scales. For each scale,
        initializes generator and discriminator, optionally loads checkpoint if
        fine-tuning, then trains using the data loader. Logs training time for
        each scale and total training time.
        """
        start_train_time = time.time()

        for scale in range(self.start_scale, self.stop_scale + 1):
            self.model.cur_scale = scale
            scale_start_time = time.time()

            scale_path = os.path.join(self.output_path, str(scale))
            results_path = os.path.join(scale_path, RESULT_FACIES_PATH)

            create_dirs(scale_path)
            create_dirs(results_path)

            self.model.init_scale_generator(scale)
            self.model.init_scale_discriminator(scale)

            if self.fine_tuning:
                self.__load_model(scale)

            writer = SummaryWriter(log_dir=scale_path)
            data_iterator = iter(self.data_loader)
            for batch_id, (
                self.facies, 
                self.wells, 
                self.seismic
            ) in enumerate(data_iterator):
                self.stacked_data = [mask * facie for mask, facie in zip(self.wells, self.facies)]
                self.model.masked_facies = self.stacked_data
                self.train_scale(scale, writer, scale_path, results_path, batch_id)
                self.model.save_scale(scale, scale_path)

            scale_end_time = time.time()
            elapsed = format_time(int(scale_end_time - scale_start_time))
            print(f"Scale {scale + 1} training time: {elapsed}")

        end_train_time = time.time()
        print(
            "\nTotal training time:",
            format_time(int(end_train_time - start_train_time)),
        )

    def train_scale(
        self,
        scale: int,
        writer: SummaryWriter,
        scale_path: str,
        results_path: str,
        batch_id: int,
    ) -> None:
        """Train the model at a specific pyramid scale.

        Initializes optimizers and schedulers, then iterates through training
        epochs alternating between discriminator and generator updates. Saves
        generated facies visualizations at regular intervals.

        Parameters
        ----------
        scale : int
            Current pyramid scale index.
        writer : SummaryWriter
            TensorBoard writer for logging metrics.
        scale_path : str
            Directory to save scale-specific checkpoints and optimizers.
        results_path : str
            Directory to save generated facies visualizations.
        batch_id : int
            Current batch index within the epoch.
        """
        mask_indexes = list(range(self.batch_size))

        real = self.facies[scale][mask_indexes].to(self.device)
        mask = self.wells[scale][mask_indexes].to(self.device)
        masked_facie = self.stacked_data[scale][mask_indexes].to(self.device)

        generator_optimizer = optim.Adam(
            self.model.generator.gens[-1].parameters(),
            lr=self.lr_g,
            betas=(self.beta1, 0.999),
        )
        discriminator_optimizer = optim.Adam(
            cast(Discriminator, self.model.discriminator).parameters(),
            lr=self.lr_d,
            betas=(self.beta1, 0.999),
        )
        generator_scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer=generator_optimizer, milestones=[self.lr_decay], gamma=self.gamma
        )
        discriminator_scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer=discriminator_optimizer,
            milestones=[self.lr_decay],
            gamma=self.gamma,
        )

        if self.fine_tuning:
            self.__load_optimizers(
                scale,
                scale_path,
                generator_optimizer,
                discriminator_optimizer,
                generator_scheduler,
                discriminator_scheduler,
            )

        _, prev_rec = self.__initialize_noise(scale, masked_facie, real, mask_indexes)

        epochs: 'tqdm[int]' = tqdm(range(1, self.num_iter + 1))

        for epoch in epochs:

            (
                discriminator_loss,
                discriminator_loss_real,
                discriminator_loss_fake,
                discriminator_loss_gp,
            ) = self.model.optimize_discriminator(mask_indexes, real, discriminator_optimizer)

            generator_loss, generator_loss_fake, generator_loss_rec, _, _ = (
                self.model.optimize_generator(
                    mask_indexes,
                    real,
                    mask,
                    cast(torch.Tensor, prev_rec),
                    generator_optimizer,
                )
            )

            self.__log_epoch(
                epochs,
                writer,
                epoch,
                scale,
                batch_id,
                generator_loss,
                discriminator_loss,
                discriminator_loss_real,
                discriminator_loss_fake,
                discriminator_loss_gp,
                generator_loss_fake,
                generator_loss_rec,
            )

            if epoch % self.save_interval == 0 or epoch == self.num_iter:
                self.__save_generated_facies(scale, epoch, results_path, mask)

            generator_scheduler.step()
            discriminator_scheduler.step()

        self.__save_optimizers(
            scale_path,
            generator_optimizer,
            discriminator_optimizer,
            generator_scheduler,
            discriminator_scheduler,
        )

    def load(self, path: str, until_scale: int | None = None) -> None:
        """Load saved models and set the starting scale for training.

        Parameters
        ----------
        path : str
            Path to the directory containing model checkpoint files.
        until_scale : int | None, optional
            Load models up to and including this scale. If None, loads all
            available scales. Defaults to None.
        """
        self.start_scale = self.model.load(path, load_shapes=False, until_scale=until_scale)

    def __load_model(self, scale: int) -> None:
        """Load generator and discriminator state dicts for a specific scale.

        Parameters
        ----------
        scale : int
            Scale index to load models for.

        Raises
        ------
        Exception
            If model files cannot be loaded from checkpoint path.
        """
        try:
            generator_path = os.path.join(str(self.checkpoint_path), str(scale), G_FILE)
            discriminator_path = os.path.join(str(self.checkpoint_path), str(scale), D_FILE)

            self.model.generator.gens[-1].load_state_dict(
                load(generator_path, self.device, as_type=Mapping[str, Any])
            )
            cast(Discriminator, self.model.discriminator).load_state_dict(
                load(discriminator_path, self.device, as_type=Mapping[str, Any])
            )
        except Exception as e:
            cp = self.checkpoint_path
            print("Error loading models from", cp, "/", scale, ":", e)
            raise

    def __load_optimizers(
        self,
        scale: int,
        scale_path: str,
        generator_optimizer: optim.Optimizer,
        discriminator_optimizer: optim.Optimizer,
        generator_scheduler: optim.lr_scheduler.LRScheduler,
        discriminator_scheduler: optim.lr_scheduler.LRScheduler,
    ) -> None:
        """Load optimizer and scheduler state dictionaries from checkpoint.

        Parameters
        ----------
        scale : int
            Current scale index (used for error reporting).
        scale_path : str
            Directory containing optimizer and scheduler checkpoints.
        generator_optimizer : optim.Optimizer
            Generator optimizer to load state into.
        discriminator_optimizer : optim.Optimizer
            Discriminator optimizer to load state into.
        generator_scheduler : optim.lr_scheduler.LRScheduler
            Generator learning rate scheduler to load state into.
        discriminator_scheduler : optim.lr_scheduler.LRScheduler
            Discriminator learning rate scheduler to load state into.

        Raises
        ------
        Exception
            If optimizer/scheduler files cannot be loaded.
        """
        try:
            generator_optimizer.load_state_dict(
                load(os.path.join(scale_path, OPT_G_FILE), self.device)
            )
            discriminator_optimizer.load_state_dict(
                load(os.path.join(scale_path, OPT_D_FILE), self.device)
            )
            generator_scheduler.load_state_dict(
                load(os.path.join(scale_path, SCH_G_FILE), self.device)
            )
            discriminator_scheduler.load_state_dict(
                load(os.path.join(scale_path, SCH_D_FILE), self.device)
            )
        except Exception as e:
            print("Error loading optimizers from", scale_path, "/", scale, ":", e)
            raise e

    def __initialize_noise(
        self,
        scale: int,
        masked_facie: torch.Tensor,
        real: torch.Tensor,
        mask_indexes: list[int],
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Initialize noise tensors and compute noise amplitude for the current scale.

        For scale 0, initializes reconstruction noise with amplitude 1.0.
        For subsequent scales, computes reconstruction from previous scales and
        calculates amplitude based on RMSE between real and reconstructed images.

        Parameters
        ----------
        scale : int
            Current pyramid scale index.
        masked_facie : torch.Tensor
            Well-conditioned facies data for this scale.
        real : torch.Tensor
            Real facies images at this scale.
        mask_indexes : list[int]
            Indices of masked samples in the batch.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor | None]
            Tuple of (z_rec, prev_rec) where z_rec is the initialized noise tensor
            and prev_rec is the reconstruction from previous scales (None for scale 0).
        """

        def update_rec_noise(rec: torch.Tensor) -> None:
            if len(self.model.rec_noise) <= scale:
                self.model.rec_noise.append(rec)
            else:
                self.model.rec_noise[scale] = (rec + self.model.rec_noise[scale]) / 2

        if scale == 0:
            z_rec = generate_noise(
                (1, *self.scales_list[scale][2:]),
                device=self.device,
                num_samp=self.batch_size,
            )
            z_rec = torch.cat([z_rec, masked_facie], dim=1)
            z_rec = F.pad(z_rec, [self.zero_padding] * 4, value=0)
            self.model.noise_amp.append(1.0) if len(self.model.rec_noise) == 0 else None
            update_rec_noise(z_rec)
            prev_rec = None
        else:
            z_rec = torch.zeros(
                (self.batch_size, 1, *self.scales_list[scale][2:]), device=self.device
            )
            z_rec = torch.cat([z_rec, masked_facie], dim=1)
            z_rec = F.pad(z_rec, [self.zero_padding] * 4, value=0)
            update_rec_noise(z_rec)

            z_in = self.model.get_noise(mask_indexes, rec=True)
            with torch.no_grad():
                prev_rec = self.model.generator(
                    z_in,
                    self.model.noise_amp,
                    stop_scale=len(self.model.generator.gens) - 2,
                )
                prev_rec = interpolate(prev_rec, (real.shape[-2], real.shape[-1]))

            rec_loss = nn.MSELoss()(real, prev_rec)
            RMSE = torch.sqrt(rec_loss).detach().item()
            amp = self.noise_amp * RMSE
            if len(self.model.noise_amp) <= scale:
                self.model.noise_amp.append(amp)
            else:
                self.model.noise_amp[scale] = (amp + self.model.noise_amp[scale]) / 2

        return z_rec, prev_rec

    def __log_epoch(
        self,
        epochs: 'tqdm[int]',
        writer: SummaryWriter,
        epoch: int,
        scale: int,
        batch_id: int,
        generator_loss: float,
        discriminator_loss: float,
        discriminator_loss_real: float,
        discriminator_loss_fake: float,
        discriminator_loss_gp: float,
        generator_loss_fake: float,
        generator_loss_rec: float,
    ) -> None:
        """Log training metrics for the current epoch to TensorBoard and console.

        Updates the progress bar description and writes all loss components
        to TensorBoard for visualization.

        Parameters
        ----------
        epochs : tqdm[int]
            Progress bar for epoch iteration.
        writer : SummaryWriter
            TensorBoard writer for logging metrics.
        epoch : int
            Current epoch number.
        scale : int
            Current pyramid scale index.
        batch_id : int
            Current batch index.
        generator_loss : float
            Total generator loss.
        discriminator_loss : float
            Total discriminator loss.
        discriminator_loss_real : float
            Discriminator loss on real samples.
        discriminator_loss_fake : float
            Discriminator loss on generated samples.
        discriminator_loss_gp : float
            Gradient penalty loss component.
        generator_loss_fake : float
            Generator adversarial loss component.
        generator_loss_rec : float
            Generator reconstruction loss component.
        """
        epochs.set_description(
            "Stage [{}/{}] | Batch [{}/{}] | Loss [G: {:2.3f}| D: {:2.3f}] Epoch".format(
                scale + 1,
                self.stop_scale + 1,
                batch_id + 1,
                self.num_of_batchs,
                generator_loss,
                discriminator_loss,
            )
        )

        # Ensure scalars are plain Python floats for tensorboardX typing
        writer.add_scalar(  # type: ignore
            "Loss/train/discriminator/real", -float(discriminator_loss_real), epoch
        )
        writer.add_scalar(  # type: ignore
            "Loss/train/discriminator/fake", float(discriminator_loss_fake), epoch
        )
        writer.add_scalar(  # type: ignore
            "Loss/train/discriminator/gradient_penalty",
            float(discriminator_loss_gp),
            epoch,
        )
        writer.add_scalar(  # type: ignore
            "Loss/train/discriminator", float(discriminator_loss), epoch
        )
        writer.add_scalar(  # type: ignore
            "Loss/train/generator/fake", float(generator_loss_fake), epoch
        )
        writer.add_scalar(  # type: ignore
            "Loss/train/generator/reconstruction", float(generator_loss_rec), epoch
        )
        writer.add_scalar("Loss/train/generator", float(generator_loss), epoch)  # type: ignore

    def __save_generated_facies(
        self, scale: int, epoch: int, results_path: str, mask: torch.Tensor
    ) -> None:
        indexes = torch.randint(self.batch_size, (self.num_real_facies,))
        real_facies = self.facies[scale][indexes]
        # get_noise expects a list[int] of indexes; convert tensor indices to python lists
        noises = [
            self.model.get_noise([int(index.item())] * self.num_generated_per_real)
            for index in indexes
        ]
        with torch.no_grad():
            generated_facies = [
                self.model.generator(noise, self.model.noise_amp) for noise in noises
            ]
            generated_facies = [generated_facie.clip(-1, 1) for generated_facie in generated_facies]
        plot_generated_facies(
            generated_facies,
            real_facies,
            mask[indexes],
            scale,
            epoch,
            out_dir=results_path,
            save=True,
        )

    @staticmethod
    def __save_optimizers(
        scale_path: str,
        generator_optimizer: optim.Optimizer,
        discriminator_optimizer: optim.Optimizer,
        generator_scheduler: optim.lr_scheduler.LRScheduler,
        discriminator_scheduler: optim.lr_scheduler.LRScheduler,
    ) -> None:
        """Save optimizer and scheduler state dictionaries to disk.

        Parameters
        ----------
        scale_path : str
            Directory to save checkpoint files.
        generator_optimizer : optim.Optimizer
            Generator optimizer to save.
        discriminator_optimizer : optim.Optimizer
            Discriminator optimizer to save.
        generator_scheduler : optim.lr_scheduler.LRScheduler
            Generator learning rate scheduler to save.
        discriminator_scheduler : optim.lr_scheduler.LRScheduler
            Discriminator learning rate scheduler to save.
        """
        torch.save(generator_optimizer.state_dict(), str(os.path.join(scale_path, OPT_G_FILE)))
        torch.save(
            discriminator_optimizer.state_dict(),
            str(os.path.join(scale_path, OPT_D_FILE)),
        )
        torch.save(generator_scheduler.state_dict(), str(os.path.join(scale_path, SCH_G_FILE)))
        torch.save(
            discriminator_scheduler.state_dict(),
            str(os.path.join(scale_path, SCH_D_FILE)),
        )
