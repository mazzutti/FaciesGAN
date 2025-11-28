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
from facies_dataset import FaciesDataset
from log import format_time
from models.discriminator import Discriminator
from models.facies_gan import FaciesGAN
from ops import create_dirs, facie_resize, generate_noise, load
from protocols import TrainningOptions
from utils import plot_generated_facies


class Trainer:
    def __init__(
        self,
        device: torch.device,
        options: TrainningOptions,
        fine_tuning: bool = False,
        checkpoint_path: str | None = None,
    ):
        self.device: torch.device = device

        # Training parameters
        self.start_scale: int = options.start_scale
        self.stop_scale: int = options.stop_scale
        self.out_path: str = options.out_path
        self.num_iter: int = options.num_iter
        self.save_interval: int = options.save_interval
        self.batch_size: int = (
            options.batch_size
            if (options.batch_size < options.num_train_facies)
            else options.num_train_facies
        )
        self.batch_size: int = (
            self.batch_size
            if not (len(options.wells) > 0 and options.batch_size < len(options.wells))
            else len(options.wells)
        )
        self.fine_tuning: bool = fine_tuning
        self.checkpoint_path: str | None = checkpoint_path

        self.num_real_facies: int = options.num_real_facies
        self.num_generated_per_real: int = options.num_generated_per_real
        self.wells = options.wells

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
        self.masks: list[torch.Tensor] = []
        self.masked_facies: list[torch.Tensor] = []

        dataset: FaciesDataset = FaciesDataset(options)
        self.scales_list: list[tuple[int, ...]] = dataset.scales_list

        if len(options.wells) > 0:
            for i in range(len(self.scales_list)):
                dataset.facies_pyramid[i] = dataset.facies_pyramid[i][list(options.wells)]
                dataset.masks_pyramid[i] = dataset.masks_pyramid[i][list(options.wells)]
        elif options.num_train_facies < len(dataset):
            idxs = torch.randperm(len(dataset))[: options.num_train_facies]
            for i in range(len(self.scales_list)):
                dataset.facies_pyramid[i] = dataset.facies_pyramid[i][idxs]
                dataset.masks_pyramid[i] = dataset.masks_pyramid[i][idxs]

        self.data_loader = DataLoader(dataset, batch_size=options.batch_size, shuffle=False)

        self.num_of_batchs: int = len(dataset) // self.batch_size

        self.model: FaciesGAN = FaciesGAN(device, options, self.masked_facies)
        self.model.shapes = self.scales_list

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
        """
        Train the model across multiple scales.

        This method initializes the generator and discriminator for each scale,
        loads the model if fine-tuning, and iterates over the data loader to train
        the model at each scale. It also logs the training time for each scale and
        the total training time.

        Returns:
            None
        """
        start_train_time = time.time()

        for scale in range(self.start_scale, self.stop_scale + 1):
            self.model.cur_scale = scale
            scale_start_time = time.time()

            scale_path = os.path.join(self.out_path, str(scale))
            results_path = os.path.join(scale_path, RESULT_FACIES_PATH)

            create_dirs(scale_path)
            create_dirs(results_path)

            self.model.init_scale_generator(scale)
            self.model.init_scale_discriminator(scale)

            if self.fine_tuning:
                self.__load_model(scale)

            writer = SummaryWriter(log_dir=scale_path)
            data_iterator = iter(self.data_loader)
            for batch_id, (self.facies, self.masks) in enumerate(data_iterator):
                self.masked_facies = [mask * facie for mask, facie in zip(self.masks, self.facies)]
                self.model.masked_facies = self.masked_facies
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
        """
        Train the model at a specific scale.

        Args:
            scale (int): The current scale index.
            writer (SummaryWriter): TensorBoard writer for logging.
            scale_path (str): Path to save the scale-specific models and optimizers.
            results_path (str): Path to save the generated facies.
            batch_id (int): The current batch index.
        """
        mask_indexes = list(range(self.batch_size))

        real = self.facies[scale][mask_indexes].to(self.device)
        mask = self.masks[scale][mask_indexes].to(self.device)
        masked_facie = self.masked_facies[scale][mask_indexes].to(self.device)

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

        epochs = tqdm(range(1, self.num_iter + 1))

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
        """
        Load the models and update the start scale for training.

        Args:
            path (str): The path to the model files.
            until_scale (int, optional): The scale until which to load the models. Defaults to None.
        """
        self.start_scale = self.model.load(path, load_shapes=False, until_scale=until_scale)

    def __load_model(self, scale: int) -> None:
        """
        Load the generator and discriminator models for a given scale.

        Args:
            scale (int): The scale index to load the models for.
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
        """
        Load the state dictionaries for the optimizers and schedulers.

        Args:
            scale (int): The current scale index.
            scale_path (str): Path to the scale-specific models and optimizers.
            generator_optimizer (optim.Optimizer): Optimizer for the generator.
            discriminator_optimizer (optim.Optimizer): Optimizer for the discriminator.
            generator_scheduler (optim.lr_scheduler.LRScheduler): Scheduler for the generator optimizer.
            discriminator_scheduler (optim.lr_scheduler.LRScheduler): Scheduler for the discriminator optimizer.
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
        """
        Initialize the noise for the given scale.

        Args:
            scale (int): The current scale index.
            masked_facie (torch.Tensor): The masked facie tensor.
            real (torch.Tensor): The real facie tensor.
            mask_indexes (List[int]): The list of masked_facie indexes.

        Returns:
            Tuple[torch.Tensor, Optional[torch.Tensor]]: The initialized noise tensor and the previous
            reconstruction tensor.
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
                prev_rec = facie_resize(prev_rec, (real.shape[-2], real.shape[-1]))

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
        epochs: tqdm[int],
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
        """
        Log the losses for the current epoch.

        Args:
            epochs (tqdm): The tqdm progress bar for epochs.
            writer (SummaryWriter): TensorBoard writer for logging.
            epoch (int): The current epoch number.
            scale (int): The current scale index.
            batch_id (int): The current batch index.
            generator_loss (float): The total generator loss.
            discriminator_loss (float): The total discriminator loss.
            discriminator_loss_real (float): The discriminator loss for real samples.
            discriminator_loss_fake (float): The discriminator loss for fake samples.
            discriminator_loss_gp (float): The gradient penalty loss for the discriminator.
            generator_loss_fake (float): The generator loss for fake samples.
            generator_loss_rec (float): The reconstruction loss for the generator.
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
        """
        Save the state dictionaries for the optimizers and schedulers.

        Args:
            scale_path (str): Path to save the scale-specific models and optimizers.
            generator_optimizer (optim.Optimizer): Optimizer for the generator.
            discriminator_optimizer (optim.Optimizer): Optimizer for the discriminator.
            generator_scheduler (optim.lr_scheduler._LRScheduler): Scheduler for the generator optimizer.
            discriminator_scheduler (optim.lr_scheduler._LRScheduler): Scheduler for the discriminator optimizer.
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
