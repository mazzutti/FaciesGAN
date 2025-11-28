import argparse
import math
import os
from collections.abc import Mapping, Sequence
from types import SimpleNamespace
from typing import Any, cast

import torch
import torch.nn as nn
import torch.nn.functional as F

import ops as ops
from config import AMP_FILE, D_FILE, G_FILE, M_FILE, REC_FILE, SHAPE_FILE
from models.discriminator import Discriminator
from models.generator import Generator


class FaciesGAN:
    def __init__(
        self,
        device: torch.device,
        options: argparse.Namespace | SimpleNamespace,
        masked_facies: Sequence[torch.Tensor] = (),
        *args: tuple[Any, ...],
        **kwargs: dict[str, Any],
    ) -> None:
        """
        Initialize the FaciesGAN class.

        Args:
            device (torch.device): The device to run the model on (CPU or GPU).
            options (argparse.Namespace): The options containing hyperparameters and configurations.
            masked_facies (torch.Tensor, optional): The masked facies tensor. Defaults to None.
        """
        super().__init__(*args, **kwargs)
        self.device = device

        # Image parameters
        self.facie_num_channels = options.facie_num_channels + 1
        self.zero_padding = options.num_layer * math.floor(options.kernel_size / 2)

        # Training parameters
        self.discriminator_steps = options.discriminator_steps
        self.generator_steps = options.generator_steps
        self.lambda_grad = options.lambda_grad
        self.alpha = options.alpha

        # Network parameters
        self.num_feature = options.num_feature
        self.min_num_feature = options.min_num_feature
        self.num_layer = options.num_layer
        self.kernel_size = options.kernel_size
        self.padding_size = options.padding_size

        self.shapes: list[tuple[int, ...]] = []
        self.rec_noise: list[torch.Tensor] = []
        self.noise_amp: list[float] = []
        # Current training scale (set by Trainer during training)
        self.cur_scale: int = 0

        # Accept any sequence for `masked_facies` (tuple/list). Default to an
        # empty tuple so the default is immutable and safe as a module-level
        # default. Convert to a per-instance list for internal mutation.
        if masked_facies:
            self.masked_facies: list[torch.Tensor] = list(masked_facies)
        else:
            self.masked_facies = []

        self.generator: Generator = Generator(
            self.num_layer, self.kernel_size, self.padding_size, self.facie_num_channels
        ).to(self.device)
        self.discriminator = None

    def init_scale_generator(self, scale: int) -> None:
        """
        Initialize the generator for a given scale.

        Args:
            scale (int): The current scale index.
        """
        num_feature, min_num_feature = self.get_num_features(scale)

        self.generator.create_scale(num_feature, min_num_feature)
        self.generator.gens[-1] = self.generator.gens[-1].to(self.device)

        # Reinitialize the weights if the Generator features were doubled.
        if scale % 4 == 0:
            self.generator.gens[-1].apply(ops.weights_init)
        else:
            self.generator.gens[-1].load_state_dict(self.generator.gens[-2].state_dict())

        if len(self.generator.gens) > 1:
            self.generator.gens[-2] = ops.reset_grads(self.generator.gens[-2])
            self.generator.gens[-2].eval()

    def init_scale_discriminator(self, scale: int) -> None:
        """
        Initialize the discriminator for a given scale.

        Args:
            scale (int): The current scale index.
        """
        num_feature, min_num_feature = self.get_num_features(scale)

        # If the Discriminator features were doubled, recreate the Discriminator and reinitialize the weights.
        # If not, continue with the current Discriminator
        if scale % 4 == 0:
            self.discriminator = Discriminator(
                num_feature,
                min_num_feature,
                self.num_layer,
                self.kernel_size,
                self.padding_size,
                self.facie_num_channels - 1,
            ).to(self.device)
            self.discriminator.apply(ops.weights_init)

    def get_num_features(self, scale: int) -> tuple[int, int]:
        """
        Calculate the number of features for the generator and discriminator at a given scale.

        Args:
            scale (int): The current scale index.

        Returns:
            tuple[int, int]: A tuple containing the number of features and the minimum number of features.
        """
        num_feature = min(self.num_feature * pow(2, math.floor(scale / 4)), 128)
        min_num_feature = min(self.min_num_feature * pow(2, math.floor(scale / 4)), 128)

        return num_feature, min_num_feature

    def get_noise(
        self, mask_indexes: list[int], rec: bool = False, last: bool = False
    ) -> list[torch.Tensor]:
        """
        Generate noise for the GAN at different scales.

        Args:
            mask_indexes (list[int]): Indexes of the masks.
            rec (bool): If True, return the reconstruction noise.
            last (bool): If True, return only the last scale noise.

        Returns:
            list[torch.Tensor]: Generated noise (one tensor per scale).
        """

        def generate_noise(index: int) -> torch.Tensor:
            shape = self.shapes[index][2:]
            z = ops.generate_noise(
                (self.facie_num_channels - 1, *shape),
                device=self.device,
                num_samp=len(mask_indexes),
            )
            # Use masked facies for this scale only if we have one stored
            # for the current `index` (per-instance list default avoids
            # shared-mutable defaults while allowing empty lists).
            if index < len(self.masked_facies):
                z = torch.cat([z, self.masked_facies[index][mask_indexes].to(self.device)], dim=1)
            elif index == 0:
                z = z.expand(z.shape[0], self.facie_num_channels, *shape)
            else:
                z = ops.generate_noise(
                    (self.facie_num_channels, *shape),
                    device=self.device,
                    num_samp=len(mask_indexes),
                )
            return F.pad(z, [self.zero_padding] * 4, value=0)

        if rec:
            return self.rec_noise.copy()
        if last:
            return [generate_noise(len(self.rec_noise) - 1)]
        return [generate_noise(i) for i in range(len(self.rec_noise))]

    def optimize_discriminator(
        self,
        mask_indexes: list[int],
        real: torch.Tensor,
        discriminator_optimizer: torch.optim.Optimizer,
    ) -> tuple[float, float, float, float]:
        """
        Optimize the discriminator for a given set of real and generated images.

        Args:
            mask_indexes (list[int]): Indexes of the masks.
            real (torch.Tensor): Real images.
            discriminator_optimizer (torch.optim.Optimizer): Optimizer for the discriminator.

        Returns:
            tuple[float, float, float, float]: Total loss, real loss, fake loss, and gradient penalty loss.
        """
        fixed_noise = self.get_noise(mask_indexes, last=True)

        total_loss = 0.0
        # Initialize losses as torch tensors so their types remain consistent
        # (avoids a union of int and Tensor which confuses static analyzers
        # when calling .backward()). Place them on the correct device.
        real_loss = torch.tensor(0.0, device=self.device)
        fake_loss = torch.tensor(0.0, device=self.device)
        gp_loss = torch.tensor(0.0, device=self.device)
        for _ in range(self.discriminator_steps):
            discriminator_optimizer.zero_grad()

            # Calculate loss for real images
            real_output = cast(Discriminator, self.discriminator)(real)
            real_loss = -real_output.mean()

            # Generate fake images
            noises = self.get_noise(mask_indexes)
            noises[-1] = fixed_noise[0]
            with torch.no_grad():
                fake = self.generator(noises, self.noise_amp)

            # Calculate loss for fake images
            fake_output = cast(Discriminator, self.discriminator)(fake.detach())
            fake_loss = fake_output.mean((0, 2, 3))

            # Backpropagate the losses
            real_loss.backward()
            fake_loss.backward()

            # Calculate and backpropagate gradient penalty
            gp_loss = ops.calc_gradient_penalty(
                cast(Discriminator, self.discriminator),
                real,
                fake,
                self.lambda_grad,
                self.device,
            )
            gp_loss.backward()  # type: ignore

            # Update the discriminator
            discriminator_optimizer.step()

            # Accumulate the losses
            total_loss += real_loss.item() + fake_loss.item() + gp_loss.item()

        return total_loss, real_loss.item(), fake_loss.item(), gp_loss.item()

    def optimize_generator(
        self,
        mask_indexes: list[int],
        real: torch.Tensor,
        mask: torch.Tensor,
        rec_in: torch.Tensor,
        generator_optimizer: torch.optim.Optimizer,
    ) -> tuple[float, float, float, torch.Tensor, torch.Tensor | None]:
        """
        Optimize the generator for a given set of real and generated images.

        Args:
            mask_indexes (list[int]): Indexes of the masks.
            real (torch.Tensor): Real images.
            mask (torch.Tensor): Mask tensor.
            rec_in (torch.Tensor): Input tensor for reconstruction.
            generator_optimizer (torch.optim.Optimizer): Optimizer for the generator.

        Returns:
            tuple[float, float, float, torch.Tensor, torch.Tensor]: Total loss, fake loss, reconstruction loss,
            generated fake images, and reconstructed images.
        """
        # Use scalar tensors for losses to keep types consistent (avoid
        # mixing Python floats and torch tensors which confuses static
        # analyzers when calling .item()).
        generator_loss = 0.0
        generator_loss_fake = torch.tensor(0.0, device=self.device)
        generator_loss_rec = torch.tensor(0.0, device=self.device)
        fake, rec = None, None

        for _ in range(self.generator_steps):
            generator_optimizer.zero_grad()

            noises = self.get_noise(mask_indexes)
            fake = self.generator(noises, self.noise_amp)

            generator_loss_fake = -cast(Discriminator, self.discriminator)(fake).mean()
            generator_loss_fake.backward()

            generator_loss_rec = torch.zeros(1, device=self.device)
            rec = None

            if self.alpha != 0:
                rec_noise = self.get_noise(mask_indexes, rec=True)
                rec = self.generator(
                    rec_noise,
                    self.noise_amp,
                    in_facie=rec_in,
                    start_scale=len(self.generator.gens) - 1,
                )
                generator_loss_rec = self.alpha * nn.MSELoss()(rec, real)
                generator_loss_rec.backward()

            generator_masked_loss = (
                100 * self.alpha * nn.MSELoss(reduction="mean")(fake * mask, real * mask)
            )
            generator_loss = (
                float(generator_masked_loss.item())
                + float(generator_loss_fake.item())
                + float(generator_loss_rec.item())
            )

            generator_optimizer.step()

        return (
            generator_loss,
            float(generator_loss_fake.item()),
            float(generator_loss_rec.item()),
            cast(torch.Tensor, fake).detach(),
            rec.detach() if rec is not None else None,
        )

    def save_scale(self, scale: int, path: str) -> None:
        """
        Save the current scale's generator, discriminator, reconstruction noise, shapes, and noise amplitude
        to the specified path.

        Args:
            scale (int): The scale index to save.
            path (str): The directory path where the files will be saved.
        """
        torch.save(self.generator.gens[scale].state_dict(), os.path.join(path, G_FILE))
        torch.save(
            cast(Discriminator, self.discriminator).state_dict(),
            os.path.join(path, D_FILE),
        )
        torch.save(self.rec_noise, os.path.join(path, REC_FILE))
        torch.save(self.shapes[scale], os.path.join(path, SHAPE_FILE))
        torch.save(self.masked_facies[scale], os.path.join(path, M_FILE))

        with open(os.path.join(path, AMP_FILE), "w") as f:
            f.write(str(self.noise_amp[scale]))

    def load(
        self,
        path: str,
        load_discriminator: bool = True,
        load_shapes: bool = True,
        until_scale: int | None = None,
        load_masked_facies: bool = True,
    ) -> int:
        """
        Load the generator, discriminator, reconstruction noise, shapes, and noise amplitude from the specified path.

        Args:
            path (str): The directory path where the files are saved.
            load_discriminator (bool): Whether to load the discriminator. Default is True.
            load_shapes (bool): Whether to load the shapes. Default is True.
            until_scale (int): The scale index up to which to load. Default is None.
            load_masked_facies (bool): Whether to load the masked facies. Default is True.

        Returns:
            int: The number of scales loaded.
        """
        scales_path = sorted(int(scale) for scale in next(os.walk(path))[1])

        if until_scale is not None:
            scales_path = [scale for scale in scales_path if scale <= until_scale]

        if load_masked_facies:
            self.masked_facies = []
        for scale in scales_path:
            try:
                num_feature, min_num_feature = self.get_num_features(scale)

                if load_discriminator:
                    self.discriminator = Discriminator(
                        num_feature,
                        min_num_feature,
                        self.num_layer,
                        self.kernel_size,
                        self.padding_size,
                        self.facie_num_channels,
                    ).to(self.device)
                    self.discriminator.load_state_dict(
                        ops.load(
                            os.path.join(path, str(scale), D_FILE),
                            self.device,
                            as_type=Mapping[str, Any],
                        )
                    )

                if load_shapes:
                    self.shapes.append(
                        ops.load(
                            os.path.join(path, str(scale), SHAPE_FILE),
                            as_type=tuple[int, ...],
                        )
                    )

                self.generator.create_scale(num_feature, min_num_feature)
                self.generator.gens[scale].to(self.device)
                self.generator.gens[scale].load_state_dict(
                    ops.load(
                        os.path.join(path, str(scale), G_FILE),
                        self.device,
                        as_type=Mapping[str, Any],
                    )
                )
                self.generator.gens[scale] = ops.reset_grads(self.generator.gens[scale])
                self.generator.gens[scale].eval()

                self.rec_noise.append(
                    ops.load(
                        os.path.join(path, str(scale), REC_FILE),
                        self.device,
                        as_type=torch.Tensor,
                    )
                )

                with open(os.path.join(path, str(scale), AMP_FILE)) as f:
                    self.noise_amp.append(float(f.readline().strip()))

                if load_masked_facies:
                    self.masked_facies.append(
                        ops.load(
                            os.path.join(path, str(scale), M_FILE),
                            self.device,
                            as_type=torch.Tensor,
                        )
                    )

            except Exception as e:
                load_dir = os.path.join(path, str(scale))
                print("Error loading models from", load_dir, ". There may be files missing.")
                raise e
        return len(scales_path)
