import argparse


class TrainningOptions(argparse.Namespace):
    min_size: int
    max_size: int
    crop_size: int
    stop_scale: int
    batch_size: int
    facie_num_channels: int
    input_path: str
    output_path: str
    noise_amp: float
