# FaciesGAN: Conditional SinGAN for Transversal Geological Facies Prediction

## 📌 Overview
This repository implements FaciesGAN to generate transversal geological facies realizations, conditioned on well log data. The model learns geological patterns from an input training set and generates high-resolution facies images that honor well constraints using a multi-scale progressive GAN architecture.

## ✨ Key Features
* **Multi-scale Progressive Training**: FaciesGAN architecture with pyramid-based learning from coarse to fine scales
* **Parallel Multi-Scale Training**: Train multiple pyramid scales simultaneously to reduce wall-clock time
* **Well Conditioning**: Incorporates well log data as constraints to ensure geological realism
* **Neural Interpolation**: Advanced interpolators (nearest, neural, well-based) for smooth multi-scale representations
* **Color Palette Encoding**: Efficient RGB-to-label conversion for categorical facies data
* **Cached Pyramid Generation**: Performance-optimized with joblib caching for faster training
* **Type-Safe Codebase**: Full type hints with Pylance strict mode for enhanced code quality
* **Flexible Data Management**: Centralized file handling through DataFiles enum

## 🏗️ Architecture Components

### Core Modules
* **`models/`** - Model code. The project uses a framework-agnostic base plus framework-specific
    implementations under `models/torch/` (PyTorch).
    - `models/base.py` - Framework-agnostic `FaciesGAN` base class and abstract hooks
    - `models/torch/` - PyTorch adapter implementations: `generator.py`, `discriminator.py`,
        `facies_gan.py`, and helper `types.py`
    - Custom layers and utilities are colocated with the framework adapter to keep
        framework-specific logic separated from the core orchestration.
* **`interpolators/`** - Multi-scale interpolation strategies:
  - `BaseInterpolator`: Common functionality for all interpolators
  - `NearestInterpolator`: Fast nearest-neighbor interpolation
  - `NeuralSmoother`: Neural network-based smooth interpolation
  - `WellInterpolator`: Well-conditioned interpolation
* **`color_encoder.py`** - Palette-based RGB ↔ label conversion
* **`gen_pyramids.py`** - Cached pyramid generation utilities
* **`ops.py`** - Core operations: image loading, noise generation, device management
* **`data_files.py`** - Centralized data path management via DataFiles enum

### Data Structure
```
FaciesGAN/
├── data/
│   ├── facies/              # Facies images (.png) and tensors (.pt)
│   ├── wells/               # Well log data and mapping (.npz)
│   └── seismic/             # Seismic images (.png)
├── interpolators/
│   ├── base.py              # Base interpolator class
│   ├── config.py            # Interpolator configuration
│   ├── nearest.py           # Nearest-neighbor interpolator
│   ├── neural.py            # Neural network interpolator
│   └── well.py              # Well-conditioned interpolator
├── models/
│   ├── base.py              # Framework-agnostic FaciesGAN base class
│   ├── torch/               # PyTorch framework-specific implementations
│   │   ├── generator.py     # Multi-scale generator (PyTorch)
│   │   ├── discriminator.py # Multi-scale discriminator (PyTorch)
│   │   └── facies_gan.py    # PyTorch FaciesGAN adapter
├── results/                 # Training outputs and generated facies
├── train.py                 # Training script
├── gen_facies.py            # Facies generation script
├── main.py                  # Main entry point
├── dataset.py               # Dataset loading and preprocessing
├── color_encoder.py         # Color palette encoder
├── gen_pyramids.py          # Pyramid generation with caching
├── ops.py                   # Utility operations
├── data_files.py            # File path management
├── options.py               # Training and generation options
└── requirements.txt         # Python dependencies
````

## 🚀 Installation

### Prerequisites
* Python 3.10+
* PyTorch with CUDA/MPS support (for GPU acceleration)
* 8GB+ GPU memory recommended for training

### Setup
```sh
git clone https://github.com/mazzutti/FaciesGAN.git
cd FaciesGAN
pip install -r requirements.txt
```

### Development Setup
For contributing and development:
```sh
pip install -r requirements-dev.txt  # Includes black, flake8, mypy, etc.
```

## 🎯 Quick Start

### Training FaciesGAN
Train the model on your facies dataset with multi-scale progressive learning:
```sh
python main.py --input_path data --num_iter 40 --batch_size 200 \
    --save_interval 10 --num_train_facies 200 --gpu_device 0 \
    --min_size 16 --max_size 128 --stop_scale 8 --num_parallel_scales 2
```

### Smoke test (one-iteration run)
Run a short smoke test to validate the environment and the recent refactors:
```sh
./.venv/bin/python main.py --input-path data --output-path results/ \\
    --num-train-pyramids 1 --num-iter 1 --num-parallel-scales 7 \\
    --no-tensorboard --use-wells --use-seismic
```
This is the exact command used during development for a quick end-to-end sanity check.

### Performance Profiling

Enable PyTorch profiler with the `--use-profiler` flag to analyze performance bottlenecks.
The profiling behavior depends on your hardware backend:

#### CUDA/CPU Profiling
Exports a Chrome trace to `<output_path>/profiler_trace.json` that can be viewed at `chrome://tracing`:

```sh
# Profile CUDA training with parallel scales
python main.py --input_path data --num_iter 100 --batch_size 200 \
    --num_train_pyramids 50 --gpu_device 0 --stop_scale 6 \
    --num_parallel_scales 2 --use-profiler

# After training completes, open chrome://tracing and load profiler_trace.json
```

#### MPS (Apple Silicon) Profiling
Uses `torch.mps.profiler` to generate OS Signpost traces for Xcode Instruments:

```sh
# Profile MPS training (automatically opens Instruments after completion)
python main.py --input_path data --num_iter 100 --batch_size 200 \
    --num_train_pyramids 50 --stop_scale 6 \
    --num_parallel_scales 2 --use-profiler
```

**Workflow for MPS profiling:**
1. Open Xcode Instruments before training
2. Select the "Logging" instrument
3. Click the record button
4. Run your training command with `--use-profiler`
5. View detailed signpost intervals showing MPS kernel execution in the Instruments timeline

**Note**: The script will automatically attempt to open Instruments after profiling, but traces
must be captured during execution. Launch Instruments and start recording before running training.

**Key Training Parameters:**
**Key Training Parameters:**
- `--min_size`: Starting resolution for coarse scale (int, default: 16)
- `--max_size`: Final high-resolution output (int, default: 128)
- `--stop_scale`: Number of scales in the pyramid (int, default: 8)
- `--num_train_pyramids`: Number of pyramid groups sampled per training step (int, default: 1)
- `--num_train_facies`: Number of facies images to use for training (int, default: 200)
- `--batch_size`: Batch size for training (int, default: 200)
- `--num_iter`: Training iterations per scale (int, default: 40)
- `--save_interval`: Save checkpoint every N iterations (int, default: 10)
- `--num_parallel_scales`: Number of scales trained in parallel per group (int, default: 2)
- `--well-loss-penalty`: Multiplier applied to well-conditioning loss (float, default: 10.0)
- `--lr_g`, `--lr_d`: Learning rates for generator and discriminator (float, default: 0.0005)
- `--alpha`: Reconstruction loss weight (float, default: 100)
- `--beta`: Well conditioning weight (float, default: 0.1)
- `--gpu_device`: GPU device index or identifier (int, default: 0). If not available, CPU or MPS backends are used when supported.
- `--use-profiler`: Enable PyTorch profiler (flag). Produces Chrome traces on CUDA/CPU and OS Signpost traces for MPS.
- `--no-tensorboard`: Disable TensorBoard logging (flag)
- `--use-wells`, `--use-seismic`: Enable conditioning on well and seismic data respectively (flags)

## Exact defaults

The authoritative defaults live in [options.py](options.py). Below are the most commonly-used defaults copied from that file for quick reference; use the linked file for the full list and authoritative source of truth.

- `alpha`: 10
- `batch_size`: 1
- `lr_g`, `lr_d`: 5e-05
- `min_size`: 12
- `max_size`: 1024
- `stop_scale`: 6
- `num_iter`: 2000
- `num_train_pyramids`: 200
- `num_parallel_scales`: 2
- `save_interval`: 100
- `output_path`: results
- `gpu_device`: 0
- `use_wells`: False
- `use_seismic`: False
- `well_loss_penalty`: 10.0

See [options.py](options.py) for the complete `TrainningOptions` defaults and per-field documentation.

Below is the complete `TrainningOptions` field list and defaults (authoritative source: [options.py](options.py)).

- `alpha` (int): 10
- `batch_size` (int): 1
- `beta1` (float): 0.5
- `crop_size` (int): 256
- `discriminator_steps` (int): 3
- `num_img_channels` (int): 3
- `gamma` (float): 0.9
- `generator_steps` (int): 3
- `gpu_device` (int): 0
- `img_color_range` (tuple[int,int]): (0, 255)
- `input_path` (str): "data"
- `kernel_size` (int): 3
- `lambda_grad` (float): 0.1
- `lr_d` (float): 5e-05
- `lr_decay` (int): 1000
- `lr_g` (float): 5e-05
- `manual_seed` (int | None): None
- `max_size` (int): 1024
- `min_num_feature` (int): 32
- `min_size` (int): 12
- `noise_amp` (float): 0.1
- `min_noise_amp` (float): 0.1
- `scale0_noise_amp` (float): 1.0
- `well_loss_penalty` (float): 10.0
- `lambda_diversity` (float): 1.0
- `num_diversity_samples` (int): 3
- `num_feature` (int): 32
- `num_generated_per_real` (int): 5
- `num_iter` (int): 2000
- `num_layer` (int): 5
- `noise_channels` (int): 3
- `num_real_facies` (int): 5
- `num_train_pyramids` (int): 200
- `num_parallel_scales` (int): 2
- `num_workers` (int): 4
- `output_path` (str): "results"
- `padding_size` (int): 0
- `regen_npy_gz` (bool): False
- `save_interval` (int): 100
- `start_scale` (int): 0
- `stride` (int): 1
- `stop_scale` (int): 6
- `use_cpu` (bool): False
- `use_wells` (bool): False
- `use_seismic` (bool): False
- `wells_mask_columns` (tuple[int, ...]): ()
- `enable_tensorboard` (bool): True
- `enable_plot_facies` (bool): True

If you rely on these defaults programmatically, prefer importing `TrainningOptions` from `options.py` to ensure you always have the authoritative values.

### Generating New Facies
Generate new facies realizations from trained models:
```sh
python gen_facies.py --how_many 500 \
    --model_path results/2025_03_25_08_28_23_facies_gan \
    --out_path results/generated --plot_well_mask --wells 8
```

**Generation Options:**
* `--how_many`: Number of facies realizations to generate
* `--model_path`: Path to trained model checkpoint
* `--plot_well_mask`: Visualize well constraints
* `--wells`: Number of wells to condition on

## 🔬 Technical Details

### Multi-Scale Pyramid Architecture
FaciesGAN employs a progressive training strategy across multiple scales:

1. **Pyramid Generation**: Input facies images are interpolated to multiple resolutions using neural smoothers
2. **Progressive Training**: Start from coarse scale, progressively add finer scales
3. **Scale-Specific Networks**: Each scale has its own generator and discriminator
4. **Cached Processing**: Pyramid generation is cached using joblib for efficiency

### Interpolation Strategies
Three interpolation methods for multi-scale representations:

* **Nearest Interpolator**: Fast baseline using nearest-neighbor resampling
* **Neural Smoother**: Learned interpolation with neural networks for smooth transitions
* **Well Interpolator**: Incorporates well log constraints during interpolation

### Color Encoding
The `ColorEncoder` class manages palette-based conversions:
* Extracts unique colors from RGB facies images
* Maps RGB pixels to categorical label indices
* Supports MPS/CUDA devices with proper dtype handling
* Enables efficient categorical cross-entropy loss

### Data Management
The `DataFiles` enum centralizes all data paths:
```python
from data_files import DataFiles

# Access data paths consistently
facies_path = DataFiles.FACIES.as_data_path()
wells_path = DataFiles.WELLS.as_data_path()
```

Supports configurable patterns for:
* Image files: `*.png`
* Model checkpoints: `*.pt`
* Mapping files: `*.npz`

## 📊 Dataset Format

### Input Data Structure
```
data/
├── facies/
│   ├── xz_crossline_000.png    # Facies images (RGB)
│   ├── xz_crossline_000.pt     # NeuralSmoother models (optional)
│   └── ...
├── wells/
│   ├── xz_crossline_000.png    # Well log visualizations
│   ├── wells_maping.npz        # Well position mapping
│   └── ...
└── seismic/
    ├── xz_crossline_000.png    # Seismic data (optional)
    └── ...
```

### Data Requirements
* **Facies Images**: RGB PNG images with distinct colors for each facies class
* **Well Data**: Binary masks or labeled images showing well locations
* **Consistent Naming**: Files should follow `xz_crossline_XXX.png` pattern
* **Color Palette**: Each unique RGB value represents one facies class

## 🛠️ Advanced Usage

### Custom Training Configuration
Create custom training configurations by modifying `options.py`:

```python
from options import TrainningOptions

opts = TrainningOptions(
    min_size=12,          # Starting resolution
    max_size=256,         # Final resolution
    stop_scale=10,        # Number of pyramid scales
    num_iter=50,          # Iterations per scale
    batch_size=100,       # Batch size
    lr_g=0.0005,          # Generator learning rate
    lr_d=0.0005,          # Discriminator learning rate
    alpha=100,            # Reconstruction loss weight
    beta=0.1,             # Well conditioning weight
)
```

### Using Different Interpolators
Switch between interpolation methods:

```python
from interpolators.config import InterpolatorConfig
from interpolators.nearest import NearestInterpolator
from interpolators.neural import NeuralSmoother

# Fast nearest-neighbor interpolation
nearest = NearestInterpolator(InterpolatorConfig())

# Neural network-based smooth interpolation
neural = NeuralSmoother(model_path, InterpolatorConfig())

# Generate pyramids
pyramid = neural.interpolate(image_path, scale_list)
```

### Caching and Performance
Pyramid generation is automatically cached using joblib:

```python
from gen_pyramids import to_facies_pyramids, to_wells_pyramids

# First call computes and caches
pyramids = to_facies_pyramids(scale_list)  # Slow

# Subsequent calls use cache
pyramids = to_facies_pyramids(scale_list)  # Fast!
```

Cache is stored in `.cache/` directory. Clear it to force recomputation:
```sh
rm -rf .cache/
```

## 🔍 Code Quality

### Type Safety
The codebase uses strict type checking with Pylance:
* Full type hints throughout
* Strict mode enabled in VS Code workspace
* Type stubs for external libraries (`types-requirements.txt`)

### Linting and Formatting
Development tools configured:
* **Black**: Code formatting
* **Flake8**: Linting (with E501 line length relaxed)
* **MyPy**: Static type checking
* **Ruff**: Fast Python linter

Run quality checks:
```sh
black .
flake8 .
mypy .
```

## 📈 Recent Improvements

### Version 2.0 Updates (December 2025)
* ✅ **Interpolator Architecture**: Refactored with base class and multiple implementations
* ✅ **ColorEncoder**: Efficient palette-based RGB conversion with device support
* ✅ **Pyramid Caching**: Added joblib-based caching for 10x faster repeated training
* ✅ **C API Clarification**: C-side trainer API now uses the canonical header `trainning/mlx_trainer_api.h` and `MLXTrainer_*` symbols; the older `c_trainer_api.h` wrapper was removed.
* ✅ **DataFiles Enum**: Centralized file path management
* ✅ **Type Safety**: Complete type hints with Pylance strict mode
* ✅ **Dataset Restructure**: Organized data into facies/wells/seismic directories
* ✅ **Documentation**: Comprehensive docstrings for all modules
* ✅ **Code Quality**: Black formatting, Flake8 linting, MyPy type checking
* ✅ **Parallel Trainer**: New `Trainer` supports training multiple scales in parallel
    (use `--num_parallel_scales` to control group size). Each group consumes a
    single batch of pyramids and trains its scales concurrently.
* ✅ **TensorBoard Logging**: Training writes a global log directory under
    ``<output_path>/tensorboard_logs`` and also creates a per-scale
    SummaryWriter inside each scale folder (``<output_path>/<scale>/``) for
    easier per-scale inspection.
* ✅ **Performance Profiling**: Added `--use-profiler` flag with backend-specific
    profiling support: Chrome traces for CUDA/CPU, OS Signpost traces for MPS
    with automatic Xcode Instruments integration.

- ✅ **Well-conditioning parameter**: Added `--well-loss-penalty` (float, default
    10.0) to control the multiplier applied to well-conditioning loss terms during
    training. Set this flag to adjust how strongly generated facies honor well
    constraints.

**Implementation notes & small tips**
- `models/torch/` now contains the PyTorch-specific implementations; `models/base.py`
    contains the framework-agnostic orchestration and abstract hooks.
- `load_amp` support for scale amplitude files was added to `models/base.py` so
    pyramid amplitude files (`*.pt`) are loaded automatically during resume.
- Weight initialization: you can initialize model weights on CPU and then call
    `model.to(device)`. This reduces unnecessary GPU memory use during init. If
    a specific init requires device tensors, create them on the target device.

### Breaking Changes
* `generate.py` renamed to `gen_facies.py`
* Data directory structure changed to `data/{facies,wells,seismic}/`
* Import paths updated for interpolators package
* New DataFiles enum for consistent path access
