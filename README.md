# FaciesGAN: Conditional SinGAN for Transversal Geological Facies Prediction

## 📌 Overview
This repository implements FaciesGAN to generate transversal geological facies realizations, conditioned on well log data. The model learns geological patterns from an input training set and generates high-resolution facies images that honor well constraints using a multi-scale progressive GAN architecture.

## ✨ Key Features
* **Multi-scale Progressive Training**: FaciesGAN architecture with pyramid-based learning from coarse to fine scales
* **Well Conditioning**: Incorporates well log data as constraints to ensure geological realism
* **Neural Interpolation**: Advanced interpolators (nearest, neural, well-based) for smooth multi-scale representations
* **Color Palette Encoding**: Efficient RGB-to-label conversion for categorical facies data
* **Cached Pyramid Generation**: Performance-optimized with joblib caching for faster training
* **Type-Safe Codebase**: Full type hints with Pylance strict mode for enhanced code quality
* **Flexible Data Management**: Centralized file handling through DataFiles enum

## 🏗️ Architecture Components

### Core Modules
* **`models/`** - Generator and Discriminator networks with custom layers
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
│   ├── generator.py         # Multi-scale generator
│   ├── discriminator.py     # Multi-scale discriminator
│   ├── custom_layer.py      # Custom neural network layers
│   └── facies_gan.py        # Main FaciesGAN model
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
    --min_size 16 --max_size 128 --stop_scale 8
```

**Key Parameters:**
* `--min_size`: Starting resolution for coarse scale (default: 16)
* `--max_size`: Final high-resolution output (default: 128)
* `--stop_scale`: Number of scales in the pyramid (default: 8)
* `--num_train_facies`: Number of facies images to use for training
* `--batch_size`: Batch size for training
* `--num_iter`: Training iterations per scale
* `--save_interval`: Save checkpoint every N iterations

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
from interpolators.nearest import NearestInterpolator
from interpolators.neural import NeuralSmoother
from interpolators.config import InterpolatorConfig

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
* ✅ **DataFiles Enum**: Centralized file path management
* ✅ **Type Safety**: Complete type hints with Pylance strict mode
* ✅ **Dataset Restructure**: Organized data into facies/wells/seismic directories
* ✅ **Documentation**: Comprehensive docstrings for all modules
* ✅ **Code Quality**: Black formatting, Flake8 linting, MyPy type checking

### Breaking Changes
* `generate.py` renamed to `gen_facies.py`
* Data directory structure changed to `data/{facies,wells,seismic}/`
* Import paths updated for interpolators package
* New DataFiles enum for consistent path access
