# CSinGAN: Conditional SinGAN for Transversal Geological Facies Prediction

## 📌 Overview

This repository implements **Conditional SinGAN (CSinGAN)** to generate **transversal geological facies realizations**,
conditioned on well log data. The model learns geological patterns from an input training set and generates
high-resolution facies images that honor **well constraints**.

## ✨ Features

- **CSinGAN architecture** adapted for geological facies modeling
- **Conditioning on well log data** to ensure realistic geological constraints
- **Unsupervised multiscale learning** from a single geological facies image
- **Facies simulation with geological consistency**
- **Easy integration with geostatistical workflows**

## 📁 Repository Structure

```
CSinGAN-Geology/
│── assets/               # Example images & visualization scripts
│── data/                 # Training facies images and well log masks
│── models/               # CSinGAN generator and discriminator
│── notebooks/            # Jupyter notebooks for experiments
│── scripts/              # Training and inference scripts
│── results/              # Generated facies realizations
│── README.md             # Project documentation
│── requirements.txt      # Python dependencies
│── train.py              # Main training script
│── generate.py           # Generate new facies realizations
```

## 🚀 Installation

### Prerequisites

- Python 3.8+
- TensorFlow 2.x
- NumPy, Matplotlib, OpenCV

### Setup

```sh
git clone https://github.com/yourusername/CSinGAN-Geology.git
cd CSinGAN-Geology
pip install -r requirements.txt
```

## 🔥 Training CSinGAN

Train the model on your facies dataset:

```sh
python train.py --epochs 5000 --lr 0.0002 --batch_size 16
```

Options:

- `--epochs`: Number of training iterations
- `--lr`: Learning rate for generator & discriminator
- `--batch_size`: Batch size for training

## 🏗️ Generating Facies Realizations

After training, generate facies images with:

```sh
python generate.py --num_samples 10 --input_image data/facies_example.png --well_mask data/well_mask.png
```

This will generate **10 facies images** while **honoring the well constraints**.

## 📊 Visualization

To visualize generated facies, use:

```sh
python scripts/visualize_results.py --results_dir results/
```

## 📖 How It Works

1. **Multiscale training:** CSinGAN learns geological features progressively across multiple scales.
2. **Conditional generation:** Well log data is incorporated as a constraint into the generator.
3. **Realistic facies synthesis:** The trained model generates facies that resemble real-world geology.

## 📌 Example Results

Generated facies images conditioned on well log constraints:

## 🔗 References

- [SinGAN: Learning a Generative Model from a Single Natural Image](https://arxiv.org/abs/1905.01164)

## 🤝 Contributing

Pull requests and contributions are welcome! Feel free to submit issues or suggest improvements.

## 📜 License

This project is licensed under the **MIT License**. See `LICENSE.md` for details.

---

📧 **Contact**: If you have any questions, reach out at `tiagomzt at gmail dot com` or open an issue!

