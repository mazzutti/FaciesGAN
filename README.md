# FaciesGAN: Conditional SinGAN for Transversal Geological Facies Prediction

## 📌 Overview

This repository implements **FaciesGAN** to generate **transversal geological facies realizations**,
conditioned on well log data. The model learns geological patterns from an input training set and generates
high-resolution facies images that honor **well constraints**.

## ✨ Features

- **FaciesGAN architecture** adapted for geological facies modeling
- **Conditioning on well log data** to ensure realistic geological constraints
- **Unsupervised multiscale learning** from a single geological facies image
- **Facies simulation with geological consistency**
- **Easy integration with geostatistical workflows**

## 📁 Repository Structure

```
CSinGAN-Geology/
│── data/                 # Training facies images and well log masks
│── models/               # FaciesGAN generator and discriminator
│── results/              # Generated facies realizations
│── README.md             # Project documentation
│── requirements.txt      # Python dependencies
│── train.py              # The training script
│── validate.py           # The validation script
│── main.py               # The main script
```

## 🚀 Installation

### Setup

```sh
git clone https://github.com/mazzutti/FaciesGAN.git
cd FaciesGAN
pip install -r requirements.txt
```

## 🔥 Training FaciesGAN

Train the model on your facies dataset:

```sh
python main.py
```

## 📖 How It Works

1. **Multiscale training:** FaciesGAN learns geological features progressively across multiple scales.
2. **Conditional generation:** Well log data is incorporated as a constraint into the generator.
3. **Realistic facies synthesis:** The trained model generates facies that resemble real-world geology.

## 🤝 Contributing

Pull requests and contributions are welcome! Feel free to submit issues or suggest improvements.

## 📜 License

This project is licensed under the **MIT License**. See `LICENSE.md` for details.

---

📧 **Contact**: If you have any questions, reach out at `tiagomzt at gmail dot com` or open an issue!

