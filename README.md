# Seismic-MambaSR

## 📌 Introduction

Numerous post-stack seismic profiles are occasionally affected by high-frequency attenuation and sparse acquisition, which leads to certain low-resolution issues. Recently, deep-learning methods have shown good performance in the super-resolution reconstruction of post-stack seismic profiles. Most methods are based on two classical architectures: convolutional neural network (CNN) and Transformer. Generally, Transformer can capture more global features compared with CNN and usually shows better reconstruction performance. However, its core operation, i.e., the self-attention mechanism, usually leads to high computational complexity. Mamba is a deep learning architecture based on the state space models that can reduce training time while maintaining performance similar to Transformer. Therefore, we first introduce Mamba into the super-resolution reconstruction of post-stack seismic profiles and propose a novel framework, called seismic super-resolution reconstruction Mamba (Seismic-MambaSR). The Seismic-MambaSR consists of three parts: a local feature extraction module (LFEM), a global feature extraction module (GFEM), and a high-resolution seismic profile reconstruction module (HSPRM). First, the LFEM is used to extract the local features of small-scale faults, fractures, and folds in the input seismic profiles using several stacked convolutional layers. Then, the GFEM adopts a number of Mamba-based multi-level residual state space blocks to extract the global features of monoclines, synclines, and regional faults in the seismic profiles. Finally, the HSPRM reconstructs the high-resolution seismic profiles using the pixel shuffle and convolution operations. In addition, we adopt the charbonnier loss as the loss function of Seismic-MambaSR to further enhance the performance of resolution enhancement. In both synthetic and field examples, the Seismic-MambaSR outperforms both CNN-based and Transformer-based methods. Also, it provides lower computational costs compared to Transformer, while being comparable to CNN-based approaches.

Dataset: [Download Link](https://drive.google.com/file/d/19gPVpDLa3USGj8g6TYZG5IoA3dJ_4MmO/view?usp=drive_link)

---

## 🛠 Installation

1. Clone this repository:

```bash
git clone https://github.com/yws2017/Seismic-MambaSR.git
cd Seismic-MambaSR
```

2. Create a conda environment:

```bash
conda create -n mamba python=3.9 -y
conda activate mamba
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

---

### 💡 Environment Notes

This code has been tested under the following environment. It may also work for other versions:

* **Ubuntu 20.04**
* **CUDA 11.8**
* **Python 3.9**

> ⚠️ If you are using a newer CUDA version (e.g., 12.x), please refer to the official GitHub pages for `causal_conv_1d` and `mamba_ssm` to find compatible versions.

---

### 🔧 Notes：Installing Mamba-related Libraries

There are three ways to install the necessary libraries for Seismic-MambaSR:

**Option 1: Install specific versions manually**

```bash
pip install causal_conv1d==1.0.0
pip install mamba_ssm==1.0.1
```

**Option 2: Using `requirements.txt` in a new conda environment**

After creating a new conda environment, install all Python libraries using:

```bash
pip install -r requirements.txt
```

> This will include `mamba_ssm`, `causal_conv1d`, and all other dependencies required for running Seismic-MambaSR.

---

## 📁 Dataset Preparation

Download the dataset from: [Seismic Dataset](https://drive.google.com/file/d/19gPVpDLa3USGj8g6TYZG5IoA3dJ_4MmO/view?usp=drive_link)

Organize it under `../data` as follows:

```
data/
├── train/
│   ├── high/   # High-resolution seismic images
│   └── low/    # Low-resolution images
├── val/
│   ├── high/
│   └── low/
└── test/
    ├── high/
    └── low/
```

---

## 🚀 Training

Run the training script from the `code` directory:

```bash
cd code
python train.py
```

The script will automatically load training and validation data from `../data/train` and `../data/val`.

---

## 🧪 Testing

Run the testing script to evaluate the trained model:

```bash
cd code
python test.py
```

The script will read low-resolution seismic profiles from `../data/test/low` and generate high-resolution results.

