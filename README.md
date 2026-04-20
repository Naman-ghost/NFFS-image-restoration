# 🧠 NFFS-image-restoration


![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![OpenCV](https://img.shields.io/badge/OpenCV-4.5+-green.svg)
![NumPy](https://img.shields.io/badge/NumPy-1.21+-lightblue.svg)
![Matplotlib](https://img.shields.io/badge/Matplotlib-3.4+-orange.svg)

🎥 **Project Explanation Video:**  
[Watch here](https://drive.google.com/file/d/1oLxkRQsbLV2dBW46GK0HYe36m8OIeGwv/view?usp=sharing)


**Neural Frequency Fusion Studio (NFFS)** is a hybrid spatial-frequency domain image transformation system. Instead of relying on global, static filters that force a trade-off between edge preservation and noise suppression, NFFS utilizes a lightweight neural decision network to adaptively blend spatial and frequency domains on a per-patch basis.

Developed as part of the CS342 Digital Image Processing curriculum.

## 🚀 Overview

Traditional Digital Image Processing (DIP) struggles with complex, spatially-variant degradations. Global spatial smoothing blurs critical edges, while global frequency low-pass filtering causes ringing. 

**NFFS solves this by:**
1. Splitting the image into localized patches.
2. Extracting dual-domain features (local variance, edge density, FFT magnitude).
3. Using a Decision Network to generate a continuous spatial weight ($\alpha$).
4. Fusing an Unsharp Mask (Spatial) and an Ideal Low-Pass Filter (Frequency) pixel-by-pixel.

## ✨ Key Features

* **Dual Representation Engine:** Processes structural gradients and FFT magnitude spectrums simultaneously.
* **Lightweight Decision Network:** Maps patch features to optimal transformation weights.
* **Explainable AI Output:** Generates a comprehensive $3 \times 4$ diagnostic grid including:
  * Decision Heatmaps (Spatial vs. Hybrid vs. Frequency)
  * Continuous Alpha Blending Maps
  * Residual Error Visualizations
* **Quantitative Validation:** Automatically calculates and compares PSNR and SSIM against single-domain baselines.

## 🛠️ Tech Stack
* **Language:** Python 3.x
* **Core Libraries:** `numpy`, `opencv-python` (cv2), `scipy`
* **Visualization:** `matplotlib` 

## ⚙️ Installation & Usage

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/yourusername/NFFS-image-restoration.git](https://github.com/yourusername/NFFS-image-restoration.git)
   cd NFFS-image-restoration
   ```

2. **Install dependencies:**
   ```bash
   pip install numpy opencv-python scipy matplotlib
   ```

3. **Run the pipeline:**
   To run the system on a synthetic test image with default degradation (Gaussian noise $\sigma=18$, blur 1.5):
   ```bash
   python nffs_pipeline.py
   ```

4. **Custom Usage:**
   You can pass a specific image and adjust the frequency cutoff or patch size via CLI arguments:
   ```bash
   python nffs_pipeline.py --image path/to/image.jpg --noise 20 --patch 32 --cutoff 0.35
   ```

## 📊 Results
Our adaptive hybrid approach consistently outperforms global filtering. On standard synthetic degradation tests, the NFFS pipeline achieves a **+6.0 dB PSNR** recovery over baseline spatial-only filters, maintaining a near-perfect SSIM by suppressing noise while preserving structural high frequencies.

*(Check the generated `nffs_output.jpg` in the root directory for a full visual breakdown of the FFT spectrums and decision maps).*

## 👥 Authors
* Naman Singh 
* Raunak Anand 
* Visvjit Kumar Singh 
