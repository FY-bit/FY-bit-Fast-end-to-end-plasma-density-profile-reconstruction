# Fast-end-to-end-plasma-density-profile-reconstruction
📘 Overview

This project provides a real-time deep learning framework for reconstructing plasma density profiles directly from raw microwave reflectometer I/Q signals on the EAST tokamak.
Traditional physics-based inversion methods are slow and require expert tuning.
Our approach performs end-to-end reconstruction in a single forward pass — mapping time-domain I/Q signals directly to the full density profile.

🔑Key Features

Input: Raw in-phase and quadrature (I/Q) reflectometer signals of shape [12500, 6]
Output: Plasma density profile with 80 radial points
Architecture: Multi-scale Inception-based CNN with residual connections

🚀How to use

By simply installing the required Python libraries, defining the input data shape, and setting model hyperparameters, you can use the model immediately.

1️⃣ Installation
Step 1. Clone the repository
```bash
git clone https://github.com/FY-bit/FY-bit-Fast-end-to-end-plasma-density-profile-reconstruction.git
cd FY-bit-Fast-end-to-end-plasma-density-profile-reconstruction
```

Step 2. Create and activate a Python environment
```bash
conda create -n fybit python=3.9 -y
conda activate fybit
```

Step 3. Install dependencies

Example
```bash
pip install torch numpy scipy matplotlib
```
Note: For GPU acceleration, make sure your CUDA version ≥ 12.4.

2️⃣ Data Preparation
The model takes raw microwave reflectometer I/Q signals as input.
Each data sample contains six channels representing the real and imaginary components from three frequency bands

Example
```bash
x = np.random.randn(1, 12500, 6)
```

3️⃣ Define Model Hyperparameters
```bash
model = Inception1DNet_v2(input_length=12500,
                            input_channels=6,
                            initial_channels=32,
                            branch_channels_A=128,
                            branch_channels_B=256,
                            num_modules_A=6,
                            num_modules_B=4,
                            target_length=80,
                            dropout=0.0)
```

4️⃣ Use the Model
```bash
python Inception_1D.py
```
