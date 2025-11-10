# FY-bit-Fast-end-to-end-plasma-density-profile-reconstruction
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
