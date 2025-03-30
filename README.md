# Efficient-Image-Restoration-Using-SwinIR-TensorRT-and-TritonServer

This project accelerates SwinIR-based image restoration using TensorRT and TritonServer for efficient deployment. SwinIR achieves state-of-the-art performance in image super-resolution, denoising, and compression artifact reduction. This repository focuses on converting SwinIR models to TensorRT and deploying them with TritonServer.

### Prerequisites

- **GPU**: Compute Capability 8.0 or higher (Ampere+)
- **Docker**: Nvidia NGC containers
  - `nvcr.io/nvidia/pytorch:24.03-py3` for model conversion
  - `nvcr.io/nvidia/tritonserver:24.03-py3` for TritonServer

### Model Conversion

- **PTH → ONNX**: Convert the SwinIR model from PTH to ONNX format.
- **ONNX → Plan**: Use TensorRT to convert the ONNX model to an optimized engine file (.plan).

### Benchmark (A40)

| Model Type      | Latency (ms) |
|-----------------|--------------|
| Dynamic         | 541          |
| Static 256x256  | 378          |
| Static 512x512  | 1,606        |

### TritonServer Setup

- **Model Repository Structure**: Organize the model repository for TritonServer.
- **Run TritonServer**: Launch TritonServer to serve the model.

### Async Server Setup

- **Install Dependencies**: Install required dependencies for the asynchronous server.
- **Start Server**: Launch the custom server to interact with TritonServer.
- **Run Client**: Execute a client to interact with the server.
