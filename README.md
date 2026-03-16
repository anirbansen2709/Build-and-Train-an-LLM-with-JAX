# Build-and-Train-an-LLM-with-JAX
https://learn.deeplearning.ai/courses/build-and-train-an-llm-with-jax

What is JAX?
1. NumPy-style, functional API: The JAX core library lets you write code that looks like numpy
2. Automatic differentiation: Gradients are first-class functions (grad, jacfwd, jacrev)-critical for ML and optimization.
3. JIT compilation: jit() compiles Python functions to optimized XLA code.
4. Vectorization & parallelism: vmap () and pmap () let you scale across
batches and devices cleanly.
5. GPU/TPU support: Same code, different hardware-no rewrites.

```
import jax.numpy as jnp
from jax import grad, vmap, jit

def predict (params, inputs):
  for W, bin params:
    outputs = jnp.dot(inputs, W) + b
    inputs = jnp.tanh (outputs)
    return outputs
def loss(params, batch):
  inputs, targets = batch
  preds = predict (params, inputs)
  return jnp.sum ((preds - targets) ** 2)

gradient_fun = jit(grad(loss))
perexample_grads = jit(
  vmap(grad(loss), in_axes=(None, 0)),
  in_shardings=..., out_shardings=...)
```

JAX Ecosystem
<img width="886" height="449" alt="Screenshot 2026-03-11 at 10 37 23 PM" src="https://github.com/user-attachments/assets/e1d3cdb6-5d36-48b3-a6f2-984f86706f3c" />

This repository contains a complete implementation of a **MiniGPT-style Large Language Model (LLM)** built from scratch using the **JAX** ecosystem. This project follows the modern "Google-stack" for machine learning, focusing on high-performance functional programming and scalable model design.

## 🚀 Project Overview

The goal of this project is to build, train, and deploy a **20-million parameter transformer** model. Leveraging **JAX** allows for significant performance benefits, including Just-In-Time (JIT) compilation and efficient hardware acceleration across CPUs, GPUs, and TPUs.

### Key Components:
* **Architecture:** Multi-head self-attention transformer using **Flax/NNX**.
* **Optimization:** Stochastic Gradient Descent and AdamW via **Optax**.
* **Data Pipeline:** High-speed, deterministic data loading with **Grain**.
* **Checkpointing:** Management of model states and metadata using **Orbax**.
* **Dataset:** Designed for training on the **TinyStories** dataset to learn complex language reasoning at a small scale.

---

## 📂 Repository Structure

| File | Description |
| :--- | :--- |
| `01_Overview_of_JAX.ipynb` | Fundamental concepts of JAX: PRNG keys, JIT, and Vmap. |
| `02_Building_the_Architecture.ipynb` | Defining the Transformer, Embedding layers, and Attention heads. |
| `03_Data_Loading.ipynb` | Implementing the Grain pipeline for tokenization and batching. |
| `04_Training_and_Saving.ipynb` | The core training loop and model persistence with Orbax. |
| `05_Final_MiniGPT.ipynb` | End-to-end inference and a simple chat interface for the model. |

---

## 🛠️ Technical Implementation Details

### 1. Functional State Management
In JAX, models are typically stateless. We use **Flax/NNX** to bridge the gap between functional JAX transformations and object-oriented model definitions, allowing for a clean, modular, and maintainable codebase.

### 2. Parallelism and Performance
The training loop utilizes `jax.jit` to compile the loss function and optimizer updates into XLA (Accelerated Linear Algebra) kernels. This significantly reduces Python overhead and maximizes hardware utilization.

### 3. Training Dynamics
* **Model Size:** ~20M parameters.
* **Precision:** Bfloat16/Float32 mixed precision.
* **Context Window:** Optimized for sequence lengths relevant to the TinyStories dataset.

---

## 📖 Getting Started

### Installation
Clone the repository and install the required dependencies:

```bash
git clone [https://github.com/anirbansen2709/Build-and-Train-an-LLM-with-JAX.git](https://github.com/anirbansen2709/Build-and-Train-an-LLM-with-JAX.git)
cd Build-and-Train-an-LLM-with-JAX
pip install jax jaxlib flax optax grain orbax-checkpoint
