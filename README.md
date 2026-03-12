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

