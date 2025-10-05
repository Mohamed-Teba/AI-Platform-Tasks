# 🧠 Neural Graph Network Demo

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/) [![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)](https://pytorch.org/)

A simple PyTorch-based neural network implementation demonstrating forward and backward passes on a custom "Graph" model. This project showcases basic building blocks like ReLU, Sigmoid, Tanh activations, and gradient computation. Perfect for learning PyTorch fundamentals! 🚀

## 📋 Features
- **Custom Neural Network**: A single-input Graph module with multiple linear layers, activations, and a final output.
- **Forward Pass**: Computes intermediate activations and combines them for the final prediction.
- **Backward Pass**: Automatic differentiation to compute gradients.
- **Interactive Demo**: Jupyter notebook with print statements for step-by-step visualization.

## 🛠️ Installation
1. Install dependencies:
   ```
   pip install torch notebook
   ```

## 📖 Usage
Run the Jupyter notebook to see the magic in action! 📓

1. Open the notebook:
   ```
   jupyter notebook AI_Platform_Task_01.ipynb
   ```
2. Execute the cells step-by-step:
   - Import libraries 🛠️
   - Define the `Graph` model 🏗️
   - Run forward pass with input `x = 2.0` ➡️
   - Compute gradients via backward pass 🔄

### Quick Code Snippet
```python
import torch
import torch.nn as nn
import torch.nn.functional as F

# Instantiate and run
model = Graph()
x = torch.tensor(2.0, requires_grad=True)
y = model(x)
y.backward()
print(f"Gradient: {x.grad.item():.6f}")
```

## 📊 Example Output
When you run the forward pass with `x = 2.0`:

```
Input (x): 2.0000
Forward Pass:
L1 ReLU Output (A1): [1.100000023841858, 0.6000000238418579, 1.100000023841858]
L2 Sigmoid Output (A2): [0.8807970285415649, 0.05732417479157448]
Combined (S1+S2): 3.7381
Tanh Output: 0.9989
Final Output (y): 2.4977
```

Backward pass:
```
Gradient: 0.004634
```

## 📄 License
This project is open-source under the MIT License. See [LICENSE](LICENSE.Txt) for details.

