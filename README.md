# 🧠 AI Platform Tasks Repository

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/) [![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)](https://pytorch.org/) [![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-green)](https://jupyter.org/)

Welcome to the **AI Platform Tasks** repository! This project is a collection of hands-on Jupyter notebooks designed to explore fundamental concepts in artificial intelligence, machine learning, and neural networks using PyTorch. Each task builds practical skills through code implementations, visualizations, and experiments. Whether you're a beginner or brushing up on DL basics, dive in! 🚀

## 📋 Overview
This repo contains a series of progressive tasks:
- **Task 01**: Building a custom neural network with forward/backward passes (ReLU, Sigmoid, Tanh activations).
- **Task 02**: [Add description if available, e.g., "Data loading and simple training loop"].
- **Task 03**: [Add description, e.g., "Convolutional layers for image processing"].
- ... (and more as you add notebooks!)

Focus areas include:
- Neural network architectures 🏗️
- Activation functions and gradients 🔄
- Optimization and loss functions 📈
- Data handling with PyTorch datasets 📊
- Model evaluation and visualization 📊

## 🛠️ Installation
1. Clone the repo:
   ```
   git clone https://github.com/Mohamed-Teba/AI-Platform-Tasks.git
   cd AI-Platform-Tasks
   ```
2. Create a virtual environment (recommended):
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

## 📖 Usage
Run individual tasks or explore the entire series! 📓

1. Launch Jupyter Notebook:
   ```
   jupyter notebook
   ```
2. Open a task notebook (e.g., `AI_Platform_Task_01.ipynb`) and execute cells step-by-step.
3. Experiment: Tweak parameters, add plots, or extend models to see changes in real-time.

### Quick Start Example
For Task 01 (Neural Graph Network):
```python
import torch
import torch.nn as nn
import torch.nn.functional as F

# Load and run the model (see notebook for full code)
model = Graph()  # Custom model class
x = torch.tensor(2.0, requires_grad=True)
y = model(x)
y.backward()
print(f"Output: {y.item():.4f}, Gradient: {x.grad.item():.6f}")
```

### Repo Structure
```
AI-Platform-Tasks/
├── README.md              # This file 📄
├── requirements.txt       # Dependencies 📦
├── AI_Platform_Task_01.ipynb  # Neural network basics 🧠
├── AI_Platform_Task_02.ipynb  # [Next task] 🔄
├── ...                    # More tasks
└── outputs/               # Generated plots/results (optional) 📊
```

## 📊 Example Outputs
Across tasks, expect detailed logs like:
- Forward pass activations and sums
- Gradient computations
- Loss curves (in later tasks)

## 🤝 Contributing
Love the project? Contribute by:
- Adding new tasks or notebooks ✨
- Fixing bugs or improving code 🐛
- Suggesting enhancements via issues 💡

1. Fork the repo and create a branch: `git checkout -b feature/amazing-task`
2. Commit changes: `git commit -m "Add cool new task"`
3. Push and open a PR! 🎉

Report issues: [Open an issue](https://github.com/Mohamed-Teba/AI-Platform-Tasks/issues)

## 📄 License
This project is open-source under the MIT License. See [LICENSE](LICENSE) for details.

---

Built with ❤️ by Mohamed Teba | Last updated: October 06, 2025 | Questions? Reach out! 😊

---

**To download this as a .md file:** Copy the content above (starting from `# 🧠 AI Platform Tasks Repository` to the end), paste it into a text editor (like Notepad or VS Code), and save the file as `README.md`. Upload it to your GitHub repo! 💾
