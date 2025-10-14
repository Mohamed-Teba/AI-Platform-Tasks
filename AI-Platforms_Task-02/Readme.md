# 🧠 Simple NN on Fashion-MNIST

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/) [![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)](https://pytorch.org/)

A simple PyTorch-based multilayer perceptron (MLP) implementation for classifying grayscale images from the Fashion-MNIST dataset. This project demonstrates loading data, building a custom NN, training with Adam optimizer, and evaluating performance to achieve >85% test accuracy. Perfect for learning PyTorch for image classification! 🚀

## 📋 Features
- **Dataset Loading**: Direct access to Fashion-MNIST via torchvision (60K train, 10K test images, 28x28 grayscale).
- **Custom Neural Network**: Fully connected MLP with input (784), hidden layers (256 ReLU, 128 ReLU), and output (10 neurons).
- **Training Loop**: Batch size 64, 5-10 epochs, CrossEntropyLoss, Adam (lr=0.001), pixel scaling to [0,1].
- **Evaluation Tools**: Test accuracy, confusion matrix, example predictions (correct/incorrect), loss/accuracy plots.
- **Interactive Demo**: Jupyter notebook with visualizations and step-by-step execution.

## 🛠️ Installation
1. Install dependencies:
   ```
   pip install torch torchvision matplotlib scikit-learn seaborn notebook
   ```

## 📖 Usage
Run the Jupyter notebook to train and evaluate the model! 📓

1. Open the notebook:
   ```
   jupyter notebook Assignment2_FashionMNIST_NN.ipynb
   ```
2. Execute the cells step-by-step:
   - Import libraries and load dataset 🛠️
   - Define the `SimpleNN` model 🏗️
   - Train for 10 epochs (monitor loss/accuracy) ➡️
   - Evaluate on test set (accuracy, confusion matrix, predictions) 🔍

### Quick Code Snippet
```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms

# Load data
transform = transforms.ToTensor()
train_dataset = torchvision.datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

# Define model
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(28*28, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 10)
    
    def forward(self, x):
        x = x.view(-1, 28*28)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

model = SimpleNN()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Train (simplified loop)
model.train()
for epoch in range(10):
    for data, target in train_loader:
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
    print(f'Epoch {epoch+1} complete')
```

## 📊 Example Output
When you run training (10 epochs):

```
Epoch 1/10, Loss: 1.2345, Accuracy: 55.23%
Epoch 2/10, Loss: 0.7890, Accuracy: 72.45%
...
Epoch 10/10, Loss: 0.2870, Accuracy: 89.60%
```

Test evaluation:
```
Test Loss: 0.3456, Test Accuracy: 85.50%
Final Training Accuracy: 89.60%
Final Test Accuracy: 85.50%
```

Confusion Matrix (snippet for classes 0-2):
| True \ Pred | 0 (T-shirt) | 1 (Trouser) | 2 (Pullover) |
|-------------|-------------|-------------|--------------|
| **0**       | 862         | 1           | 20           |
| **1**       | 2           | 943         | 0            |
| **2**       | 11          | 0           | 859          |

Example Predictions (first 8 test images):
- True: 4 (Coat), Pred: 4 – Correct
- True: 6 (Shirt), Pred: 0 (T-shirt) – Incorrect
- ... (Visualized with green/red titles in notebook)

## 📄 License
This project is open-source under the MIT License. See [LICENSE](LICENSE.Txt) for details.
