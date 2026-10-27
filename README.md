# ANN Regression — Power Plant Energy Prediction

## 📌 Project Overview

This project uses an **Artificial Neural Network (ANN)** built with PyTorch to predict power plant energy output from environmental and operating conditions.

The model learns the relationship between four input features—ambient temperature, exhaust vacuum, ambient pressure, and relative humidity—and the target variable, `PE` (produced energy).

The project covers data preprocessing, feature scaling, neural network construction, model training, loss visualization, and regression evaluation.

## 🎯 Objectives

- Explore and preprocess power plant data.
- Separate input features from the target variable.
- Split the dataset into training and testing sets.
- Standardize input features.
- Build a feedforward ANN using PyTorch.
- Train the network using backpropagation and the Adam optimizer.
- Monitor training and validation loss.
- Evaluate predictions using Mean Squared Error (MSE) and the R² score.
- Compare predicted energy output with actual values.

## 🛠️ Technologies Used

- **Python** — Programming language
- **Pandas** — Data loading and manipulation
- **NumPy** — Numerical operations
- **Matplotlib** — Training loss visualization
- **Scikit-learn** — Data splitting, feature scaling, and evaluation
- **PyTorch** — Neural network implementation and training
- **Jupyter Notebook** — Interactive development environment

## 📂 Project Structure

```text
ANN-Regression/
│
├── ANN_Regression.ipynb
├── powerplant_data.csv
├── best_model.pt       # Generated model checkpoint
└── README.md
```

`best_model.pt` is generated when the notebook trains and saves the network.

## 📊 Dataset Description

The project uses `powerplant_data.csv`, containing 9,568 records and five columns.

| Feature | Description |
|---|---|
| `AT` | Ambient temperature |
| `V` | Exhaust vacuum |
| `AP` | Ambient pressure |
| `RH` | Relative humidity |
| `PE` | Produced energy — prediction target |

The notebook uses the first four columns as input features and `PE` as the target.

### Dataset Summary

- **Total records:** 9,568
- **Total columns:** 5
- **Input features:** 4
- **Target variable:** `PE`
- **Missing values:** None reported by the notebook

## ⚙️ Project Workflow

### 1. Load the Dataset

The notebook uses Pandas to load the CSV file.

```python
import pandas as pd
import numpy as np

df = pd.read_csv("powerplant_data.csv")

print(df.head())
print(df.shape)
print(df.isnull().sum())
```

### 2. Separate Features and Target

The input variables are stored in `X`, while the produced energy target is stored in `y`.

```python
X = df.drop("PE", axis=1)
y = df["PE"]
```

### 3. Split the Dataset

The dataset is split into training and testing sets using an 80:20 ratio.

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)
```

This creates 7,654 training records and 1,914 testing records.

### 4. Standardize the Input Features

The notebook uses `StandardScaler` to scale the input features.

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.fit_transform(X_test)
```

**Improvement recommended:** Fit the scaler on the training data only, then transform the test data with the same fitted scaler. This avoids using test-set statistics during preprocessing.

```python
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

### 5. Convert Data into PyTorch Tensors

The scaled features and target values are converted into `float32` tensors.

```python
import torch

X_train_tensor = torch.tensor(
    X_train_scaled,
    dtype=torch.float32
)

y_train_tensor = torch.tensor(
    y_train.values,
    dtype=torch.float32
).view(-1, 1)

X_test_tensor = torch.tensor(
    X_test_scaled,
    dtype=torch.float32
)

y_test_tensor = torch.tensor(
    y_test.values,
    dtype=torch.float32
).view(-1, 1)
```

### 6. Create DataLoaders

PyTorch's `TensorDataset` and `DataLoader` are used to organize data into batches of 32 samples.

```python
from torch.utils.data import TensorDataset, DataLoader

train_dataset = TensorDataset(
    X_train_tensor,
    y_train_tensor
)

test_dataset = TensorDataset(
    X_test_tensor,
    y_test_tensor
)

train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=32
)
```

### 7. Build the ANN Model

The notebook defines a feedforward neural network with two hidden layers.

```python
import torch.nn as nn

class ANN(nn.Module):
    def __init__(self):
        super(ANN, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(4, 6),
            nn.ReLU(),
            nn.Linear(6, 6),
            nn.ReLU(),
            nn.Linear(6, 1)
        )

    def forward(self, x):
        return self.model(x)
```

**Architecture:**

| Layer | Configuration |
|---|---|
| Input layer | 4 features |
| Hidden layer 1 | 6 neurons, ReLU activation |
| Hidden layer 2 | 6 neurons, ReLU activation |
| Output layer | 1 neuron for energy prediction |

The model produces a continuous numerical prediction, making it suitable for regression.

### 8. Configure Training

The notebook uses Mean Squared Error as the loss function and Adam as the optimizer.

```python
import torch.optim as optim

model = ANN()

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters())
```

### 9. Train the Model

The notebook trains the network for 100 epochs. Each epoch performs forward propagation, calculates the loss, computes gradients through backpropagation, and updates the model parameters.

```python
epochs = 100

for epoch in range(epochs):
    model.train()

    for xb, yb in train_loader:
        optimizer.zero_grad()

        outputs = model(xb)
        loss = criterion(outputs, yb)

        loss.backward()
        optimizer.step()
```

The complete notebook also calculates validation loss after each epoch and plots the training and validation loss curves.

### 10. Evaluate the Model

The notebook evaluates predictions using MSE and the R² score.

```python
from sklearn.metrics import r2_score

model.eval()

with torch.no_grad():
    train_preds = model(X_train_tensor)
    test_preds = model(X_test_tensor)

    train_mse = criterion(
        train_preds,
        y_train_tensor
    )

    test_mse = criterion(
        test_preds,
        y_test_tensor
    )

print("Training MSE:", train_mse.item())
print("Testing MSE:", test_mse.item())

print(
    "R² score:",
    r2_score(y_test, test_preds.numpy())
)
```

## 📈 Results

The notebook's recorded evaluation outputs are:

| Metric | Recorded value |
|---|---:|
| Training MSE | 20.78 |
| Testing MSE | 19.40 |
| Testing R² score | 0.9322 |
| Training samples | 7,654 |
| Testing samples | 1,914 |
| Training epochs | 100 |

The recorded R² score indicates that the model explains approximately 93.22% of the variance in the test targets under the notebook's evaluation setup.

**Important:** These metrics reflect the notebook as currently implemented. The scaler is fitted separately on the test set, and the checkpoint-selection code contains a variable-name typo. Correcting these issues and rerunning the notebook may change the results.

## 📉 Visualizations

The notebook generates:

- A training and validation loss plot across epochs.
- A table comparing predicted and actual energy values.

These outputs help assess model learning and compare the network's predictions with the observed target values.

## 🔍 Key Concepts

| Concept | Description |
|---|---|
| ANN | A neural network that learns relationships between input features and target values. |
| ReLU | An activation function used in the hidden layers. |
| Forward Propagation | Computes predictions from input data. |
| Backpropagation | Computes gradients used to update network parameters. |
| Adam Optimizer | Updates model parameters using adaptive gradient estimates. |
| MSE | Measures the average squared difference between predictions and actual values. |
| R² Score | Measures how much target variance is explained by the model. |
| StandardScaler | Standardizes input features using training-set statistics when applied correctly. |

## 💻 Installation and Usage

### 1. Clone the Repository

```bash
git clone <your-repository-url>
cd ANN-Regression
```

Replace `<your-repository-url>` with your GitHub repository URL.

### 2. Install Dependencies

```bash
pip install numpy pandas matplotlib scikit-learn torch jupyter
```

### 3. Verify the Dataset

Ensure `powerplant_data.csv` is in the same directory as `ANN_Regression.ipynb`.

### 4. Run the Notebook

```bash
jupyter notebook ANN_Regression.ipynb
```

Run the cells sequentially to load the dataset, train the network, visualize losses, and evaluate predictions.

## 🔧 Recommended Improvements

- Use `scaler.transform(X_test)` instead of fitting a new scaler on the test set.
- Correct the checkpoint variable typo: `best_val_los` should be `best_val_loss`.
- Save a checkpoint only when validation loss improves.
- Consider using a separate validation set rather than using the test set during training.
- Add MAE and RMSE metrics for additional evaluation.
- Tune the number of hidden neurons, learning rate, and batch size.
- Save the fitted scaler alongside the model for consistent future predictions.

## 🚀 Applications

This regression approach can be explored for:

- Power plant energy output prediction
- Energy performance analysis
- Environmental condition modeling
- Operational planning and forecasting
- Studying relationships between operating conditions and energy production

The notebook is an educational machine learning project, not a production-ready power plant control system.

## 👨‍💻 Author

**Sohrab Ahmad Siddiqui**

B.Tech Graduate — Computer Science Engineering (Artificial Intelligence and Machine Learning)

## 📄 License

This project is intended for educational and learning purposes. Add a license file if you plan to distribute it under a specific open-source license.
