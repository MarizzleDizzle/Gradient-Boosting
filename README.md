# Gradient Boosting Implementation and Visualization

A comprehensive Python implementation of the Gradient Boosting algorithm with extensive visualization tools.

##  Project Structure

```
Gradient Boosting/
├── gradient_boosting/          # Core algorithm package
│   ├── __init__.py
│   ├── gradient_boosting.py    # Main GB implementation
│   ├── decision_tree.py        # Decision tree for weak learners
│   └── loss_functions.py       # Loss functions (MSE, LogLoss)
│
├── visualization/              # Visualization package
│   ├── __init__.py
│   ├── visualizer.py           # Main GB visualizations
│   ├── tree_visualizer.py      # Tree structure visualization
│   └── performance_visualizer.py # Performance metrics
│
├── example_regression.py       # Regression example
├── example_classification.py   # Classification example
├── requirements.txt            # Dependencies
└── README.md                   # This file
```

##  Features

### Gradient Boosting Package (`gradient_boosting`)

1. **GradientBoostingRegressor**
   - Implements gradient boosting for regression tasks
   - Configurable hyperparameters (n_estimators, learning_rate, max_depth, etc.)
   - Tracks training loss across iterations
   - Supports subsampling for stochastic gradient boosting

2. **GradientBoostingClassifier**
   - Binary classification with log loss
   - Probability predictions
   - Staged predictions for visualization

3. **DecisionTreeRegressor**
   - Custom decision tree implementation
   - Used as weak learners in the ensemble
   - Configurable depth and split criteria

4. **Loss Functions**
   - SquaredLoss for regression
   - LogLoss for binary classification
   - Gradient computation for optimization

### Visualization Package (`visualization`)

1. **GradientBoostingVisualizer**
   - Training loss curves
   - Staged predictions (1D and 2D)
   - Residuals evolution
   - Feature importance plots
   - Animated boosting process

2. **TreeVisualizer**
   - Visual tree structure diagrams
   - Text-based tree printing
   - Node and edge annotations

3. **PerformanceVisualizer**
   - Predictions vs actual values
   - Residual analysis
   - Confusion matrices
   - ROC curves
   - Learning curves

##  Installation

### Requirements

```bash
pip install -r requirements.txt
```

Required packages:
- numpy
- matplotlib
- scikit-learn (for metrics and example data)

##  Usage

### Regression Example

```python
from gradient_boosting import GradientBoostingRegressor
from visualization import GradientBoostingVisualizer

# Create and train model
model = GradientBoostingRegressor(
    n_estimators=50,
    learning_rate=0.1,
    max_depth=3
)
model.fit(X_train, y_train)

# Visualize
viz = GradientBoostingVisualizer(model)
viz.plot_training_loss()
viz.plot_staged_predictions_1d(X_train, y_train)
```

### Classification Example

```python
from gradient_boosting import GradientBoostingClassifier
from visualization import PerformanceVisualizer

# Create and train model
model = GradientBoostingClassifier(
    n_estimators=50,
    learning_rate=0.1,
    max_depth=3
)
model.fit(X_train, y_train)

# Visualize performance
perf_viz = PerformanceVisualizer(model, X_train, y_train, X_test, y_test)
perf_viz.plot_confusion_matrix()
perf_viz.plot_roc_curve()
```

### Running Examples

```bash
# Run regression example
python example_regression.py

# Run classification example
python example_classification.py
```

##  Algorithm Overview

Gradient Boosting builds an ensemble of weak learners (decision trees) sequentially:

1. **Initialize** with a constant prediction (mean for regression, log-odds for classification)
2. **For each iteration**:
   - Calculate the negative gradient (residuals)
   - Fit a decision tree to the residuals
   - Update predictions by adding the scaled tree prediction
3. **Final prediction** is the sum of all tree predictions

Key equation:
```
F_m(x) = F_{m-1}(x) + η * h_m(x)
```
Where:
- F_m(x): prediction at stage m
- η: learning rate
- h_m(x): new tree fitted to residuals

## Visualizations

### Training Loss
Shows how the loss decreases with each boosting iteration.

### Staged Predictions
- **1D**: Shows prediction curve evolution
- **2D**: Shows decision boundary evolution with contour plots

### Residuals Evolution
Histograms showing how residuals shrink over iterations.

### Feature Importance
Bar chart showing which features are most frequently used for splits.

### Performance Metrics
- Predictions vs Actual scatter plots
- Residual plots
- Confusion matrices
- ROC curves
- Learning curves (train vs test)

### Tree Structure
Visual diagram of individual decision trees with:
- Decision nodes (feature + threshold)
- Leaf nodes (prediction values)
- Split paths



##  Customization

### Hyperparameters

- `n_estimators`: Number of boosting stages (trees)
- `learning_rate`: Shrinkage parameter (0 < η ≤ 1)
- `max_depth`: Maximum depth of individual trees
- `min_samples_split`: Minimum samples to split a node
- `min_samples_leaf`: Minimum samples at leaf nodes
- `subsample`: Fraction of samples for stochastic boosting

### Adding New Loss Functions

Extend the loss function base class:

```python
class CustomLoss:
    def __call__(self, y_true, y_pred):
        # Calculate loss
        pass
    
    def gradient(self, y_true, y_pred):
        # Calculate negative gradient
        pass
    
    def init_estimate(self, y):
        # Initial prediction
        pass
```

##  Notes

- This is an educational implementation focusing on clarity
- For production use, consider scikit-learn's GradientBoostingRegressor/Classifier
- Visualizations work best with small to medium datasets
- 1D visualizations require single-feature data
- 2D visualizations require two-feature data



##  License

This project is open source and available for educational purposes.

---

**Author**: Maria Hadi
**Date**: November 2025  
**Version**: 1.0.0

