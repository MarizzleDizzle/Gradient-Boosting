"""
Gradient Boosting Package
==========================

A comprehensive implementation of the Gradient Boosting algorithm for regression
and classification tasks.

What is a Decision Tree?
------------------------
A Decision Tree is a single model that makes predictions by learning a series of
if-then decision rules from the training data. It splits the data into branches
based on feature values, creating a tree-like structure.

Example:
    Think of it as a flowchart:
    - "Is house size > 1500 sq ft?"
        - YES: "Is neighborhood == 'downtown'?"
            - YES: Predict $400k
            - NO: Predict $300k
        - NO: Predict $200k

Key characteristics:
    - Fast to train and predict
    - Easy to understand and visualize
    - Prone to overfitting (memorizes training data)
    - High variance (small data changes cause big changes)
    - Limited accuracy when used alone

What is Gradient Boosting?
---------------------------
Gradient Boosting is an ensemble method that combines many weak decision trees
sequentially. Each new tree learns to correct the errors (residuals) made by
all previous trees combined.

The Process:
    1. Start with a simple prediction (e.g., the average)
    2. Calculate the errors (actual - predicted)
    3. Train a new tree to predict these errors
    4. Add this tree's predictions to the ensemble
    5. Repeat steps 2-4 for N iterations
    6. Final prediction = initial + sum of all tree predictions

Example:
    Predicting house price = $300k

    Step 1: Initial prediction = $250k (average)
            Error = $50k

    Step 2: Tree 1 learns to predict this $50k error
            Predicts: +$40k
            New prediction = $250k + $40k = $290k
            Error = $10k

    Step 3: Tree 2 learns to predict this $10k error
            Predicts: +$8k
            New prediction = $290k + $8k = $298k
            Error = $2k

    Continue until error is small...

Key characteristics:
    - High accuracy (often best-in-class)
    - Handles complex non-linear patterns
    - Resistant to overfitting (with proper tuning)
    - Slower to train (builds trees sequentially)
    - Harder to interpret than single tree

Relationship:
    - Decision Trees are the BUILDING BLOCKS
    - Gradient Boosting is the CONSTRUCTION METHOD
    - Together they create a powerful ensemble model

Usage Examples:
---------------

Regression Example:
    >>> from gradient_boosting import GradientBoostingRegressor
    >>> import numpy as np
    >>>
    >>> # Training data: house features and prices
    >>> X = np.array([[1500, 3], [2000, 4], [1200, 2]])  # [sqft, bedrooms]
    >>> y = np.array([300, 400, 250])  # prices in $1000s
    >>>
    >>> # Create and train model (builds 100 decision trees)
    >>> model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1)
    >>> model.fit(X, y)
    >>>
    >>> # Predict new house
    >>> new_house = np.array([[1600, 3]])
    >>> prediction = model.predict(new_house)
    >>> print(f"Predicted price: ${prediction[0]:.0f}k")

Classification Example:
    >>> from gradient_boosting import GradientBoostingClassifier
    >>>
    >>> # Training data: customer features and purchase decision
    >>> X = np.array([[25, 50000], [45, 80000], [35, 60000]])  # [age, income]
    >>> y = np.array([0, 1, 1])  # 0=no purchase, 1=purchase
    >>>
    >>> # Create and train model
    >>> model = GradientBoostingClassifier(n_estimators=50)
    >>> model.fit(X, y)
    >>>
    >>> # Predict for new customer
    >>> new_customer = np.array([[30, 55000]])
    >>> prediction = model.predict(new_customer)
    >>> probability = model.predict_proba(new_customer)
    >>> print(f"Will purchase: {prediction[0]}, Probability: {probability[0]:.2%}")

Single Decision Tree Example:
    >>> from gradient_boosting import DecisionTreeRegressor
    >>>
    >>> # Train a single tree (simpler, faster, less accurate)
    >>> tree = DecisionTreeRegressor(max_depth=3)
    >>> tree.fit(X, y)
    >>> prediction = tree.predict(new_house)

When to Use:
------------
Use Single Decision Tree when:
    - You need fast training/prediction
    - Interpretability is crucial (must explain decisions)
    - You have very limited data (<100 samples)
    - You want a quick baseline

Use Gradient Boosting when:
    - Accuracy is the top priority
    - You have sufficient data (1000+ samples recommended)
    - You can afford longer training time
    - Complex patterns need to be captured
    - You're working on competitions or production systems

See Also:
---------
- example_regression.py : Complete regression workflow with visualizations
- example_classification.py : Complete classification workflow
"""
from gradient_boosting.gradient_boosting import GradientBoostingRegressor, GradientBoostingClassifier
from gradient_boosting.decision_tree import DecisionTreeRegressor
from gradient_boosting.loss_functions import SquaredLoss, LogLoss
__all__ = [
    'GradientBoostingRegressor',
    'GradientBoostingClassifier',
    'DecisionTreeRegressor',
    'SquaredLoss',
    'LogLoss'
]
__version__ = '1.0.0'
