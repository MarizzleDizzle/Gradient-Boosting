"""
Example usage demonstrating the improvement over single decision trees:

    >>> # Compare single tree vs gradient boosting
    >>> from gradient_boosting import DecisionTreeRegressor, GradientBoostingRegressor
    >>>
    >>> # Single tree
    >>> tree = DecisionTreeRegressor(max_depth=3)
    >>> tree.fit(X_train, y_train)
    >>> tree_pred = tree.predict(X_test)
    >>> tree_mse = np.mean((y_test - tree_pred) ** 2)
    >>>
    >>> # Gradient boosting
    >>> gb = GradientBoostingRegressor(n_estimators=100, max_depth=3)
    >>> gb.fit(X_train, y_train)
    >>>
    >>> # Monitor improvement
    >>> for i, pred in enumerate(gb.staged_predict(X_test)):
    ...     if i % 20 == 0:  # Every 20 trees
    ...         mse = np.mean((y_test - pred) ** 2)
    ...         print(f"Trees: {i+1}, MSE: {mse:.4f}")

Comparison Summary:
-------------------

                    | Single Tree | Gradient Boosting
--------------------|-------------|-------------------
Training Speed      | Very Fast   | Slower
Prediction Speed    | Very Fast   | Fast
Accuracy            | Moderate    | High
Interpretability    | High        | Low
Overfitting Risk    | High        | Low (with tuning)
Hyperparameters     | Few         | Several
Memory Usage        | Low         | Moderate
Variance            | High        | Low

The key insight: Gradient Boosting trades speed and simplicity for
significantly better accuracy by combining many weak learners.

Gradient Boosting Implementation for Regression and Classification
===================================================================

What is Gradient Boosting?
---------------------------
Gradient Boosting is an ensemble machine learning technique that combines
multiple weak learners (typically decision trees) to create a strong predictor.
Unlike a single decision tree, gradient boosting builds trees sequentially,
where each new tree corrects the errors of all previous trees.

The "Boosting" Process:
    Think of it as a team of experts where each person focuses on fixing
    the mistakes made by the team so far:
    
    Round 1: Expert 1 makes initial predictions (often has errors)
    Round 2: Expert 2 learns to correct Expert 1's mistakes
    Round 3: Expert 3 learns to correct remaining mistakes
    ...
    Final: Combine all expert opinions for best prediction

The "Gradient" Part:
    The algorithm uses gradient descent optimization. Each tree is fitted
    to the negative gradient (residuals) of the loss function with respect
    to the current model's predictions.

Key Differences from Single Decision Tree:
-------------------------------------------

Single Decision Tree:
    - ONE model making predictions
    - Learns ALL patterns at once
    - Prediction = Tree(X)
    - Fast but less accurate
    - High variance (sensitive to data changes)

Gradient Boosting:
    - MANY models working together (typically 50-500 trees)
    - Each model learns REMAINING patterns (errors)
    - Prediction = Initial + LR*Tree₁(X) + LR*Tree₂(X) + ... + LR*TreeN(X)
    - Slower but much more accurate
    - Low variance (robust to data changes)

The Algorithm Step-by-Step:
----------------------------

For Regression (predicting continuous values):

    1. Initialize: F₀(x) = mean(y)
       Start with a simple guess (average of all target values)
    
    2. For m = 1 to M (number of trees):
        a. Calculate residuals: rᵢ = yᵢ - F_{m-1}(xᵢ)
           Residuals = actual - current prediction
        
        b. Fit tree hₘ(x) to predict residuals
           Train a decision tree on these errors
        
        c. Update: F_m(x) = F_{m-1}(x) + learning_rate * hₘ(x)
           Add the tree's predictions to current model
    
    3. Final prediction: F_M(x)
       Sum of initial guess + all tree contributions

Concrete Example:
    Predicting house price = $300,000
    
    Initial (F₀): $250,000 (average of all houses)
    Error: $50,000
    
    Tree 1 learns: "When bedrooms>3, add $40k"
    Prediction: $250k + 0.1*$400k = $290k
    Error: $10,000
    
    Tree 2 learns: "When sqft>1800, add $8k"
    Prediction: $290k + 0.1*$80k = $298k
    Error: $2,000
    
    Tree 3 learns: "When age<5, add $2k"
    Prediction: $298k + 0.1*$20k = $300k
    Error: $0 ✓

Mathematical Foundation:
------------------------

Objective: Minimize loss function L(y, F(x))

For regression with squared loss:
    L(y, F) = ½(y - F)²
    
    Gradient: ∂L/∂F = -(y - F) = -residual
    
    So fitting to negative gradient = fitting to residuals

For classification with log loss:
    L(y, F) = -[y*log(p) + (1-y)*log(1-p)]
    where p = sigmoid(F)
    
    Gradient: ∂L/∂F = p - y

The learning rate controls the contribution of each tree:
    - Higher rate (0.3): Faster learning, more overfitting risk
    - Lower rate (0.01): Slower learning, needs more trees, often better

Hyperparameters Explained:
---------------------------

n_estimators: Number of trees to build
    - More trees = better fit, but longer training
    - Too few: underfitting
    - Too many: overfitting (eventually), longer training
    - Typical: 50-500
    - Recommendation: Start with 100

learning_rate: Shrinkage parameter
    - Controls how much each tree contributes
    - Lower = more robust, needs more trees
    - Higher = faster training, more overfitting risk
    - Typical: 0.01-0.3
    - Recommendation: 0.1 (or lower with more trees)
    - Rule: learning_rate * n_estimators ≈ constant

max_depth: Maximum depth of each tree
    - Deeper = more complex interactions
    - Gradient boosting uses SHALLOW trees (unlike single trees)
    - Typical: 2-5 (called "stumps" if depth=1)
    - Recommendation: 3-4
    - Why shallow? Many simple trees > few complex trees

subsample: Fraction of samples for each tree
    - < 1.0 adds randomness (stochastic gradient boosting)
    - Reduces overfitting, speeds up training
    - Typical: 0.5-1.0
    - Recommendation: 0.8

min_samples_split, min_samples_leaf: Tree constraints
    - Control individual tree complexity
    - Higher = simpler trees, less overfitting
    - Usually use defaults (2, 1)

When to Use Gradient Boosting:
-------------------------------

Use Gradient Boosting when:
    ✓ Accuracy is critical (competitions, production)
    ✓ You have structured/tabular data
    ✓ Dataset is medium to large (1000+ samples)
    ✓ You can afford training time
    ✓ You need feature importance
    ✓ Data has complex non-linear patterns

DON'T use Gradient Boosting when:
    ✗ You need real-time training
    ✗ Interpretability is crucial (use single tree)
    ✗ Dataset is very small (<100 samples)
    ✗ Data is high-dimensional sparse (text, images - use deep learning)

Real-World Applications:
------------------------
- Ranking: Search engines, recommendation systems
- Finance: Credit scoring, fraud detection, risk assessment
- Healthcare: Disease prediction, treatment outcome
- E-commerce: Click-through rate, conversion prediction
- Insurance: Claim severity, customer churn

Gradient boosting (especially XGBoost, LightGBM) is often the winning
solution in machine learning competitions on tabular data.

Examples:
---------

Example 1: Basic Regression
    >>> from gradient_boosting import GradientBoostingRegressor
    >>> import numpy as np
    >>> 
    >>> # Data: [feature1, feature2] -> target
    >>> X = np.array([[1, 2], [2, 3], [3, 4], [4, 5]])
    >>> y = np.array([3, 5, 7, 9])
    >>> 
    >>> # Train model (builds 50 trees)
    >>> model = GradientBoostingRegressor(n_estimators=50, learning_rate=0.1)
    >>> model.fit(X, y)
    >>> 
    >>> # Predict
    >>> predictions = model.predict(np.array([[2.5, 3.5]]))
    >>> print(predictions)  # Should be close to 6

Example 2: Classification
    >>> from gradient_boosting import GradientBoostingClassifier
    >>> 
    >>> # Binary classification data
    >>> X = np.array([[1, 2], [2, 3], [3, 1], [4, 2]])
    >>> y = np.array([0, 0, 1, 1])  # 0 or 1
    >>> 
    >>> # Train
    >>> clf = GradientBoostingClassifier(n_estimators=50)
    >>> clf.fit(X, y)
    >>> 
    >>> # Predict class and probability
    >>> pred_class = clf.predict(np.array([[2.5, 2.5]]))
    >>> pred_prob = clf.predict_proba(np.array([[2.5, 2.5]]))

Example 3: Monitoring Training
    >>> model = GradientBoostingRegressor(n_estimators=100)
    >>> model.fit(X_train, y_train)
    >>> 
    >>> # Check if model is still improving
    >>> print(model.training_loss[:10])  # First 10 iterations
    >>> print(model.training_loss[-10:]) # Last 10 iterations
    >>> # Loss should decrease over time

Example 4: Staged Predictions (Model Evolution)
    >>> model = GradientBoostingRegressor(n_estimators=100)
    >>> model.fit(X_train, y_train)
    >>> 
    >>> # See how predictions improve with more trees
    >>> for i, pred in enumerate(model.staged_predict(X_test)):
"""
import numpy as np
from .decision_tree import DecisionTreeRegressor
from .loss_functions import SquaredLoss, LogLoss


class GradientBoostingRegressor:
    """
    Gradient Boosting Regressor for Continuous Target Prediction

    An ensemble method that builds an additive model by sequentially adding
    decision trees, where each tree fits the residual errors of the previous
    ensemble. This creates a strong predictor from many weak learners.

    How It Works:
    -------------
    Instead of building one large complex tree, gradient boosting builds
    many small simple trees. Each new tree focuses on correcting the mistakes
    (residuals) of all previous trees combined.

    Process:
        1. Start with initial prediction (mean of targets)
        2. Calculate errors: actual_value - prediction
        3. Build a small tree to predict these errors
        4. Update predictions: prediction += learning_rate * tree_prediction
        5. Repeat steps 2-4 for n_estimators iterations
        6. Final prediction = sum of all tree contributions

    Why Multiple Small Trees?
    --------------------------
    - One big tree: Memorizes training data (overfits)
    - Many small trees: Each captures a small pattern
    - Combined: They create complex patterns without overfitting
    - This is "ensemble learning" - wisdom of the crowd

    Parameters:
    -----------
    n_estimators : int, default=100
        The number of boosting stages (trees) to build.

        Effect:
            - More trees → better fit on training data
            - Too few → underfitting (doesn't learn enough)
            - Too many → overfitting (eventually), longer training

        Guidelines:
            - Start with 100
            - If underfitting: increase to 200-500
            - If overfitting: decrease or lower learning_rate
            - Watch training_loss to see if still improving

    learning_rate : float, default=0.1
        Shrinks the contribution of each tree.

        Effect:
            - Lower rate → more robust, needs more trees
            - Higher rate → faster training, more overfitting risk
            - Formula: final_prediction += learning_rate * tree_prediction

        Guidelines:
            - Start with 0.1
            - For production: try 0.01-0.05 with more trees
            - Trade-off: learning_rate × n_estimators ≈ constant
            - Lower is almost always better (with enough trees)

    max_depth : int, default=3
        Maximum depth of each individual tree.

        Effect:
            - Deeper → captures more complex interactions
            - Gradient boosting uses SHALLOW trees (unlike single trees)
            - Depth 1 = "stumps" (single split)
            - Depth 3 = up to 8 leaf nodes

        Guidelines:
            - Start with 3
            - Range: 2-5 for most problems
            - Rarely need > 8
            - Many shallow trees > few deep trees

    min_samples_split : int, default=2
        Minimum samples required to split an internal node.
        Higher values prevent overfitting.

    min_samples_leaf : int, default=1
        Minimum samples required in a leaf node.
        Higher values create smoother predictions.

    subsample : float, default=1.0
        Fraction of samples used to fit each tree.

        Effect:
            - < 1.0 adds randomness (stochastic gradient boosting)
            - Reduces overfitting
            - Speeds up training

        Guidelines:
            - 1.0: Use all data (deterministic)
            - 0.8: Good balance
            - 0.5-0.7: High variance data

    Attributes:
    -----------
    trees : list of DecisionTreeRegressor
        The collection of fitted sub-estimators.

    init_estimate : float
        The initial prediction (mean of target values).

    training_loss : list of float
        The loss at each boosting iteration on training data.
        Use this to diagnose convergence and overfitting.

    loss : SquaredLoss
        The loss function being optimized.

    Methods:
    --------
    fit(X, y)
        Build the gradient boosting model from training data.

    predict(X)
        Predict target values for samples in X.

    staged_predict(X)
        Predict at each boosting stage (useful for monitoring).

    Examples:
    ---------

    Example 1: Basic Usage
        >>> from gradient_boosting import GradientBoostingRegressor
        >>> import numpy as np
        >>>
        >>> # Training data
        >>> X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
        >>> y = np.array([10, 20, 30, 40])
        >>>
        >>> # Create and train model
        >>> gb = GradientBoostingRegressor(
        ...     n_estimators=100,
        ...     learning_rate=0.1,
        ...     max_depth=3
        ... )
        >>> gb.fit(X, y)
        >>>
        >>> # Make predictions
        >>> predictions = gb.predict(np.array([[2, 3], [6, 7]]))
        >>> print(predictions)  # Should be close to [15, 35]

    Example 2: Monitoring Training Progress
        >>> gb = GradientBoostingRegressor(n_estimators=100)
        >>> gb.fit(X_train, y_train)
        >>>
        >>> # Plot training loss
        >>> import matplotlib.pyplot as plt
        >>> plt.plot(gb.training_loss)
        >>> plt.xlabel('Iteration')
        >>> plt.ylabel('Training Loss')
        >>> plt.title('Convergence of Gradient Boosting')
        >>> plt.show()
        >>>
        >>> # Check if still improving
        >>> recent_improvement = gb.training_loss[-10] - gb.training_loss[-1]
        >>> if recent_improvement < 0.001:
        ...     print("Model has converged")

    Example 3: Hyperparameter Tuning
        >>> from sklearn.model_selection import train_test_split
        >>> from sklearn.metrics import mean_squared_error
        >>>
        >>> X_train, X_val, y_train, y_val = train_test_split(X, y)
        >>>
        >>> # Try different configurations
        >>> configs = [
        ...     {'n_estimators': 50, 'learning_rate': 0.1, 'max_depth': 3},
        ...     {'n_estimators': 100, 'learning_rate': 0.05, 'max_depth': 3},
        ...     {'n_estimators': 200, 'learning_rate': 0.01, 'max_depth': 4},
        ... ]
        >>>
        >>> best_mse = float('inf')
        >>> for config in configs:
        ...     model = GradientBoostingRegressor(**config)
        ...     model.fit(X_train, y_train)
        ...     pred = model.predict(X_val)
        ...     mse = mean_squared_error(y_val, pred)
        ...     if mse < best_mse:
        ...         best_mse = mse
        ...         best_config = config
        >>>
        >>> print(f"Best config: {best_config}")

    Example 4: Staged Predictions
        >>> model = GradientBoostingRegressor(n_estimators=100)
        >>> model.fit(X_train, y_train)
        >>>
        >>> # See how predictions evolve
        >>> X_test_sample = X_test[:1]  # One sample
        >>> for i, pred in enumerate(model.staged_predict(X_test_sample)):
        ...     if i % 20 == 0:
        ...         print(f"After {i+1} trees: prediction = {pred[0]:.2f}")

    Example 5: Handling Overfitting
        >>> # If training MSE << test MSE (overfitting):
        >>>
        >>> # Solution 1: Regularization
        >>> model = GradientBoostingRegressor(
        ...     n_estimators=100,
        ...     learning_rate=0.01,  # Lower rate
        ...     max_depth=2,         # Shallower trees
        ...     subsample=0.8,       # Stochastic boosting
        ...     min_samples_split=10 # Conservative splits
        ... )
        >>>
        >>> # Solution 2: Early stopping (manual)
        >>> model = GradientBoostingRegressor(n_estimators=500)
        >>> model.fit(X_train, y_train)
        >>>
        >>> best_iter = 0
        >>> best_val_mse = float('inf')
        >>> for i, pred in enumerate(model.staged_predict(X_val)):
        ...     mse = mean_squared_error(y_val, pred)
        ...     if mse < best_val_mse:
        ...         best_val_mse = mse
        ...         best_iter = i + 1
        >>>
        >>> print(f"Optimal n_estimators: {best_iter}")

    Example 6: Complex Pattern Learning
        >>> # Non-linear pattern: y = sin(x) + noise
        >>> X = np.linspace(0, 10, 100).reshape(-1, 1)
        >>> y = np.sin(X.ravel()) + np.random.normal(0, 0.1, 100)
        >>>
        >>> # Single tree struggles with smooth functions
        >>> from gradient_boosting import DecisionTreeRegressor
        >>> tree = DecisionTreeRegressor(max_depth=5)
        >>> tree.fit(X, y)
        >>> tree_pred = tree.predict(X)  # Step-like predictions
        >>>
        >>> # Gradient boosting learns smooth approximation
        >>> gb = GradientBoostingRegressor(n_estimators=50, max_depth=2)
        >>> gb.fit(X, y)
        >>> gb_pred = gb.predict(X)  # Smooth sine wave
        >>>
        >>> # GB MSE will be much lower than single tree

    Tips for Success:
    -----------------
    1. **Start Simple**: Use default parameters first

    2. **Monitor Training**: Check training_loss to ensure convergence

    3. **Validate Performance**: Always use separate validation set

    4. **Tune Learning Rate**: Lower is usually better (with more trees)

    5. **Keep Trees Shallow**: max_depth=3-5 usually optimal

    6. **Use Subsample**: 0.8 adds robustness without much cost

    7. **Scale Features**: Not required, but can help

    8. **Check for Overfitting**: Compare train vs validation error

    Common Pitfalls:
    ----------------
    ❌ Using too deep trees (max_depth > 8)
       → Overfitting, slow training

    ❌ Too high learning_rate (> 0.3)
       → Unstable, poor generalization

    ❌ Not enough trees with low learning_rate
       → Underfitting

    ❌ No validation set
       → Can't detect overfitting

    See Also:
    ---------
    DecisionTreeRegressor : Single tree (building block)
    GradientBoostingClassifier : For classification tasks
    example_regression.py : Complete workflow example

    Notes:
    ------
    - Time complexity: O(n_estimators * n_samples * n_features * max_depth)
    - Space complexity: O(n_estimators * max_depth)
    - Suitable for: Tabular data with 100-1M samples
    - Not suitable for: Images, text (use deep learning instead)

    This implementation minimizes squared loss: L(y, F) = ½(y - F)²
    """

    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=3,
                 min_samples_split=2, min_samples_leaf=1, subsample=1.0):
        """
        Initialize Gradient Boosting Regressor.

        Parameters:
        -----------
        n_estimators : int
            Number of boosting stages (trees)
        learning_rate : float
            Learning rate shrinks the contribution of each tree
        max_depth : int
            Maximum depth of individual trees
        min_samples_split : int
            Minimum samples required to split a node
        min_samples_leaf : int
            Minimum samples required at a leaf node
        subsample : float
            Fraction of samples to be used for fitting individual trees
        """
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.subsample = subsample
        self.loss = SquaredLoss()
        self.trees = []
        self.init_estimate = None
        self.training_loss = []

    def fit(self, X, y):
        """
        Build the gradient boosting model from training data.


        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Training data
        y : array-like, shape (n_samples,)
            Target values

        Returns:
        --------
        self
        """
        X = np.array(X)
        y = np.array(y)

        # Initialize with the mean
        self.init_estimate = self.loss.init_estimate(y)
        y_pred = np.full(y.shape, self.init_estimate)

        self.trees = []
        self.training_loss = []

        for i in range(self.n_estimators):
            # Calculate negative gradient (residuals)
            residuals = self.loss.gradient(y, y_pred)

            # Subsample if required
            if self.subsample < 1.0:
                n_samples = int(self.subsample * X.shape[0])
                indices = np.random.choice(X.shape[0], n_samples, replace=False)
                X_subset = X[indices]
                residuals_subset = residuals[indices]
            else:
                X_subset = X
                residuals_subset = residuals

            # Fit a tree to the residuals
            tree = DecisionTreeRegressor(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf
            )
            tree.fit(X_subset, residuals_subset)

            # Update predictions
            update = tree.predict(X)
            y_pred += self.learning_rate * update

            # Store the tree
            self.trees.append(tree)

            # Calculate and store training loss
            loss = self.loss(y, y_pred)
            self.training_loss.append(loss)

        return self

    def predict(self, X):
        """
        Predict target values for X using the trained ensemble.

        This method combines predictions from all trained trees to make
        the final prediction. Each tree contributes a small correction,
        and all contributions are summed together.

        Formula:
        --------
        prediction = init_estimate + Σ(learning_rate * tree_i.predict(X))

        Example:
            init_estimate = 100
            tree1 predicts: +10
            tree2 predicts: +5
            tree3 predicts: +2
            learning_rate = 0.1

            final = 100 + 0.1*10 + 0.1*5 + 0.1*2 = 101.7

        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Input samples to predict. Must have the same number of
            features as the training data.

            Examples:
                Single sample: [[1.5, 2.3, 4.1]]
                Multiple samples: [[1.5, 2.3], [3.2, 1.8], [2.1, 4.5]]

        Returns:
        --------
        y_pred : ndarray, shape (n_samples,)
            Predicted target values for each input sample.

        Examples:
        ---------

        Example 1: Single prediction
            >>> model = GradientBoostingRegressor()
            >>> model.fit(X_train, y_train)
            >>>
            >>> # Predict for one house
            >>> new_house = [[1500, 3, 10]]  # sqft, beds, age
            >>> price = model.predict(new_house)
            >>> print(f"Predicted price: ${price[0]:.2f}k")

        Example 2: Batch predictions
            >>> # Predict for multiple samples
            >>> new_houses = [
            ...     [1500, 3, 10],
            ...     [2000, 4, 5],
            ...     [1200, 2, 15]
            ... ]
            >>> prices = model.predict(new_houses)
            >>> for i, price in enumerate(prices):
            ...     print(f"House {i+1}: ${price:.2f}k")

        Example 3: Comparing with training predictions
            >>> # Training predictions
            >>> train_pred = model.predict(X_train)
            >>> train_error = np.mean((y_train - train_pred) ** 2)
            >>>
            >>> # Test predictions
            >>> test_pred = model.predict(X_test)
            >>> test_error = np.mean((y_test - test_pred) ** 2)
            >>>
            >>> print(f"Training MSE: {train_error:.4f}")
            >>> print(f"Test MSE: {test_error:.4f}")
            >>>
            >>> if test_error > train_error * 2:
            ...     print("Warning: Possible overfitting")

        Example 4: Prediction uncertainty
            >>> # GB doesn't provide uncertainty directly, but you can
            >>> # use staged predictions to estimate stability
            >>> staged_preds = list(model.staged_predict(X_test))
            >>>
            >>> # Check variance in later stages (should be low)
            >>> last_10 = np.array(staged_preds[-10:])
            >>> prediction_std = np.std(last_10, axis=0)
            >>>
            >>> # High std = unstable prediction (may need more data/tuning)
            >>> print(f"Prediction stability: {prediction_std}")

        Notes:
        ------
        - Time complexity: O(n_samples * n_estimators * max_depth)
        - Space complexity: O(n_samples) for output array
        - Much faster than training (no optimization needed)
        - Can be batched for large datasets
        - Predictions are deterministic (same input → same output)

        Performance:
        ------------
        - Single sample: ~microseconds
        - 1000 samples: ~milliseconds
        - 1M samples: ~seconds
        - Faster than neural networks, slower than linear models

        Tips:
        -----
        - Always check predictions on validation data first
        - Look for unrealistic predictions (too high/low)
        - Compare against simple baseline (mean, median)
        - Monitor prediction distribution vs training distribution

        Common Issues:
        --------------
        Issue: Predictions all near the mean
        → Model didn't learn (underfitting)
        → Try: more trees, deeper trees, higher learning_rate

        Issue: Predictions exactly match training data
        → Severe overfitting
        → Try: shallower trees, fewer trees, lower learning_rate

        Issue: Some predictions are NaN
        → Check input data for NaN values
        → Check if features have same scale as training

        Issue: Predictions way outside training range
        → Extrapolation (model unreliable outside training domain)
        → Solution: Only predict on similar data to training
        """
        X = np.array(X)
        y_pred = np.full(X.shape[0], self.init_estimate)

        for tree in self.trees:
            y_pred += self.learning_rate * tree.predict(X)

        return y_pred

    def staged_predict(self, X):
        """
        Predict at each stage for visualization purposes.

        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Input data

        Yields:
        -------
        array : Predictions after each boosting stage
        """
        X = np.array(X)
        y_pred = np.full(X.shape[0], self.init_estimate)
        yield y_pred.copy()

        for tree in self.trees:
            y_pred += self.learning_rate * tree.predict(X)
            yield y_pred.copy()


class GradientBoostingClassifier:
    """
    Gradient Boosting for binary classification problems.
    """

    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=3,
                 min_samples_split=2, min_samples_leaf=1, subsample=1.0):
        """
        Initialize Gradient Boosting Classifier.

        Parameters:
        -----------
        n_estimators : int
            Number of boosting stages (trees)
        learning_rate : float
            Learning rate shrinks the contribution of each tree
        max_depth : int
            Maximum depth of individual trees
        min_samples_split : int
            Minimum samples required to split a node
        min_samples_leaf : int
            Minimum samples required at a leaf node
        subsample : float
            Fraction of samples to be used for fitting individual trees
        """
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.subsample = subsample
        self.loss = LogLoss()
        self.trees = []
        self.init_estimate = None
        self.training_loss = []

    def fit(self, X, y):
        """
        Fit the gradient boosting classifier.

        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Training data
        y : array-like, shape (n_samples,)
            Target values (0 or 1)

        Returns:
        --------
        self
        """
        X = np.array(X)
        y = np.array(y)

        # Initialize with log-odds
        self.init_estimate = self.loss.init_estimate(y)
        F = np.full(y.shape, self.init_estimate)

        self.trees = []
        self.training_loss = []

        for i in range(self.n_estimators):
            # Calculate probabilities
            y_pred = self.loss.sigmoid(F)

            # Calculate negative gradient (residuals)
            residuals = self.loss.gradient(y, y_pred)

            # Subsample if required
            if self.subsample < 1.0:
                n_samples = int(self.subsample * X.shape[0])
                indices = np.random.choice(X.shape[0], n_samples, replace=False)
                X_subset = X[indices]
                residuals_subset = residuals[indices]
            else:
                X_subset = X
                residuals_subset = residuals

            # Fit a tree to the residuals
            tree = DecisionTreeRegressor(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf
            )
            tree.fit(X_subset, residuals_subset)

            # Update F
            update = tree.predict(X)
            F += self.learning_rate * update

            # Store the tree
            self.trees.append(tree)

            # Calculate and store training loss
            y_pred = self.loss.sigmoid(F)
            loss = self.loss(y, y_pred)
            self.training_loss.append(loss)

        return self

    def predict_proba(self, X):
        """
        Predict class probabilities for X.

        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Input data

        Returns:
        --------
        array : Predicted probabilities
        """
        X = np.array(X)
        F = np.full(X.shape[0], self.init_estimate)

        for tree in self.trees:
            F += self.learning_rate * tree.predict(X)

        return self.loss.sigmoid(F)

    def predict(self, X):
        """
        Predict class labels for X.

        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Input data

        Returns:
        --------
        array : Predicted class labels (0 or 1)
        """
        probas = self.predict_proba(X)
        return (probas >= 0.5).astype(int)

    def staged_predict_proba(self, X):
        """
        Predict probabilities at each stage for visualization purposes.

        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Input data

        Yields:
        -------
        array : Predicted probabilities after each boosting stage
        """
        X = np.array(X)
        F = np.full(X.shape[0], self.init_estimate)
        yield self.loss.sigmoid(F.copy())

        for tree in self.trees:
            F += self.learning_rate * tree.predict(X)
            yield self.loss.sigmoid(F.copy())
"""
Gradient Boosting Package
A comprehensive implementation of the Gradient Boosting algorithm for regression and classification.
"""

from .gradient_boosting import GradientBoostingRegressor, GradientBoostingClassifier
from .decision_tree import DecisionTreeRegressor
from .loss_functions import SquaredLoss, LogLoss

__all__ = [
    'GradientBoostingRegressor',
    'GradientBoostingClassifier',
    'DecisionTreeRegressor',
    'SquaredLoss',
    'LogLoss'
]

__version__ = '1.0.0'

