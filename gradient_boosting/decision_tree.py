"""
Decision Tree Implementation for Gradient Boosting
===================================================

What is a Decision Tree?
------------------------
A Decision Tree is a supervised learning algorithm that makes predictions by
learning simple decision rules inferred from data features. It creates a
tree-like model where:

- Internal nodes represent decisions based on features
- Branches represent the outcome of decisions
- Leaf nodes represent final predictions

Visual Example:

    [Root: Size > 1500?]
           /        \\
         No         Yes
        /            \\
    [Bedrooms>2?]   [Age<10?]
      /    \\         /    \\
    No     Yes      No    Yes
    /       \\       /      \\
  $200k   $250k   $350k  $400k

How It Works:
-------------
1. **Splitting**: At each node, the algorithm asks a question about a feature
   Example: "Is square footage > 1500?"

2. **Criterion**: Choose the split that best separates the data
   - For regression: Minimize Mean Squared Error (MSE)
   - Measure: Variance reduction

3. **Recursion**: Repeat the process for each branch until:
   - Maximum depth is reached
   - Minimum samples threshold is hit
   - All samples in node are similar

4. **Prediction**: For a new sample, follow the tree's decisions until
   reaching a leaf node, then return that leaf's value

Mathematical Foundation:
------------------------
The tree minimizes the prediction error at each split.

Split Selection:
    For each possible feature f and threshold t:
        Left group: X[f] <= t
        Right group: X[f] > t

        MSE_left = mean((y_left - mean(y_left))²)
        MSE_right = mean((y_right - mean(y_right))²)

        Weighted MSE = (n_left/n_total)*MSE_left + (n_right/n_total)*MSE_right

    Choose the split with lowest Weighted MSE

Leaf Prediction:
    prediction = mean(y_samples_in_leaf)

Key Concepts:
-------------
**Overfitting**: When a tree is too deep, it memorizes training data
    - Symptoms: Perfect training accuracy, poor test accuracy
    - Solutions: Limit max_depth, increase min_samples_split

**Underfitting**: When a tree is too shallow, it can't capture patterns
    - Symptoms: Poor training and test accuracy
    - Solutions: Increase max_depth, decrease min_samples_split

**Variance**: Decision trees have high variance (sensitive to data changes)
    - Small change in data → completely different tree
    - Gradient Boosting addresses this by combining many trees

Usage in Gradient Boosting:
----------------------------
In gradient boosting, decision trees are used as "weak learners":
    - Kept intentionally small (shallow, max_depth=2-5)
    - Each tree predicts the residuals (errors) of previous trees
    - Many weak trees combine to create a strong predictor

Standalone Usage:
    >>> tree = DecisionTreeRegressor(max_depth=3)
    >>> tree.fit(X_train, y_train)
    >>> predictions = tree.predict(X_test)

Within Gradient Boosting:
    The GradientBoostingRegressor internally uses many DecisionTreeRegressors:
    >>> gb = GradientBoostingRegressor(n_estimators=100, max_depth=3)
    >>> gb.fit(X_train, y_train)  # Builds 100 trees internally

Examples:
---------

Example 1: Simple House Price Prediction
    >>> import numpy as np
    >>> from gradient_boosting import DecisionTreeRegressor
    >>>
    >>> # Data: [square_feet, bedrooms, age]
    >>> X = np.array([
    ...     [1500, 3, 10],
    ...     [2000, 4, 5],
    ...     [1200, 2, 15],
    ... ])
    >>> y = np.array([300, 400, 250])  # prices in $1000s
    >>>
    >>> # Train tree
    >>> tree = DecisionTreeRegressor(max_depth=2)
    >>> tree.fit(X, y)
    >>>
    >>> # Predict
    >>> new_house = np.array([[1600, 3, 8]])
    >>> price = tree.predict(new_house)
    >>> print(f"Predicted: ${price[0]:.0f}k")

Example 2: Understanding Tree Depth
    >>> # Shallow tree (simple, may underfit)
    >>> shallow = DecisionTreeRegressor(max_depth=1)
    >>> shallow.fit(X, y)
    >>>
    >>> # Deep tree (complex, may overfit)
    >>> deep = DecisionTreeRegressor(max_depth=10)
    >>> deep.fit(X, y)
    >>>
    >>> # Balanced tree (usually best)
    >>> balanced = DecisionTreeRegressor(max_depth=3)
    >>> balanced.fit(X, y)

Example 3: Controlling Tree Growth
    >>> tree = DecisionTreeRegressor(
    ...     max_depth=5,           # Stop at depth 5
    ...     min_samples_split=10,  # Need 10+ samples to split
    ...     min_samples_leaf=5     # Each leaf must have 5+ samples
    ... )
    >>> tree.fit(X_train, y_train)

Hyperparameters Explained:
---------------------------
max_depth: Maximum depth of the tree
    - Controls complexity
    - Higher = more complex, more overfitting risk
    - Typical values: 3-10 (standalone), 2-5 (in boosting)
    - Default: 3

min_samples_split: Minimum samples needed to split a node
    - Controls when to stop growing
    - Higher = more conservative, prevents overfitting
    - Typical values: 2-20
    - Default: 2

min_samples_leaf: Minimum samples required in a leaf
    - Ensures predictions are based on enough data
    - Higher = smoother predictions, less overfitting
    - Typical values: 1-10
    - Default: 1

Comparison with Gradient Boosting:
-----------------------------------
Single Decision Tree:
    ✓ Fast training (seconds)
    ✓ Fast prediction (microseconds)
    ✓ Easy to interpret
    ✓ No hyperparameter tuning needed
    ✗ Lower accuracy
    ✗ High variance
    ✗ Prone to overfitting

Gradient Boosting (many trees):
    ✓ High accuracy
    ✓ Low variance
    ✓ Resistant to overfitting
    ✗ Slower training (minutes)
    ✗ Harder to interpret
    ✗ Requires hyperparameter tuning

The Decision Tree is the fundamental building block. Gradient Boosting
combines many small trees to overcome the limitations of a single tree.
"""
import numpy as np


class TreeNode:
    """A node in the decision tree."""

    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature        # Feature index to split on
        self.threshold = threshold    # Threshold value for split
        self.left = left             # Left child node
        self.right = right           # Right child node
        self.value = value           # Prediction value for leaf nodes


class DecisionTreeRegressor:
    """
    Decision Tree Regressor for Regression Tasks

    A single decision tree that predicts continuous values by recursively
    splitting the data based on feature thresholds.

    How it works:
    -------------
    1. Start with all data at root node
    2. Find the best feature and threshold to split on
       (best = minimizes mean squared error)
    3. Split data into left (<=threshold) and right (>threshold)
    4. Recursively repeat for each child until stopping criteria
    5. Leaf nodes predict the average of their samples

    Stopping criteria:
    - Reached max_depth
    - Node has < min_samples_split samples
    - Split would create leaf with < min_samples_leaf samples
    - All samples in node have same target value

    Parameters:
    -----------
    max_depth : int, default=3
        Maximum depth of the tree. Deeper trees can model more complex
        patterns but are more prone to overfitting.

        Example:
            max_depth=1: Only one split (decision stump)
            max_depth=3: Up to 8 leaf nodes
            max_depth=5: Up to 32 leaf nodes

    min_samples_split : int, default=2
        Minimum number of samples required to split an internal node.
        Increasing this value prevents the tree from creating splits
        based on very few samples (reduces overfitting).

        Example:
            min_samples_split=2: Can split even with just 2 samples
            min_samples_split=20: Only splits nodes with 20+ samples

    min_samples_leaf : int, default=1
        Minimum number of samples required to be at a leaf node.
        Higher values create smoother predictions and prevent overfitting.

        Example:
            min_samples_leaf=1: Leaves can have just 1 sample (may overfit)
            min_samples_leaf=10: Each prediction based on 10+ samples

    Attributes:
    -----------
    root : TreeNode
        The root node of the fitted tree

    Methods:
    --------
    fit(X, y)
        Build the decision tree from training data

    predict(X)
        Predict target values for samples in X

    Example Usage:
    --------------
    Basic regression:
        >>> import numpy as np
        >>> from gradient_boosting import DecisionTreeRegressor
        >>>
        >>> # Generate simple data
        >>> X = np.array([[1], [2], [3], [4], [5]])
        >>> y = np.array([2, 4, 6, 8, 10])  # y = 2*x
        >>>
        >>> # Fit tree
        >>> tree = DecisionTreeRegressor(max_depth=3)
        >>> tree.fit(X, y)
        >>>
        >>> # Predict
        >>> predictions = tree.predict(np.array([[2.5], [3.5]]))
        >>> print(predictions)  # Should be close to [5, 7]

    Controlling overfitting:
        >>> # Shallow tree (may underfit)
        >>> simple_tree = DecisionTreeRegressor(max_depth=2)
        >>>
        >>> # Conservative splitting
        >>> conservative_tree = DecisionTreeRegressor(
        ...     max_depth=5,
        ...     min_samples_split=20,  # Need 20 samples to split
        ...     min_samples_leaf=10    # Each leaf has 10+ samples
        ... )

    Visualizing tree behavior:
        >>> # Trees create step-like predictions
        >>> X_line = np.linspace(0, 10, 100).reshape(-1, 1)
        >>> predictions = tree.predict(X_line)
        >>> # Plot will show step function, not smooth curve

    Notes:
    ------
    - Decision trees partition the feature space into rectangles
    - Predictions are constant within each partition (leaf)
    - Trees are greedy: each split is locally optimal, not globally
    - High variance: small data changes can create very different trees
    - In gradient boosting, many small trees combine to reduce variance

    Performance Tips:
    -----------------
    - For standalone use: max_depth=5-10 works well
    - For gradient boosting: max_depth=2-4 is typical
    - Larger datasets: can use deeper trees
    - More features: may need deeper trees
    - Noisy data: use shallower trees + min_samples constraints

    See Also:
    ---------
    GradientBoostingRegressor : Ensemble of many decision trees
    """

    def __init__(self, max_depth=3, min_samples_split=2, min_samples_leaf=1):
        """
        Initialize Decision Tree Regressor.

        Parameters:
        -----------
        max_depth : int
            Maximum depth of the tree
        min_samples_split : int
            Minimum number of samples required to split a node
        min_samples_leaf : int
            Minimum number of samples required at a leaf node
        """
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.root = None

    def fit(self, X, y):
        """
        Build the decision tree.

        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Training data
        y : array-like, shape (n_samples,)
            Target values
        """
        self.root = self._build_tree(X, y, depth=0)
        return self

    def _build_tree(self, X, y, depth):
        """
        Recursively build the decision tree.

        Parameters:
        -----------
        X : array-like
            Training data
        y : array-like
            Target values
        depth : int
            Current depth of the tree

        Returns:
        --------
        TreeNode : The root node of the subtree
        """
        n_samples, n_features = X.shape

        # Check stopping criteria
        if (depth >= self.max_depth or
            n_samples < self.min_samples_split or
            len(np.unique(y)) == 1):
            return TreeNode(value=np.mean(y))

        # Find the best split
        best_feature, best_threshold = self._find_best_split(X, y)

        if best_feature is None:
            return TreeNode(value=np.mean(y))

        # Split the data
        left_mask = X[:, best_feature] <= best_threshold
        right_mask = ~left_mask

        # Check minimum samples per leaf
        if np.sum(left_mask) < self.min_samples_leaf or np.sum(right_mask) < self.min_samples_leaf:
            return TreeNode(value=np.mean(y))

        # Recursively build left and right subtrees
        left_child = self._build_tree(X[left_mask], y[left_mask], depth + 1)
        right_child = self._build_tree(X[right_mask], y[right_mask], depth + 1)

        return TreeNode(feature=best_feature, threshold=best_threshold,
                       left=left_child, right=right_child)

    def _find_best_split(self, X, y):
        """
        Find the best feature and threshold to split on.

        Parameters:
        -----------
        X : array-like
            Training data
        y : array-like
            Target values

        Returns:
        --------
        tuple : (best_feature, best_threshold)
        """
        best_mse = float('inf')
        best_feature = None
        best_threshold = None

        n_samples, n_features = X.shape

        for feature in range(n_features):
            thresholds = np.unique(X[:, feature])

            for threshold in thresholds:
                left_mask = X[:, feature] <= threshold
                right_mask = ~left_mask

                if np.sum(left_mask) < self.min_samples_leaf or np.sum(right_mask) < self.min_samples_leaf:
                    continue

                # Calculate MSE for this split
                left_mse = np.var(y[left_mask]) * np.sum(left_mask)
                right_mse = np.var(y[right_mask]) * np.sum(right_mask)
                mse = (left_mse + right_mse) / n_samples

                if mse < best_mse:
                    best_mse = mse
                    best_feature = feature
                    best_threshold = threshold

        return best_feature, best_threshold

    def predict(self, X):
        """
        Predict target values for X.

        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Input data

        Returns:
        --------
        array : Predicted values
        """
        return np.array([self._predict_sample(sample, self.root) for sample in X])

    def _predict_sample(self, sample, node):
        """
        Predict target value for a single sample.

        Parameters:
        -----------
        sample : array-like
            Single input sample
        node : TreeNode
            Current node in the tree

        Returns:
        --------
        float : Predicted value
        """
        if node.value is not None:
            return node.value

        if sample[node.feature] <= node.threshold:
            return self._predict_sample(sample, node.left)
        else:
            return self._predict_sample(sample, node.right)

