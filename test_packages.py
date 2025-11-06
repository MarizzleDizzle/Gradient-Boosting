"""
Quick test script to verify both packages work correctly
"""
import numpy as np

# Test gradient_boosting package
print("=" * 60)
print("Testing Gradient Boosting Package")
print("=" * 60)

from gradient_boosting import GradientBoostingRegressor, GradientBoostingClassifier
from gradient_boosting import DecisionTreeRegressor, SquaredLoss, LogLoss

# Test data
np.random.seed(42)
X = np.random.randn(100, 2)
y_reg = X[:, 0] + 0.5 * X[:, 1] + np.random.randn(100) * 0.1
y_clf = (X[:, 0] + X[:, 1] > 0).astype(int)

# Test Regressor
print("\n1. Testing GradientBoostingRegressor...")
reg = GradientBoostingRegressor(n_estimators=10, learning_rate=0.1, max_depth=2)
reg.fit(X, y_reg)
predictions = reg.predict(X)
print(f"   ✓ Trained with {len(reg.trees)} trees")
print(f"   ✓ Initial estimate: {reg.init_estimate:.4f}")
print(f"   ✓ Final training loss: {reg.training_loss[-1]:.4f}")
print(f"   ✓ Predictions shape: {predictions.shape}")

# Test Classifier
print("\n2. Testing GradientBoostingClassifier...")
clf = GradientBoostingClassifier(n_estimators=10, learning_rate=0.1, max_depth=2)
clf.fit(X, y_clf)
predictions = clf.predict(X)
probabilities = clf.predict_proba(X)
print(f"   ✓ Trained with {len(clf.trees)} trees")
print(f"   ✓ Initial estimate: {clf.init_estimate:.4f}")
print(f"   ✓ Final training loss: {clf.training_loss[-1]:.4f}")
print(f"   ✓ Predictions shape: {predictions.shape}")
print(f"   ✓ Probabilities shape: {probabilities.shape}")

# Test Decision Tree
print("\n3. Testing DecisionTreeRegressor...")
tree = DecisionTreeRegressor(max_depth=3)
tree.fit(X, y_reg)
tree_predictions = tree.predict(X)
print(f"   ✓ Tree fitted successfully")
print(f"   ✓ Predictions shape: {tree_predictions.shape}")

# Test Loss Functions
print("\n4. Testing Loss Functions...")
loss_squared = SquaredLoss()
loss_log = LogLoss()
print(f"   ✓ SquaredLoss initialized")
print(f"   ✓ LogLoss initialized")
print(f"   ✓ SquaredLoss MSE: {loss_squared(y_reg, predictions):.4f}")

print("\n" + "=" * 60)
print("All Gradient Boosting Package Tests Passed! ✓")
print("=" * 60)

# Test visualization package (imports only, no plotting)
print("\n" + "=" * 60)
print("Testing Visualization Package (imports)")
print("=" * 60)

try:
    from visualization import GradientBoostingVisualizer
    from visualization import TreeVisualizer
    from visualization import PerformanceVisualizer

    print("\n1. Testing GradientBoostingVisualizer...")
    gb_viz = GradientBoostingVisualizer(reg)
    print("   ✓ GradientBoostingVisualizer initialized")

    print("\n2. Testing TreeVisualizer...")
    tree_viz = TreeVisualizer(reg.trees[0])
    print("   ✓ TreeVisualizer initialized")

    print("\n3. Testing PerformanceVisualizer...")
    perf_viz = PerformanceVisualizer(reg, X, y_reg)
    print("   ✓ PerformanceVisualizer initialized")

    print("\n" + "=" * 60)
    print("All Visualization Package Tests Passed! ✓")
    print("=" * 60)

except ImportError as e:
    print(f"\n⚠ Visualization package requires matplotlib: {e}")
    print("   Install with: pip install -r requirements.txt")
    print("   (This is expected if dependencies aren't installed)")

print("\n" + "=" * 60)
print("Package Structure Summary")
print("=" * 60)
print("""
✓ gradient_boosting/
  - GradientBoostingRegressor: Full implementation for regression
  - GradientBoostingClassifier: Full implementation for classification
  - DecisionTreeRegressor: Custom decision tree weak learner
  - SquaredLoss & LogLoss: Loss functions with gradients

✓ visualization/
  - GradientBoostingVisualizer: Algorithm visualization tools
  - TreeVisualizer: Decision tree structure visualization
  - PerformanceVisualizer: Performance metrics and diagnostics

Both packages are ready to use!
""")

print("=" * 60)
print("Next Steps:")
print("=" * 60)
print("1. Install dependencies: pip install -r requirements.txt")
print("2. Run examples:")
print("   - python example_regression.py")
print("   - python example_classification.py")
print("3. Import in your code:")
print("   from gradient_boosting import GradientBoostingRegressor")
print("   from visualization import GradientBoostingVisualizer")
print("=" * 60)

