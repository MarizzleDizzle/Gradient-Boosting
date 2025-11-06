"""
Comprehensive unit tests for the gradient boosting package
"""
import numpy as np
import pytest
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gradient_boosting import GradientBoostingRegressor, GradientBoostingClassifier
from gradient_boosting import DecisionTreeRegressor, SquaredLoss, LogLoss


class TestGradientBoostingRegressor:
    """Test GradientBoostingRegressor"""

    def test_initialization(self):
        """Test model initialization"""
        model = GradientBoostingRegressor(n_estimators=10, learning_rate=0.1, max_depth=3)
        assert model.n_estimators == 10
        assert model.learning_rate == 0.1
        assert model.max_depth == 3

    def test_fitting(self):
        """Test model fitting"""
        np.random.seed(42)
        X = np.random.randn(100, 3)
        y = X[:, 0] + 2 * X[:, 1] - 0.5 * X[:, 2] + np.random.randn(100) * 0.1

        model = GradientBoostingRegressor(n_estimators=10, learning_rate=0.1, max_depth=3)
        model.fit(X, y)

        assert len(model.trees) == 10, "Should have 10 trees"
        assert model.init_estimate is not None, "Should have initial estimate"

    def test_prediction(self):
        """Test model prediction"""
        np.random.seed(42)
        X = np.random.randn(100, 3)
        y = X[:, 0] + 2 * X[:, 1] - 0.5 * X[:, 2] + np.random.randn(100) * 0.1

        model = GradientBoostingRegressor(n_estimators=10, learning_rate=0.1, max_depth=3)
        model.fit(X, y)
        predictions = model.predict(X)

        assert predictions.shape == y.shape, "Predictions should match y shape"
        assert not np.any(np.isnan(predictions)), "No NaN predictions"

    def test_training_loss_tracking(self):
        """Test that training loss is tracked correctly"""
        np.random.seed(42)
        X = np.random.randn(100, 3)
        y = X[:, 0] + 2 * X[:, 1] - 0.5 * X[:, 2] + np.random.randn(100) * 0.1

        model = GradientBoostingRegressor(n_estimators=10, learning_rate=0.1, max_depth=3)
        model.fit(X, y)

        assert len(model.training_loss) == 10, "Should track loss for each iteration"
        assert model.training_loss[-1] < model.training_loss[0], "Loss should decrease"

    def test_staged_predictions(self):
        """Test staged predictions"""
        np.random.seed(42)
        X = np.random.randn(100, 3)
        y = X[:, 0] + 2 * X[:, 1] - 0.5 * X[:, 2] + np.random.randn(100) * 0.1

        model = GradientBoostingRegressor(n_estimators=10, learning_rate=0.1, max_depth=3)
        model.fit(X, y)
        staged_preds = list(model.staged_predict(X))

        assert len(staged_preds) == 11, "Should have predictions for each stage + initial"

    def test_performance(self):
        """Test model performance"""
        np.random.seed(42)
        X = np.random.randn(100, 3)
        y = X[:, 0] + 2 * X[:, 1] - 0.5 * X[:, 2] + np.random.randn(100) * 0.1

        model = GradientBoostingRegressor(n_estimators=10, learning_rate=0.1, max_depth=3)
        model.fit(X, y)
        predictions = model.predict(X)
        mse = np.mean((y - predictions) ** 2)

        assert mse < 1.0, f"MSE should be reasonable, got {mse}"

    def test_different_hyperparameters(self):
        """Test with different hyperparameters"""
        np.random.seed(42)
        X = np.random.randn(100, 3)
        y = X[:, 0] + 2 * X[:, 1] - 0.5 * X[:, 2] + np.random.randn(100) * 0.1

        model = GradientBoostingRegressor(
            n_estimators=5,
            learning_rate=0.05,
            max_depth=2,
            subsample=0.8
        )
        model.fit(X, y)
        preds = model.predict(X)

        assert preds.shape == y.shape


class TestGradientBoostingClassifier:
    """Test GradientBoostingClassifier"""

    def test_initialization(self):
        """Test model initialization"""
        model = GradientBoostingClassifier(n_estimators=10, learning_rate=0.1, max_depth=3)
        assert model.n_estimators == 10

    def test_fitting(self):
        """Test model fitting"""
        np.random.seed(42)
        X = np.random.randn(100, 2)
        y = (X[:, 0] + X[:, 1] > 0).astype(int)

        model = GradientBoostingClassifier(n_estimators=10, learning_rate=0.1, max_depth=3)
        model.fit(X, y)

        assert len(model.trees) == 10, "Should have 10 trees"

    def test_prediction(self):
        """Test model prediction"""
        np.random.seed(42)
        X = np.random.randn(100, 2)
        y = (X[:, 0] + X[:, 1] > 0).astype(int)

        model = GradientBoostingClassifier(n_estimators=10, learning_rate=0.1, max_depth=3)
        model.fit(X, y)
        predictions = model.predict(X)

        assert predictions.shape == y.shape, "Predictions should match y shape"
        assert set(predictions).issubset({0, 1}), "Predictions should be 0 or 1"

    def test_probability_prediction(self):
        """Test probability predictions"""
        np.random.seed(42)
        X = np.random.randn(100, 2)
        y = (X[:, 0] + X[:, 1] > 0).astype(int)

        model = GradientBoostingClassifier(n_estimators=10, learning_rate=0.1, max_depth=3)
        model.fit(X, y)
        probas = model.predict_proba(X)

        assert probas.shape == y.shape, "Probabilities should match y shape"
        assert np.all((probas >= 0) & (probas <= 1)), "Probabilities should be in [0, 1]"

    def test_training_loss_tracking(self):
        """Test that training loss is tracked correctly"""
        np.random.seed(42)
        X = np.random.randn(100, 2)
        y = (X[:, 0] + X[:, 1] > 0).astype(int)

        model = GradientBoostingClassifier(n_estimators=10, learning_rate=0.1, max_depth=3)
        model.fit(X, y)

        assert len(model.training_loss) == 10, "Should track loss"
        assert model.training_loss[-1] < model.training_loss[0], "Loss should decrease"

    def test_staged_predictions(self):
        """Test staged predictions"""
        np.random.seed(42)
        X = np.random.randn(100, 2)
        y = (X[:, 0] + X[:, 1] > 0).astype(int)

        model = GradientBoostingClassifier(n_estimators=10, learning_rate=0.1, max_depth=3)
        model.fit(X, y)
        staged_probas = list(model.staged_predict_proba(X))

        assert len(staged_probas) == 11, "Should have probas for each stage"

    def test_performance(self):
        """Test model accuracy"""
        np.random.seed(42)
        X = np.random.randn(100, 2)
        y = (X[:, 0] + X[:, 1] > 0).astype(int)

        model = GradientBoostingClassifier(n_estimators=10, learning_rate=0.1, max_depth=3)
        model.fit(X, y)
        predictions = model.predict(X)
        accuracy = np.mean(predictions == y)

        assert accuracy > 0.7, f"Accuracy should be reasonable, got {accuracy}"


class TestDecisionTree:
    """Test DecisionTreeRegressor"""

    def test_initialization(self):
        """Test tree initialization"""
        tree = DecisionTreeRegressor(max_depth=3, min_samples_split=2)
        assert tree.max_depth == 3

    def test_fitting(self):
        """Test tree fitting"""
        np.random.seed(42)
        X = np.random.randn(50, 2)
        y = X[:, 0] + X[:, 1]

        tree = DecisionTreeRegressor(max_depth=3, min_samples_split=2)
        tree.fit(X, y)

        assert tree.root is not None, "Should have root node"

    def test_prediction(self):
        """Test tree prediction"""
        np.random.seed(42)
        X = np.random.randn(50, 2)
        y = X[:, 0] + X[:, 1]

        tree = DecisionTreeRegressor(max_depth=3, min_samples_split=2)
        tree.fit(X, y)
        predictions = tree.predict(X)

        assert predictions.shape == y.shape, "Predictions should match y shape"
        assert not np.any(np.isnan(predictions)), "No NaN predictions"

    def test_shallow_tree(self):
        """Test with max_depth=1 (stump)"""
        np.random.seed(42)
        X = np.random.randn(50, 2)
        y = X[:, 0] + X[:, 1]

        stump = DecisionTreeRegressor(max_depth=1)
        stump.fit(X, y)
        stump_preds = stump.predict(X)

        assert stump_preds.shape == y.shape

    def test_min_samples_leaf(self):
        """Test with min_samples_leaf constraint"""
        np.random.seed(42)
        X = np.random.randn(50, 2)
        y = X[:, 0] + X[:, 1]

        tree = DecisionTreeRegressor(max_depth=5, min_samples_leaf=5)
        tree.fit(X, y)
        preds = tree.predict(X)

        assert preds.shape == y.shape


class TestLossFunctions:
    """Test loss functions"""

    def test_squared_loss_calculation(self):
        """Test SquaredLoss calculation"""
        loss = SquaredLoss()
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.1, 2.1, 2.9])

        loss_val = loss(y_true, y_pred)
        expected = np.mean((y_true - y_pred) ** 2)

        assert np.isclose(loss_val, expected), "Loss calculation incorrect"

    def test_squared_loss_gradient(self):
        """Test SquaredLoss gradient"""
        loss = SquaredLoss()
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.1, 2.1, 2.9])

        gradient = loss.gradient(y_true, y_pred)
        expected_grad = y_true - y_pred

        assert np.allclose(gradient, expected_grad), "Gradient calculation incorrect"

    def test_squared_loss_init_estimate(self):
        """Test SquaredLoss init estimate"""
        loss = SquaredLoss()
        y_true = np.array([1.0, 2.0, 3.0])

        init_est = loss.init_estimate(y_true)

        assert np.isclose(init_est, np.mean(y_true)), "Init estimate incorrect"

    def test_log_loss_calculation(self):
        """Test LogLoss calculation"""
        loss = LogLoss()
        y_true = np.array([0, 1, 1, 0])
        y_pred = np.array([0.1, 0.9, 0.8, 0.2])

        loss_val = loss(y_true, y_pred)

        assert loss_val > 0, "Loss should be positive"
        assert not np.isnan(loss_val), "Loss should not be NaN"

    def test_log_loss_gradient(self):
        """Test LogLoss gradient"""
        loss = LogLoss()
        y_true = np.array([0, 1, 1, 0])
        y_pred = np.array([0.1, 0.9, 0.8, 0.2])

        gradient = loss.gradient(y_true, y_pred)
        expected_grad = y_true - y_pred

        assert np.allclose(gradient, expected_grad), "Gradient calculation incorrect"

    def test_sigmoid_function(self):
        """Test sigmoid function"""
        x = np.array([-2, 0, 2])
        sig = LogLoss.sigmoid(x)

        assert np.all((sig > 0) & (sig < 1)), "Sigmoid should be in (0, 1)"
        assert np.isclose(sig[1], 0.5), "Sigmoid(0) should be 0.5"



class TestEdgeCases:
    """Test edge cases and error handling"""

    def test_single_feature(self):
        """Test with single feature"""
        X = np.random.randn(50, 1)
        y = X.flatten() + np.random.randn(50) * 0.1
        model = GradientBoostingRegressor(n_estimators=5)
        model.fit(X, y)
        preds = model.predict(X)

        assert preds.shape[0] == 50

    def test_small_dataset(self):
        """Test with small dataset"""
        X = np.random.randn(10, 2)
        y = np.random.randn(10)
        model = GradientBoostingRegressor(n_estimators=3, max_depth=2)
        model.fit(X, y)
        preds = model.predict(X)

        assert preds.shape[0] == 10

    def test_constant_target(self):
        """Test with constant target"""
        X = np.random.randn(30, 2)
        y = np.ones(30) * 5.0
        model = GradientBoostingRegressor(n_estimators=5)
        model.fit(X, y)
        preds = model.predict(X)

        assert np.allclose(preds, 5.0, atol=0.1), "Should predict constant"

    def test_imbalanced_classes(self):
        """Test with imbalanced classes"""
        X = np.random.randn(100, 2)
        y = np.zeros(100)
        y[:10] = 1  # Only 10% positive class
        model = GradientBoostingClassifier(n_estimators=5)
        model.fit(X, y)
        preds = model.predict(X)

        assert preds.shape[0] == 100

    def test_single_estimator(self):
        """Test with n_estimators=1"""
        X = np.random.randn(30, 2)
        y = np.random.randn(30)
        model = GradientBoostingRegressor(n_estimators=1)
        model.fit(X, y)

        assert len(model.trees) == 1


class TestReproducibility:
    """Test reproducibility with same seed"""

    def test_reproducibility(self):
        """Test reproducibility with same seed"""
        np.random.seed(42)
        X = np.random.randn(50, 2)
        y = np.random.randn(50)

        np.random.seed(42)
        X1 = np.random.randn(50, 2)
        y1 = np.random.randn(50)

        model1 = GradientBoostingRegressor(n_estimators=5)
        np.random.seed(123)
        model1.fit(X1, y1)
        preds1 = model1.predict(X1)

        model2 = GradientBoostingRegressor(n_estimators=5)
        np.random.seed(123)
        model2.fit(X1, y1)
        preds2 = model2.predict(X1)

        # Without subsampling, results should be identical
        assert np.allclose(preds1, preds2), "Same seed should give same results"


