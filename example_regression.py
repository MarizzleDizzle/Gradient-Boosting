"""
Example: Gradient Boosting Regression
Demonstrates the use of the gradient boosting package for regression tasks.
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression
from gradient_boosting import GradientBoostingRegressor
from visualization import GradientBoostingVisualizer, PerformanceVisualizer


def generate_data(n_samples=300, n_features=1):
    """Generate synthetic regression data."""
    np.random.seed(42)

    if n_features == 1:
        # Create a non-linear function for 1D visualization
        X = np.linspace(0, 10, n_samples).reshape(-1, 1)
        y = np.sin(X.ravel()) * 2 + 0.5 * X.ravel() + np.random.randn(n_samples) * 0.3
    else:
        # Multi-dimensional data
        X, y = make_regression(
            n_samples=n_samples,
            n_features=n_features,
            n_informative=min(5, n_features),
            noise=10.0,
            random_state=42
        )

    return X, y


def main():
    print("=" * 60)
    print("Gradient Boosting Regression Example")
    print("=" * 60)

    # Initialize output handler for this run
    from visualization import OutputHandler
    session_dir = OutputHandler.initialize_session()
    print(f"\nImages will be saved to: {session_dir}")

    # Generate data
    print("\n1. Generating synthetic data...")
    X, y = generate_data(n_samples=400, n_features=1)

    # Split into train and test
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42
    )

    print(f"   Training samples: {len(X_train)}")
    print(f"   Test samples: {len(X_test)}")
    print(f"   Target range: [{y_train.min():.2f}, {y_train.max():.2f}]")

    # Train model
    print("\n2. Training Gradient Boosting Regressor...")
    model = GradientBoostingRegressor(
        n_estimators=50,
        learning_rate=0.1,
        max_depth=3,
        min_samples_split=2,
        subsample=0.8
    )
    model.fit(X_train, y_train)
    print("   Training complete!")

    # Make predictions
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    # Calculate metrics
    from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
    train_mse = mean_squared_error(y_train, y_pred_train)
    test_mse = mean_squared_error(y_test, y_pred_test)
    train_r2 = r2_score(y_train, y_pred_train)
    test_r2 = r2_score(y_test, y_pred_test)
    train_mae = mean_absolute_error(y_train, y_pred_train)
    test_mae = mean_absolute_error(y_test, y_pred_test)

    print(f"\n3. Model Performance:")
    print(f"   Training MSE: {train_mse:.4f}, R²: {train_r2:.4f}, MAE: {train_mae:.4f}")
    print(f"   Test MSE: {test_mse:.4f}, R²: {test_r2:.4f}, MAE: {test_mae:.4f}")

    # Visualizations
    print("\n4. Creating visualizations...")

    # Initialize visualizers
    gb_viz = GradientBoostingVisualizer(model)
    perf_viz = PerformanceVisualizer(model, X_train, y_train, X_test, y_test)

    # Plot training loss
    print("   - Training loss curve")
    gb_viz.plot_training_loss()

    # Plot predictions vs actual (for 1D data)
    if X.shape[1] == 1:
        print("   - Predictions vs Actual (1D)")
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))

        # Sort for better visualization
        sort_idx = np.argsort(X_train.ravel())
        ax.scatter(X_train[sort_idx], y_train[sort_idx], alpha=0.5, label='Training data')
        ax.plot(X_train[sort_idx], y_pred_train[sort_idx], 'r-', linewidth=2, label='Predictions')

        sort_idx_test = np.argsort(X_test.ravel())
        ax.scatter(X_test[sort_idx_test], y_test[sort_idx_test], alpha=0.5, label='Test data', color='green')

        ax.set_xlabel('Feature')
        ax.set_ylabel('Target')
        ax.set_title('Gradient Boosting Regression: Predictions vs Actual')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()

        # Save to images folder
        output_path = OutputHandler.get_output_path('regression_predictions_1d.png')
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

    # Plot staged predictions (for 1D data)
    if X.shape[1] == 1:
        print("   - Staged predictions")
        gb_viz.plot_staged_predictions_1d(
            X_train, y_train,
            stages=[1, 5, 10, 20, 50]
        )

    # Plot feature importance
    if X.shape[1] == 1:
        feature_names = ['Feature']
    else:
        feature_names = [f'Feature {i+1}' for i in range(X.shape[1])]

    print("   - Feature importance")
    gb_viz.plot_feature_importance(feature_names=feature_names)

    # Plot residuals
    print("   - Residuals plot")
    perf_viz.plot_residuals()

    # Plot learning curve
    print("   - Learning curve")
    perf_viz.plot_learning_curve()

    # Plot predictions vs actual
    print("   - Predictions vs Actual")
    perf_viz.plot_predictions_vs_actual()

    # Multi-dimensional example
    print("\n5. Running multi-dimensional example...")
    X_multi, y_multi = generate_data(n_samples=400, n_features=5)
    X_train_m, X_test_m, y_train_m, y_test_m = train_test_split(
        X_multi, y_multi, test_size=0.25, random_state=42
    )

    model_multi = GradientBoostingRegressor(
        n_estimators=100,
        learning_rate=0.05,
        max_depth=4,
        min_samples_split=5,
        subsample=0.8
    )
    model_multi.fit(X_train_m, y_train_m)

    y_pred_multi = model_multi.predict(X_test_m)
    test_mse_multi = mean_squared_error(y_test_m, y_pred_multi)
    test_r2_multi = r2_score(y_test_m, y_pred_multi)

    print(f"   Multi-dimensional (5 features):")
    print(f"   Test MSE: {test_mse_multi:.4f}, R²: {test_r2_multi:.4f}")

    # Feature importance for multi-dimensional
    gb_viz_multi = GradientBoostingVisualizer(model_multi)
    feature_names_multi = [f'Feature {i+1}' for i in range(5)]
    print("   - Feature importance (multi-dimensional)")
    gb_viz_multi.plot_feature_importance(feature_names=feature_names_multi)

    print("\n" + "=" * 60)
    print("Example complete!")
    print(f"All visualizations have been saved to: {session_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()

