"""
Main visualizer for Gradient Boosting algorithm
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from IPython.display import HTML
from visualization.output_handler import OutputHandler


class GradientBoostingVisualizer:
    """
    Comprehensive visualizer for the Gradient Boosting algorithm.
    """

    def __init__(self, model):
        """
        Initialize the visualizer.

        Parameters:
        -----------
        model : GradientBoostingRegressor or GradientBoostingClassifier
            Trained gradient boosting model
        """
        self.model = model

    def plot_training_loss(self, figsize=(10, 6)):
        """
        Plot the training loss over boosting iterations.

        Parameters:
        -----------
        figsize : tuple
            Figure size (width, height)
        """
        plt.figure(figsize=figsize)
        plt.plot(range(1, len(self.model.training_loss) + 1),
                self.model.training_loss, 'b-', linewidth=2)
        plt.xlabel('Number of Estimators', fontsize=12)
        plt.ylabel('Training Loss', fontsize=12)
        plt.title('Gradient Boosting Training Loss', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        # Save to images folder
        output_path = OutputHandler.get_output_path('training_loss.png')
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

    def plot_staged_predictions_1d(self, X, y, stages=None, figsize=(15, 10)):
        """
        Visualize predictions at different boosting stages for 1D data.

        Parameters:
        -----------
        X : array-like, shape (n_samples, 1)
            Input data (must be 1-dimensional)
        y : array-like, shape (n_samples,)
            Target values
        stages : list or None
            Specific stages to visualize (if None, shows evenly spaced stages)
        figsize : tuple
            Figure size (width, height)
        """
        X = np.array(X)
        y = np.array(y)

        if X.shape[1] != 1:
            raise ValueError("This visualization only works with 1D input data")

        # Determine which stages to show
        if stages is None:
            n_stages = min(9, self.model.n_estimators + 1)
            stages = np.linspace(0, self.model.n_estimators, n_stages, dtype=int)

        # Create subplot grid
        n_plots = len(stages)
        n_cols = 3
        n_rows = (n_plots + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        axes = axes.flatten() if n_plots > 1 else [axes]

        # Sort X for smooth line plotting
        sort_idx = np.argsort(X.flatten())
        X_sorted = X[sort_idx]

        # Get predictions at each stage
        predictions = list(self.model.staged_predict(X))

        for idx, stage in enumerate(stages):
            ax = axes[idx]

            # Plot actual data points
            ax.scatter(X, y, alpha=0.5, label='Actual', color='blue', s=30)

            # Plot predictions
            y_pred = predictions[stage]
            ax.plot(X_sorted, y_pred[sort_idx], 'r-', linewidth=2, label='Prediction')

            ax.set_xlabel('X', fontsize=10)
            ax.set_ylabel('y', fontsize=10)
            ax.set_title(f'Stage {stage}: {"Initial" if stage == 0 else f"{stage} Trees"}',
                        fontsize=11, fontweight='bold')
            ax.legend()
            ax.grid(True, alpha=0.3)

        # Hide extra subplots
        for idx in range(n_plots, len(axes)):
            axes[idx].axis('off')

        plt.tight_layout()

        # Save to images folder
        output_path = OutputHandler.get_output_path('staged_predictions_1d.png')
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

    def plot_staged_predictions_2d(self, X, y, stages=None, feature_names=None, figsize=(15, 10)):
        """
        Visualize predictions at different boosting stages for 2D data.

        Parameters:
        -----------
        X : array-like, shape (n_samples, 2)
            Input data (must be 2-dimensional)
        y : array-like, shape (n_samples,)
            Target values
        stages : list or None
            Specific stages to visualize
        feature_names : list or None
            Names of the two features
        figsize : tuple
            Figure size (width, height)
        """
        X = np.array(X)
        y = np.array(y)

        if X.shape[1] != 2:
            raise ValueError("This visualization only works with 2D input data")

        if feature_names is None:
            feature_names = ['Feature 1', 'Feature 2']

        # Determine which stages to show
        if stages is None:
            n_stages = min(6, self.model.n_estimators + 1)
            stages = np.linspace(0, self.model.n_estimators, n_stages, dtype=int)

        # Create mesh grid for decision boundary
        h = 0.02  # step size in mesh
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                            np.arange(y_min, y_max, h))

        # Create subplot grid
        n_plots = len(stages)
        n_cols = 3
        n_rows = (n_plots + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        axes = axes.flatten() if n_plots > 1 else [axes]

        # Get predictions at each stage for mesh
        mesh_points = np.c_[xx.ravel(), yy.ravel()]

        # Check if this is a classifier or regressor
        if hasattr(self.model, 'staged_predict_proba'):
            # For classifiers, use staged_predict_proba and convert to class labels
            predictions_proba = list(self.model.staged_predict_proba(mesh_points))
            predictions = [(proba > 0.5).astype(int) for proba in predictions_proba]
        else:
            # For regressors, use staged_predict
            predictions = list(self.model.staged_predict(mesh_points))

        for idx, stage in enumerate(stages):
            ax = axes[idx]

            # Get predictions for this stage
            Z = predictions[stage].reshape(xx.shape)

            # Plot decision boundary
            contour = ax.contourf(xx, yy, Z, alpha=0.4, cmap='RdYlBu')

            # Plot training points
            scatter = ax.scatter(X[:, 0], X[:, 1], c=y, s=30,
                               edgecolor='black', linewidth=0.5, cmap='RdYlBu')

            ax.set_xlabel(feature_names[0], fontsize=10)
            ax.set_ylabel(feature_names[1], fontsize=10)
            ax.set_title(f'Stage {stage}: {"Initial" if stage == 0 else f"{stage} Trees"}',
                        fontsize=11, fontweight='bold')

            plt.colorbar(scatter, ax=ax)

        # Hide extra subplots
        for idx in range(n_plots, len(axes)):
            axes[idx].axis('off')

        plt.tight_layout()

        # Save to images folder
        output_path = OutputHandler.get_output_path('staged_predictions_2d.png')
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

    def animate_boosting_1d(self, X, y, interval=500, figsize=(10, 6)):
        """
        Create an animation showing how predictions improve with each boosting stage.

        Parameters:
        -----------
        X : array-like, shape (n_samples, 1)
            Input data (must be 1-dimensional)
        y : array-like, shape (n_samples,)
            Target values
        interval : int
            Delay between frames in milliseconds
        figsize : tuple
            Figure size (width, height)

        Returns:
        --------
        FuncAnimation : Animation object
        """
        X = np.array(X)
        y = np.array(y)

        if X.shape[1] != 1:
            raise ValueError("This animation only works with 1D input data")

        # Sort X for smooth line plotting
        sort_idx = np.argsort(X.flatten())
        X_sorted = X[sort_idx]

        # Get all predictions
        predictions = list(self.model.staged_predict(X))

        # Create figure and axis
        fig, ax = plt.subplots(figsize=figsize)

        # Initialize plot elements
        ax.scatter(X, y, alpha=0.5, label='Actual', color='blue', s=30)
        line, = ax.plot([], [], 'r-', linewidth=2, label='Prediction')
        title = ax.set_title('', fontsize=14, fontweight='bold')

        ax.set_xlabel('X', fontsize=12)
        ax.set_ylabel('y', fontsize=12)
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Set axis limits
        ax.set_xlim(X.min() - 0.5, X.max() + 0.5)
        y_min = min(y.min(), min(pred.min() for pred in predictions))
        y_max = max(y.max(), max(pred.max() for pred in predictions))
        ax.set_ylim(y_min - 0.5, y_max + 0.5)

        def animate(frame):
            y_pred = predictions[frame]
            line.set_data(X_sorted, y_pred[sort_idx])
            title.set_text(f'Boosting Stage {frame}: {"Initial Estimate" if frame == 0 else f"{frame} Trees"}\n'
                          f'Training Loss: {self.model.training_loss[frame-1]:.4f}' if frame > 0 else 'Initial Estimate')
            return line, title

        anim = FuncAnimation(fig, animate, frames=len(predictions),
                           interval=interval, blit=True, repeat=True)

        plt.close()  # Prevent duplicate display
        return anim

    def plot_residuals_evolution(self, X, y, stages=None, figsize=(15, 10)):
        """
        Visualize how residuals change across boosting stages.

        Parameters:
        -----------
        X : array-like
            Input data
        y : array-like
            Target values
        stages : list or None
            Specific stages to visualize
        figsize : tuple
            Figure size (width, height)
        """
        X = np.array(X)
        y = np.array(y)

        # Determine which stages to show
        if stages is None:
            n_stages = min(9, self.model.n_estimators + 1)
            stages = np.linspace(0, self.model.n_estimators, n_stages, dtype=int)

        # Create subplot grid
        n_plots = len(stages)
        n_cols = 3
        n_rows = (n_plots + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        axes = axes.flatten() if n_plots > 1 else [axes]

        # Get predictions at each stage
        predictions = list(self.model.staged_predict(X))

        for idx, stage in enumerate(stages):
            ax = axes[idx]

            # Calculate residuals
            residuals = y - predictions[stage]

            # Plot residuals histogram
            ax.hist(residuals, bins=30, alpha=0.7, edgecolor='black')
            ax.axvline(0, color='red', linestyle='--', linewidth=2)

            ax.set_xlabel('Residual', fontsize=10)
            ax.set_ylabel('Frequency', fontsize=10)
            ax.set_title(f'Stage {stage}: {"Initial" if stage == 0 else f"{stage} Trees"}\n'
                        f'Std: {np.std(residuals):.3f}',
                        fontsize=11, fontweight='bold')
            ax.grid(True, alpha=0.3, axis='y')

        # Hide extra subplots
        for idx in range(n_plots, len(axes)):
            axes[idx].axis('off')

        plt.tight_layout()

        # Save to images folder
        output_path = OutputHandler.get_output_path('residual_evolution.png')
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

    def plot_feature_importance(self, feature_names=None, figsize=(10, 6)):
        """
        Plot feature importance based on how often features are used in splits.

        Parameters:
        -----------
        feature_names : list or None
            Names of features
        figsize : tuple
            Figure size (width, height)
        """
        # Count feature usage across all trees
        feature_counts = {}

        for tree in self.model.trees:
            self._count_features(tree.root, feature_counts)

        if not feature_counts:
            print("No feature importance to display (no splits made)")
            return

        # Sort features by importance
        sorted_features = sorted(feature_counts.items(), key=lambda x: x[1], reverse=True)
        features, counts = zip(*sorted_features)

        # Create feature names if not provided
        if feature_names is None:
            feature_names = [f'Feature {i}' for i in range(max(features) + 1)]

        # Plot
        plt.figure(figsize=figsize)
        plt.barh([feature_names[f] for f in features], counts, color='steelblue')
        plt.xlabel('Number of Splits', fontsize=12)
        plt.ylabel('Feature', fontsize=12)
        plt.title('Feature Importance', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3, axis='x')
        plt.tight_layout()

        # Save to images folder
        output_path = OutputHandler.get_output_path('feature_importance.png')
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

    def _count_features(self, node, feature_counts):
        """
        Recursively count feature usage in tree nodes.

        Parameters:
        -----------
        node : TreeNode
            Current tree node
        feature_counts : dict
            Dictionary to store feature counts
        """
        if node.value is not None:
            return

        feature_counts[node.feature] = feature_counts.get(node.feature, 0) + 1
        self._count_features(node.left, feature_counts)
        self._count_features(node.right, feature_counts)

