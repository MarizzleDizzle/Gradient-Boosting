"""
Performance visualization tools for gradient boosting models
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, confusion_matrix, roc_curve, auc
from visualization.output_handler import OutputHandler


class PerformanceVisualizer:
    """
    Visualizer for model performance metrics and diagnostics.
    """

    def __init__(self, model, X_train, y_train, X_test=None, y_test=None):
        """
        Initialize the performance visualizer.

        Parameters:
        -----------
        model : GradientBoostingRegressor or GradientBoostingClassifier
            Trained gradient boosting model
        X_train : array-like
            Training features
        y_train : array-like
            Training targets
        X_test : array-like or None
            Test features
        y_test : array-like or None
            Test targets
        """
        self.model = model
        self.X_train = np.array(X_train)
        self.y_train = np.array(y_train)
        self.X_test = np.array(X_test) if X_test is not None else None
        self.y_test = np.array(y_test) if y_test is not None else None
        self.is_classifier = hasattr(model, 'predict_proba')

    def plot_predictions_vs_actual(self, figsize=(12, 5)):
        """
        Plot predicted vs actual values (for regression).

        Parameters:
        -----------
        figsize : tuple
            Figure size (width, height)
        """
        if self.is_classifier:
            print("This plot is only available for regression models.")
            return

        n_plots = 2 if self.X_test is not None else 1
        fig, axes = plt.subplots(1, n_plots, figsize=figsize)

        if n_plots == 1:
            axes = [axes]

        # Training data
        y_pred_train = self.model.predict(self.X_train)
        axes[0].scatter(self.y_train, y_pred_train, alpha=0.5, s=30)
        axes[0].plot([self.y_train.min(), self.y_train.max()],
                    [self.y_train.min(), self.y_train.max()],
                    'r--', linewidth=2)

        mse_train = mean_squared_error(self.y_train, y_pred_train)
        r2_train = r2_score(self.y_train, y_pred_train)

        axes[0].set_xlabel('Actual Values', fontsize=12)
        axes[0].set_ylabel('Predicted Values', fontsize=12)
        axes[0].set_title(f'Training Set\nMSE: {mse_train:.4f}, R²: {r2_train:.4f}',
                         fontsize=12, fontweight='bold')
        axes[0].grid(True, alpha=0.3)

        # Test data
        if self.X_test is not None:
            y_pred_test = self.model.predict(self.X_test)
            axes[1].scatter(self.y_test, y_pred_test, alpha=0.5, s=30, color='green')
            axes[1].plot([self.y_test.min(), self.y_test.max()],
                        [self.y_test.min(), self.y_test.max()],
                        'r--', linewidth=2)

            mse_test = mean_squared_error(self.y_test, y_pred_test)
            r2_test = r2_score(self.y_test, y_pred_test)

            axes[1].set_xlabel('Actual Values', fontsize=12)
            axes[1].set_ylabel('Predicted Values', fontsize=12)
            axes[1].set_title(f'Test Set\nMSE: {mse_test:.4f}, R²: {r2_test:.4f}',
                             fontsize=12, fontweight='bold')
            axes[1].grid(True, alpha=0.3)

        plt.tight_layout()

        # Save to images folder
        output_path = OutputHandler.get_output_path('predictions_vs_actual.png')
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

    def plot_residuals(self, figsize=(12, 5)):
        """
        Plot residuals analysis (for regression).

        Parameters:
        -----------
        figsize : tuple
            Figure size (width, height)
        """
        if self.is_classifier:
            print("This plot is only available for regression models.")
            return

        n_plots = 2 if self.X_test is not None else 1
        fig, axes = plt.subplots(1, n_plots, figsize=figsize)

        if n_plots == 1:
            axes = [axes]

        # Training residuals
        y_pred_train = self.model.predict(self.X_train)
        residuals_train = self.y_train - y_pred_train

        axes[0].scatter(y_pred_train, residuals_train, alpha=0.5, s=30)
        axes[0].axhline(y=0, color='r', linestyle='--', linewidth=2)
        axes[0].set_xlabel('Predicted Values', fontsize=12)
        axes[0].set_ylabel('Residuals', fontsize=12)
        axes[0].set_title(f'Training Set Residuals\nStd: {np.std(residuals_train):.4f}',
                         fontsize=12, fontweight='bold')
        axes[0].grid(True, alpha=0.3)

        # Test residuals
        if self.X_test is not None:
            y_pred_test = self.model.predict(self.X_test)
            residuals_test = self.y_test - y_pred_test

            axes[1].scatter(y_pred_test, residuals_test, alpha=0.5, s=30, color='green')
            axes[1].axhline(y=0, color='r', linestyle='--', linewidth=2)
            axes[1].set_xlabel('Predicted Values', fontsize=12)
            axes[1].set_ylabel('Residuals', fontsize=12)
            axes[1].set_title(f'Test Set Residuals\nStd: {np.std(residuals_test):.4f}',
                             fontsize=12, fontweight='bold')
            axes[1].grid(True, alpha=0.3)

        plt.tight_layout()

        # Save to images folder
        output_path = OutputHandler.get_output_path('residuals.png')
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

    def plot_confusion_matrix(self, figsize=(12, 5)):
        """
        Plot confusion matrix (for classification).

        Parameters:
        -----------
        figsize : tuple
            Figure size (width, height)
        """
        if not self.is_classifier:
            print("This plot is only available for classification models.")
            return

        n_plots = 2 if self.X_test is not None else 1
        fig, axes = plt.subplots(1, n_plots, figsize=figsize)

        if n_plots == 1:
            axes = [axes]

        # Training confusion matrix
        y_pred_train = self.model.predict(self.X_train)
        cm_train = confusion_matrix(self.y_train, y_pred_train)

        im1 = axes[0].imshow(cm_train, interpolation='nearest', cmap='Blues')
        axes[0].set_title(f'Training Set\nAccuracy: {accuracy_score(self.y_train, y_pred_train):.4f}',
                         fontsize=12, fontweight='bold')
        axes[0].set_xlabel('Predicted Label', fontsize=12)
        axes[0].set_ylabel('True Label', fontsize=12)

        # Add text annotations
        for i in range(cm_train.shape[0]):
            for j in range(cm_train.shape[1]):
                axes[0].text(j, i, str(cm_train[i, j]),
                           ha='center', va='center', fontsize=14)

        plt.colorbar(im1, ax=axes[0])

        # Test confusion matrix
        if self.X_test is not None:
            y_pred_test = self.model.predict(self.X_test)
            cm_test = confusion_matrix(self.y_test, y_pred_test)

            im2 = axes[1].imshow(cm_test, interpolation='nearest', cmap='Greens')
            axes[1].set_title(f'Test Set\nAccuracy: {accuracy_score(self.y_test, y_pred_test):.4f}',
                             fontsize=12, fontweight='bold')
            axes[1].set_xlabel('Predicted Label', fontsize=12)
            axes[1].set_ylabel('True Label', fontsize=12)

            # Add text annotations
            for i in range(cm_test.shape[0]):
                for j in range(cm_test.shape[1]):
                    axes[1].text(j, i, str(cm_test[i, j]),
                               ha='center', va='center', fontsize=14)

            plt.colorbar(im2, ax=axes[1])

        plt.tight_layout()

        # Save to images folder
        output_path = OutputHandler.get_output_path('confusion_matrix.png')
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

    def plot_roc_curve(self, figsize=(8, 6)):
        """
        Plot ROC curve (for binary classification).

        Parameters:
        -----------
        figsize : tuple
            Figure size (width, height)
        """
        if not self.is_classifier:
            print("This plot is only available for classification models.")
            return

        plt.figure(figsize=figsize)

        # Training ROC
        y_pred_proba_train = self.model.predict_proba(self.X_train)
        fpr_train, tpr_train, _ = roc_curve(self.y_train, y_pred_proba_train)
        roc_auc_train = auc(fpr_train, tpr_train)

        plt.plot(fpr_train, tpr_train, 'b-', linewidth=2,
                label=f'Training (AUC = {roc_auc_train:.4f})')

        # Test ROC
        if self.X_test is not None:
            y_pred_proba_test = self.model.predict_proba(self.X_test)
            fpr_test, tpr_test, _ = roc_curve(self.y_test, y_pred_proba_test)
            roc_auc_test = auc(fpr_test, tpr_test)

            plt.plot(fpr_test, tpr_test, 'g-', linewidth=2,
                    label=f'Test (AUC = {roc_auc_test:.4f})')

        # Random classifier line
        plt.plot([0, 1], [0, 1], 'r--', linewidth=2, label='Random Classifier')

        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title('ROC Curve', fontsize=14, fontweight='bold')
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        # Save to images folder
        output_path = OutputHandler.get_output_path('roc_curve.png')
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

    def plot_learning_curve(self, figsize=(10, 6)):
        """
        Plot learning curve showing training and validation performance.

        Parameters:
        -----------
        figsize : tuple
            Figure size (width, height)
        """
        if self.X_test is None:
            print("Test data is required for learning curve.")
            return

        plt.figure(figsize=figsize)

        # Get staged predictions
        train_scores = []
        test_scores = []

        if self.is_classifier:
            for train_pred in self.model.staged_predict_proba(self.X_train):
                train_scores.append(accuracy_score(self.y_train, (train_pred >= 0.5).astype(int)))

            for test_pred in self.model.staged_predict_proba(self.X_test):
                test_scores.append(accuracy_score(self.y_test, (test_pred >= 0.5).astype(int)))

            ylabel = 'Accuracy'
        else:
            for train_pred in self.model.staged_predict(self.X_train):
                train_scores.append(r2_score(self.y_train, train_pred))

            for test_pred in self.model.staged_predict(self.X_test):
                test_scores.append(r2_score(self.y_test, test_pred))

            ylabel = 'R² Score'

        stages = range(len(train_scores))

        plt.plot(stages, train_scores, 'b-', linewidth=2, label='Training')
        plt.plot(stages, test_scores, 'g-', linewidth=2, label='Test')

        plt.xlabel('Number of Estimators', fontsize=12)
        plt.ylabel(ylabel, fontsize=12)
        plt.title('Learning Curve', fontsize=14, fontweight='bold')
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        # Save to images folder
        output_path = OutputHandler.get_output_path('learning_curve.png')
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()


