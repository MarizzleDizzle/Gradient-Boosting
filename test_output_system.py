#!/usr/bin/env python3
"""
Quick test to verify the visualization output system works correctly
"""
import numpy as np
from gradient_boosting import GradientBoostingRegressor
from visualization import GradientBoostingVisualizer, OutputHandler

def test_output_system():
    """Test that images are saved to timestamped folders"""

    print("=" * 60)
    print("Testing Visualization Output System")
    print("=" * 60)

    # Initialize session
    session_dir = OutputHandler.initialize_session()
    print(f"\n✓ Session initialized: {session_dir}")

    # Create some dummy data
    np.random.seed(42)
    X = np.random.randn(50, 2)
    y = X[:, 0] + X[:, 1] + np.random.randn(50) * 0.1

    # Train a simple model
    model = GradientBoostingRegressor(n_estimators=10)
    model.fit(X, y)
    print("✓ Model trained")

    # Create visualizations
    viz = GradientBoostingVisualizer(model)

    print("\nCreating visualizations...")
    viz.plot_training_loss()
    print("  ✓ Training loss plot")

    viz.plot_feature_importance(feature_names=['Feature 1', 'Feature 2'])
    print("  ✓ Feature importance plot")

    # Verify files exist
    import os
    expected_files = ['training_loss.png', 'feature_importance.png']

    print("\nVerifying files...")
    all_exist = True
    for filename in expected_files:
        filepath = os.path.join(session_dir, filename)
        if os.path.exists(filepath):
            size = os.path.getsize(filepath)
            print(f"  ✓ {filename} ({size:,} bytes)")
        else:
            print(f"  ✗ {filename} NOT FOUND")
            all_exist = False

    print("\n" + "=" * 60)
    if all_exist:
        print("✅ TEST PASSED: All images saved correctly!")
    else:
        print("❌ TEST FAILED: Some images missing!")
    print(f"📁 Images saved to: {session_dir}")
    print("=" * 60)

    return all_exist

if __name__ == "__main__":
    success = test_output_system()
    exit(0 if success else 1)

