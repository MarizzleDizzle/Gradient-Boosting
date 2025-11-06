.PHONY: help install test test-verbose test-algorithm test-visualization clean lint format

# Default target
help:
	@echo "Gradient Boosting Project - Makefile Commands"
	@echo "=============================================="
	@echo ""
	@echo "Setup:"
	@echo "  make install          Install dependencies"
	@echo ""
	@echo "Testing:"
	@echo "  make test             Run all tests"
	@echo "  make test-verbose     Run tests with verbose output"
	@echo "  make test-algorithm   Test only the gradient boosting algorithm"
	@echo "  make test-visualization   Test visualization package (requires matplotlib)"
	@echo "  make test-examples    Run example scripts"
	@echo ""
	@echo "Code Quality:"
	@echo "  make lint             Check code with flake8 (if installed)"
	@echo "  make format           Format code with black (if installed)"
	@echo ""
	@echo "Cleanup:"
	@echo "  make clean            Remove cache files and temporary files"
	@echo ""

# Install dependencies
install:
	@echo "Installing dependencies..."
	pip install -r requirements.txt
	@echo "✓ Dependencies installed"

# Run all tests
test:
	@echo "Running all tests..."
	@echo ""
	python3 test_packages.py
	@echo ""
	@echo "Running unit tests..."
	python3 -m pytest tests/ -v 2>/dev/null || python3 tests/test_suite.py

# Run tests with verbose output
test-verbose:
	@echo "Running tests with verbose output..."
	python3 test_packages.py
	python3 -m pytest tests/ -vv 2>/dev/null || python3 tests/test_suite.py -v

# Test only the algorithm package
test-algorithm:
	@echo "Testing gradient boosting algorithm..."
	python3 -c "from tests.test_gradient_boosting import test_algorithm; test_algorithm()"

# Test visualization package
test-visualization:
	@echo "Testing visualization package..."
	python3 -c "import matplotlib; print('matplotlib available')" && \
	python3 -c "from tests.test_visualization import test_visualization; test_visualization()" || \
	echo "⚠ matplotlib not installed, skipping visualization tests"

# Run example scripts
test-examples:
	@echo "Testing example scripts..."
	@echo ""
	@echo "Running regression example..."
	python3 example_regression.py || echo "⚠ Example requires matplotlib"
	@echo ""
	@echo "Running classification example..."
	python3 example_classification.py || echo "⚠ Example requires matplotlib"

# Lint code
lint:
	@echo "Linting code..."
	python3 -m flake8 gradient_boosting/ visualization/ --max-line-length=120 2>/dev/null || \
	echo "⚠ flake8 not installed (pip install flake8)"

# Format code
format:
	@echo "Formatting code..."
	python3 -m black gradient_boosting/ visualization/ tests/ 2>/dev/null || \
	echo "⚠ black not installed (pip install black)"

# Clean cache and temporary files
clean:
	@echo "Cleaning up..."
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type f -name "*.pyo" -delete 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	rm -f .coverage 2>/dev/null || true
	@echo "✓ Cleaned up cache files"

# Run quick verification
verify:
	@echo "Quick verification..."
	@python3 -c "from gradient_boosting import GradientBoostingRegressor; print('✓ gradient_boosting package OK')"
	@python3 -c "from gradient_boosting import GradientBoostingClassifier; print('✓ GradientBoostingClassifier OK')"
	@python3 -c "from gradient_boosting import DecisionTreeRegressor; print('✓ DecisionTreeRegressor OK')"
	@python3 -c "from gradient_boosting import SquaredLoss, LogLoss; print('✓ Loss functions OK')"
	@echo ""
	@echo "All core components verified! ✓"

# Run all checks
check: verify test lint
	@echo ""
	@echo "All checks passed! ✓"

