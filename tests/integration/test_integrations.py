"""
Integration tests for framework integrations.
"""

import unittest
import numpy as np
from src.mia_simbolic.config import Config


class TestPyTorchIntegration(unittest.TestCase):
    """Test cases for PyTorch integration."""

    def setUp(self):
        """Set up test fixtures."""
        try:
            import torch
            self.torch_available = True
        except ImportError:
            self.torch_available = False

    def test_pytorch_optimizer(self):
        """Test PyTorch optimizer integration."""
        if not self.torch_available:
            self.skipTest("PyTorch not available")
        
        import torch
        import torch.nn as nn
        from src.mia_simbolic.integrations.pytorch import MIAPyTorchOptimizer
        
        # Create a simple model
        model = nn.Sequential(
            nn.Linear(2, 10),
            nn.ReLU(),
            nn.Linear(10, 1)
        )
        
        # Create optimizer
        optimizer = MIAPyTorchOptimizer(model.parameters(), lr=0.01, auto_tune=False)
        
        # Create dummy data
        x = torch.randn(100, 2)
        y = torch.randn(100, 1)
        
        # Train for a few steps
        for _ in range(5):
            def closure():
                optimizer.zero_grad()
                output = model(x)
                loss = nn.MSELoss()(output, y)
                loss.backward()
                return loss
            
            loss = optimizer.step(closure)
            
            self.assertIsNotNone(loss)
            self.assertGreater(loss.item(), 0)


class TestTensorFlowIntegration(unittest.TestCase):
    """Test cases for TensorFlow integration."""

    def setUp(self):
        """Set up test fixtures."""
        try:
            import tensorflow as tf
            self.tf_available = True
        except ImportError:
            self.tf_available = False

    def test_tensorflow_optimizer(self):
        """Test TensorFlow optimizer integration."""
        if not self.tf_available:
            self.skipTest("TensorFlow not available")
        
        import tensorflow as tf
        from src.mia_simbolic.integrations.tensorflow import MIATensorFlowOptimizer
        
        # Create a simple model
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(10, activation='relu', input_shape=(2,)),
            tf.keras.layers.Dense(1)
        ])
        
        # Create optimizer
        optimizer = MIATensorFlowOptimizer(learning_rate=0.01, auto_tune=False)
        
        # Compile model
        model.compile(optimizer=optimizer, loss='mse')
        
        # Create dummy data
        x = np.random.randn(100, 2)
        y = np.random.randn(100, 1)
        
        # Train for a few steps
        model.fit(x, y, epochs=1, batch_size=32, verbose=0)
        
        # Check that the model has been trained
        self.assertGreater(optimizer.iterations.numpy(), 0)


class TestSklearnIntegration(unittest.TestCase):
    """Test cases for scikit-learn integration."""

    def setUp(self):
        """Set up test fixtures."""
        try:
            import sklearn
            self.sklearn_available = True
        except ImportError:
            self.sklearn_available = False

    def test_sklearn_optimizer(self):
        """Test scikit-learn optimizer integration."""
        if not self.sklearn_available:
            self.skipTest("scikit-learn not available")
        
        from src.mia_simbolic.integrations.sklearn import MIASklearnOptimizer
        
        # Create optimizer
        optimizer = MIASklearnOptimizer(learning_rate=0.01, auto_tune=False, max_iter=50)
        
        # Create a simple function to minimize
        def rosenbrock(x):
            return sum(100.0 * (x[i+1] - x[i]**2)**2 + (x[i] - 1)**2 for i in range(len(x) - 1))
        
        # Initial point
        x0 = np.zeros(3)
        
        # Minimize
        result = optimizer.minimize(rosenbrock, x0)
        
        # Check results
        self.assertIn('x', result)
        self.assertIn('fun', result)
        self.assertIn('nit', result)
        self.assertIn('success', result)
        
        self.assertLess(result['fun'], rosenbrock(x0))

    def test_linear_regression(self):
        """Test linear regression integration."""
        if not self.sklearn_available:
            self.skipTest("scikit-learn not available")
        
        from src.mia_simbolic.integrations.sklearn import MIALinearRegression
        
        # Create model
        model = MIALinearRegression(learning_rate=0.01, auto_tune=False, max_iter=50)
        
        # Create dummy data
        np.random.seed(42)
        X = np.random.randn(100, 5)
        true_coef = np.array([1.5, -0.5, 0.8, -0.2, 1.0])
        y = X @ true_coef + 0.1 * np.random.randn(100)
        
        # Fit model
        model.fit(X, y)
        
        # Check results
        self.assertEqual(model.coef_.shape, true_coef.shape)
        
        # Predictions should be close to true values
        y_pred = model.predict(X)
        mse = np.mean((y - y_pred) ** 2)
        self.assertLess(mse, 0.5)


if __name__ == '__main__':
    unittest.main()