"""
Test module for the framework integrations of M.I.A.-simbolic.

This module contains tests for the PyTorch, TensorFlow, and scikit-learn integrations.
"""

import unittest
import numpy as np
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.mia_simbolic import Config


class TestPyTorchIntegration(unittest.TestCase):
    """Test cases for the PyTorch integration."""
    
    def setUp(self):
        """Set up test fixtures."""
        try:
            import torch
            import torch.nn as nn
            import torch.optim as optim
            from src.mia_simbolic.integrations.pytorch import MIAPyTorchOptimizer
            
            # Create a simple model
            self.model = nn.Sequential(
                nn.Linear(10, 5),
                nn.ReLU(),
                nn.Linear(5, 1)
            )
            
            # Create optimizer
            self.optimizer = MIAPyTorchOptimizer(
                self.model.parameters(),
                lr=0.01,
                auto_tune=False
            )
            
            # Create dummy data
            self.inputs = torch.randn(20, 10)
            self.targets = torch.randn(20, 1)
            
            # Create loss function
            self.criterion = nn.MSELoss()
            
            self.torch_available = True
        except ImportError:
            self.torch_available = False
    
    def test_pytorch_optimizer(self):
        """Test the PyTorch optimizer integration."""
        if not self.torch_available:
            self.skipTest("PyTorch not available")
        
        # Initial loss
        self.optimizer.zero_grad()
        outputs = self.model(self.inputs)
        loss = self.criterion(outputs, self.targets)
        loss.backward()
        initial_loss = loss.item()
        
        # Take a step
        self.optimizer.step()
        
        # Final loss
        self.optimizer.zero_grad()
        outputs = self.model(self.inputs)
        loss = self.criterion(outputs, self.targets)
        final_loss = loss.item()
        
        # Loss should decrease
        self.assertLess(final_loss, initial_loss)


class TestTensorFlowIntegration(unittest.TestCase):
    """Test cases for the TensorFlow integration."""
    
    def setUp(self):
        """Set up test fixtures."""
        try:
            import tensorflow as tf
            from src.mia_simbolic.integrations.tensorflow import MIATensorFlowOptimizer
            
            # Create a simple model
            self.model = tf.keras.Sequential([
                tf.keras.layers.Dense(5, activation='relu', input_shape=(10,)),
                tf.keras.layers.Dense(1)
            ])
            
            # Create optimizer
            self.optimizer = MIATensorFlowOptimizer(
                learning_rate=0.01,
                auto_tune=False
            )
            
            # Compile model
            self.model.compile(
                optimizer=self.optimizer,
                loss='mse'
            )
            
            # Create dummy data
            self.inputs = tf.random.normal((20, 10))
            self.targets = tf.random.normal((20, 1))
            
            self.tf_available = True
        except ImportError:
            self.tf_available = False
    
    def test_tensorflow_optimizer(self):
        """Test the TensorFlow optimizer integration."""
        if not self.tf_available:
            self.skipTest("TensorFlow not available")
        
        # Train for a few steps
        history = self.model.fit(
            self.inputs, self.targets,
            epochs=2,
            verbose=0
        )
        
        # Loss should decrease
        self.assertLess(history.history['loss'][1], history.history['loss'][0])


class TestSklearnIntegration(unittest.TestCase):
    """Test cases for the scikit-learn integration."""
    
    def setUp(self):
        """Set up test fixtures."""
        try:
            import sklearn
            from sklearn.datasets import make_regression
            from src.mia_simbolic.integrations.sklearn import MIALinearRegression
            
            # Create dummy data
            self.X, self.y = make_regression(
                n_samples=100,
                n_features=10,
                noise=0.1,
                random_state=42
            )
            
            # Create model
            self.model = MIALinearRegression(
                learning_rate=0.01,
                auto_tune=False,
                max_iter=100
            )
            
            self.sklearn_available = True
        except ImportError:
            self.sklearn_available = False
    
    def test_sklearn_model(self):
        """Test the scikit-learn model integration."""
        if not self.sklearn_available:
            self.skipTest("scikit-learn not available")
        
        # Fit model
        self.model.fit(self.X, self.y)
        
        # Make predictions
        y_pred = self.model.predict(self.X)
        
        # Compute R^2 score
        from sklearn.metrics import r2_score
        r2 = r2_score(self.y, y_pred)
        
        # R^2 should be reasonable for this simple problem
        self.assertGreater(r2, 0.5)


class TestOptimizer(unittest.TestCase):
    """Test cases for the optimizer integration."""
    
    def test_sklearn_optimizer(self):
        """Test the scikit-learn optimizer integration."""
        try:
            from src.mia_simbolic.integrations.sklearn import MIASklearnOptimizer
            from scipy.optimize import rosen
            
            # Create optimizer
            optimizer = MIASklearnOptimizer(
                learning_rate=0.01,
                auto_tune=False,
                max_iter=1000
            )
            
            # Define objective function
            def objective(x):
                return rosen(x)
            
            # Initial point
            x0 = np.zeros(5)
            
            # Minimize
            result = optimizer.minimize(objective, x0)
            
            # Check if optimization was successful
            self.assertTrue(result['success'])
            self.assertLess(result['fun'], 1.0)
            
        except ImportError:
            self.skipTest("scikit-learn not available")


if __name__ == '__main__':
    unittest.main()