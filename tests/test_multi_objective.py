"""
Test module for the multi-objective components of M.I.A.-simbolic.

This module contains tests for the multi-objective optimization functionality.
"""

import unittest
import numpy as np
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.mia_simbolic import Config
from src.mia_simbolic.core.multi_objective import (
    MultiObjectiveProblem, QuadraticProblem, NeuralNetworkProblem
)


class TestMultiObjectiveProblem(unittest.TestCase):
    """Test cases for the MultiObjectiveProblem class."""
    
    def test_quadratic_problem(self):
        """Test the quadratic problem implementation."""
        problem = QuadraticProblem(dimension=10)
        
        # Test objective function
        x = np.ones(10)
        obj_val = problem.objective(x)
        
        # Objective should be a weighted sum of components
        components_sum = (
            problem.alpha * problem.validation_loss(x) +
            problem.beta * problem.computational_cost(x) +
            problem.gamma * problem.regularization(x)
        )
        
        self.assertAlmostEqual(obj_val, components_sum)
        
        # Test gradient computation
        gradient = problem.gradient(x)
        self.assertEqual(gradient.shape, x.shape)
        
        # Test weight normalization
        problem.update_weights(alpha=2.0, beta=1.0, gamma=1.0)
        self.assertAlmostEqual(problem.alpha + problem.beta + problem.gamma, 1.0)
    
    def test_weight_update(self):
        """Test weight update functionality."""
        problem = QuadraticProblem(dimension=5)
        
        # Initial weights
        initial_alpha = problem.alpha
        initial_beta = problem.beta
        initial_gamma = problem.gamma
        
        # Update weights
        problem.update_weights(alpha=0.8, beta=0.1, gamma=0.1)
        
        # Check if weights are updated and normalized
        self.assertNotEqual(problem.alpha, initial_alpha)
        self.assertNotEqual(problem.beta, initial_beta)
        self.assertNotEqual(problem.gamma, initial_gamma)
        self.assertAlmostEqual(problem.alpha + problem.beta + problem.gamma, 1.0)
        
        # Check relative proportions
        self.assertAlmostEqual(problem.alpha, 0.8)
        self.assertAlmostEqual(problem.beta, 0.1)
        self.assertAlmostEqual(problem.gamma, 0.1)
    
    def test_component_functions(self):
        """Test individual component functions."""
        problem = QuadraticProblem(dimension=5)
        x = np.ones(5)
        
        # Test validation loss
        val_loss = problem.validation_loss(x)
        self.assertIsInstance(val_loss, float)
        self.assertGreaterEqual(val_loss, 0.0)
        
        # Test computational cost
        comp_cost = problem.computational_cost(x)
        self.assertIsInstance(comp_cost, float)
        self.assertGreaterEqual(comp_cost, 0.0)
        
        # Test regularization
        reg = problem.regularization(x)
        self.assertIsInstance(reg, float)
        self.assertGreaterEqual(reg, 0.0)
        self.assertAlmostEqual(reg, 0.5 * np.sum(x**2))  # L2 regularization


class TestNeuralNetworkProblem(unittest.TestCase):
    """Test cases for the NeuralNetworkProblem class."""
    
    def setUp(self):
        """Set up test fixtures."""
        try:
            import torch
            import torch.nn as nn
            
            # Create a simple model
            self.model = nn.Sequential(
                nn.Linear(10, 5),
                nn.ReLU(),
                nn.Linear(5, 1)
            )
            
            # Create dummy data loaders
            class DummyDataset:
                def __init__(self, size=10):
                    self.size = size
                    self.data = torch.randn(size, 10)
                    self.targets = torch.randn(size, 1)
                
                def __len__(self):
                    return self.size
                
                def __getitem__(self, idx):
                    return self.data[idx], self.targets[idx]
            
            from torch.utils.data import DataLoader
            
            self.train_loader = DataLoader(DummyDataset(), batch_size=2)
            self.val_loader = DataLoader(DummyDataset(), batch_size=2)
            self.criterion = nn.MSELoss()
            
            self.problem = NeuralNetworkProblem(
                model=self.model,
                train_loader=self.train_loader,
                val_loader=self.val_loader,
                criterion=self.criterion,
                device='cpu'
            )
            
            self.torch_available = True
        except ImportError:
            self.torch_available = False
    
    def test_neural_network_problem(self):
        """Test the neural network problem implementation."""
        if not self.torch_available:
            self.skipTest("PyTorch not available")
        
        # Get initial parameters
        params = self.problem._parameters_to_vector()
        
        # Test objective function
        obj_val = self.problem.objective(params)
        
        # Objective should be a weighted sum of components
        components_sum = (
            self.problem.alpha * self.problem.validation_loss(params) +
            self.problem.beta * self.problem.computational_cost(params) +
            self.problem.gamma * self.problem.regularization(params)
        )
        
        self.assertAlmostEqual(obj_val, components_sum)
        
        # Test parameter conversion
        self.problem._vector_to_parameters(params)
        params2 = self.problem._parameters_to_vector()
        
        # Parameters should be the same after conversion
        np.testing.assert_allclose(params, params2)


if __name__ == '__main__':
    unittest.main()