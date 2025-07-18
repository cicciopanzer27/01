"""
Unit tests for the MultiObjectiveProblem class.
"""

import unittest
import numpy as np
from src.mia_simbolic.core.multi_objective import MultiObjectiveProblem, QuadraticProblem
from src.mia_simbolic.config import Config


class TestMultiObjectiveProblem(unittest.TestCase):
    """Test cases for the MultiObjectiveProblem class."""

    def test_quadratic_problem(self):
        """Test the QuadraticProblem implementation."""
        dimension = 5
        problem = QuadraticProblem(dimension=dimension)
        
        # Test initialization
        self.assertEqual(problem.dimension, dimension)
        self.assertAlmostEqual(problem.alpha + problem.beta + problem.gamma, 1.0)
        
        # Test validation loss
        x = np.ones(dimension)
        val_loss = problem.validation_loss(x)
        self.assertIsInstance(val_loss, float)
        
        # Test computational cost
        comp_cost = problem.computational_cost(x)
        self.assertIsInstance(comp_cost, float)
        
        # Test regularization
        reg = problem.regularization(x)
        self.assertIsInstance(reg, float)
        self.assertAlmostEqual(reg, 0.5 * dimension)  # 0.5 * ||x||^2 for x = ones(dimension)
        
        # Test objective function
        obj_val = problem.objective(x)
        self.assertIsInstance(obj_val, float)
        self.assertAlmostEqual(obj_val, problem.alpha * val_loss + problem.beta * comp_cost + problem.gamma * reg)
        
        # Test gradient
        grad = problem.gradient(x)
        self.assertEqual(grad.shape, (dimension,))
        
        # Test weight update
        original_alpha = problem.alpha
        original_beta = problem.beta
        original_gamma = problem.gamma
        
        problem.update_weights(alpha=0.5, beta=0.3, gamma=0.2)
        
        self.assertNotEqual(problem.alpha, original_alpha)
        self.assertNotEqual(problem.beta, original_beta)
        self.assertNotEqual(problem.gamma, original_gamma)
        self.assertAlmostEqual(problem.alpha + problem.beta + problem.gamma, 1.0)


class TestCustomMultiObjectiveProblem(unittest.TestCase):
    """Test cases for a custom MultiObjectiveProblem implementation."""
    
    class SimpleMultiObjectiveProblem(MultiObjectiveProblem):
        """Simple implementation of MultiObjectiveProblem for testing."""
        
        def validation_loss(self, x):
            """Compute validation loss as sum of squares."""
            return np.sum(x**2)
        
        def computational_cost(self, x):
            """Compute computational cost as sum of absolute values."""
            return np.sum(np.abs(x))
        
        def regularization(self, x):
            """Compute regularization as L1 norm."""
            return np.sum(np.abs(x))
    
    def test_custom_problem(self):
        """Test a custom MultiObjectiveProblem implementation."""
        config = Config()
        config.alpha = 0.6
        config.beta = 0.3
        config.gamma = 0.1
        
        problem = self.SimpleMultiObjectiveProblem(config)
        
        # Test with a simple vector
        x = np.array([1.0, 2.0, 3.0])
        
        # Test individual components
        val_loss = problem.validation_loss(x)
        self.assertAlmostEqual(val_loss, 14.0)  # 1^2 + 2^2 + 3^2 = 14
        
        comp_cost = problem.computational_cost(x)
        self.assertAlmostEqual(comp_cost, 6.0)  # |1| + |2| + |3| = 6
        
        reg = problem.regularization(x)
        self.assertAlmostEqual(reg, 6.0)  # |1| + |2| + |3| = 6
        
        # Test combined objective
        obj_val = problem.objective(x)
        expected_obj = 0.6 * 14.0 + 0.3 * 6.0 + 0.1 * 6.0
        self.assertAlmostEqual(obj_val, expected_obj)
        
        # Test gradient
        grad = problem.gradient(x)
        self.assertEqual(grad.shape, (3,))


if __name__ == '__main__':
    unittest.main()