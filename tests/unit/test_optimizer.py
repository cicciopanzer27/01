"""
Unit tests for the MIAOptimizer class.
"""

import unittest
import numpy as np
from src.mia_simbolic.core.optimizer import MIAOptimizer
from src.mia_simbolic.config import Config


class TestMIAOptimizer(unittest.TestCase):
    """Test cases for the MIAOptimizer class."""

    def setUp(self):
        """Set up test fixtures."""
        self.config = Config()
        self.config.max_iterations = 100
        self.config.convergence_tolerance = 1e-5
        self.config.auto_tune = False
        self.optimizer = MIAOptimizer(
            max_iterations=100,
            convergence_tolerance=1e-5,
            auto_tune=False
        )

    def test_initialization(self):
        """Test optimizer initialization."""
        self.assertIsNotNone(self.optimizer)
        self.assertIsNotNone(self.optimizer.generator)
        self.assertIsNotNone(self.optimizer.orchestrator)
        self.assertIsNotNone(self.optimizer.validator)
        self.assertIsNone(self.optimizer.auto_tuner)

    def test_optimize_sphere(self):
        """Test optimization of the sphere function."""
        def sphere(x):
            return np.sum(x**2)

        initial_point = np.array([1.0, 1.0])
        result = self.optimizer.optimize(sphere, initial_point)

        self.assertTrue(result.success)
        self.assertLess(result.fun, 1e-3)
        self.assertLess(np.linalg.norm(result.x), 1e-2)

    def test_optimize_rosenbrock(self):
        """Test optimization of the Rosenbrock function."""
        def rosenbrock(x):
            return sum(100.0 * (x[i+1] - x[i]**2)**2 + (x[i] - 1)**2 for i in range(len(x) - 1))

        initial_point = np.array([0.0, 0.0])
        result = self.optimizer.optimize(rosenbrock, initial_point)

        self.assertTrue(result.success)
        self.assertLess(np.linalg.norm(result.x - np.array([1.0, 1.0])), 0.5)

    def test_optimize_with_bounds(self):
        """Test optimization with bounds."""
        def constrained_function(x):
            return np.sum(x**2)

        initial_point = np.array([0.5, 0.5])
        bounds = [(-1.0, 1.0), (-1.0, 1.0)]
        result = self.optimizer.optimize(constrained_function, initial_point, bounds=bounds)

        self.assertTrue(result.success)
        self.assertLess(result.fun, 1e-3)
        
        # Check that bounds are respected
        for i, (lower, upper) in enumerate(bounds):
            self.assertGreaterEqual(result.x[i], lower)
            self.assertLessEqual(result.x[i], upper)

    def test_optimization_result_properties(self):
        """Test properties of the optimization result."""
        def simple_function(x):
            return np.sum(x**2)

        initial_point = np.array([1.0, 1.0])
        result = self.optimizer.optimize(simple_function, initial_point)

        self.assertIsNotNone(result.x)
        self.assertIsNotNone(result.fun)
        self.assertIsNotNone(result.nit)
        self.assertIsNotNone(result.nfev)
        self.assertIsNotNone(result.time)
        self.assertIsNotNone(result.converged)
        self.assertIsNotNone(result.success)
        self.assertIsNotNone(result.message)
        self.assertIsNotNone(result.gradient_norm)
        self.assertIsNotNone(result.efficiency_score)


if __name__ == '__main__':
    unittest.main()