"""
Integration tests for end-to-end optimization.
"""

import unittest
import numpy as np
from src.mia_simbolic.core.optimizer import MIAOptimizer
from src.mia_simbolic.core.multi_objective import QuadraticProblem
from src.mia_simbolic.utils.monitoring import OptimizationMonitor
from src.mia_simbolic.config import Config


class TestEndToEndOptimization(unittest.TestCase):
    """Test cases for end-to-end optimization."""

    def setUp(self):
        """Set up test fixtures."""
        self.config = Config()
        self.config.max_iterations = 100
        self.config.convergence_tolerance = 1e-5
        self.config.auto_tune = False
        self.optimizer = MIAOptimizer(config=self.config)

    def test_optimize_quadratic_problem(self):
        """Test optimization of a quadratic problem."""
        dimension = 5
        problem = QuadraticProblem(dimension=dimension)
        
        # Initial point
        initial_point = np.ones(dimension)
        
        # Run optimization
        result = self.optimizer.optimize(problem.objective, initial_point)
        
        # Check results
        self.assertTrue(result.success)
        self.assertLess(result.fun, problem.objective(initial_point))
        self.assertLess(np.linalg.norm(problem.gradient(result.x)), 0.5)

    def test_optimize_with_monitoring(self):
        """Test optimization with monitoring."""
        # Create a simple test function
        def sphere(x):
            return np.sum(x**2)
        
        # Create monitor
        monitor = OptimizationMonitor(
            metrics=['loss', 'gradient_norm', 'step_size'],
            update_frequency=10,
            save_plots=False
        )
        
        # Set monitor in optimizer
        self.optimizer.monitor = monitor
        
        # Initial point
        initial_point = np.array([1.0, 1.0, 1.0])
        
        # Run optimization
        result = self.optimizer.optimize(sphere, initial_point)
        
        # Check results
        self.assertTrue(result.success)
        self.assertLess(result.fun, 1e-3)
        
        # Check monitor history
        self.assertGreater(len(monitor.history['iteration']), 0)
        self.assertGreater(len(monitor.history['loss']), 0)
        self.assertGreater(len(monitor.history['gradient_norm']), 0)
        self.assertEqual(len(monitor.history['iteration']), len(monitor.history['loss']))
        
        # Check that loss decreases
        self.assertLess(monitor.history['loss'][-1], monitor.history['loss'][0])

    def test_multi_objective_weights(self):
        """Test optimization with different multi-objective weights."""
        dimension = 3
        problem = QuadraticProblem(dimension=dimension)
        
        # Initial point
        initial_point = np.ones(dimension)
        
        # Run optimization with default weights
        result1 = self.optimizer.optimize(problem.objective, initial_point)
        
        # Change weights
        problem.update_weights(alpha=0.8, beta=0.1, gamma=0.1)
        
        # Run optimization with new weights
        result2 = self.optimizer.optimize(problem.objective, initial_point)
        
        # Results should be different due to different weights
        self.assertNotEqual(result1.fun, result2.fun)
        self.assertFalse(np.array_equal(result1.x, result2.x))


class TestEndToEndWithAutoTuning(unittest.TestCase):
    """Test cases for end-to-end optimization with auto-tuning."""

    def setUp(self):
        """Set up test fixtures."""
        self.config = Config()
        self.config.max_iterations = 50
        self.config.convergence_tolerance = 1e-5
        self.config.auto_tune = True
        self.config.auto_tune_trials = 3  # Reduce for faster tests
        self.config.auto_tune_init_points = 2
        self.optimizer = MIAOptimizer(config=self.config)

    def test_optimize_with_auto_tuning(self):
        """Test optimization with auto-tuning."""
        # Skip if scikit-optimize is not available
        try:
            import skopt
        except ImportError:
            self.skipTest("scikit-optimize not available")
        
        # Create a simple test function
        def rosenbrock(x):
            return sum(100.0 * (x[i+1] - x[i]**2)**2 + (x[i] - 1)**2 for i in range(len(x) - 1))
        
        # Initial point
        initial_point = np.zeros(3)
        
        # Run optimization
        result = self.optimizer.optimize(rosenbrock, initial_point)
        
        # Check results
        self.assertTrue(result.success)
        self.assertLess(result.fun, 1.0)  # Should be close to optimal value of 0
        
        # Check that auto-tuner was used
        self.assertIsNotNone(self.optimizer.auto_tuner)
        self.assertIsNotNone(self.optimizer.auto_tuner.best_params)


if __name__ == '__main__':
    unittest.main()