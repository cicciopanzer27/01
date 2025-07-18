"""
Test module for the M.I.A.-simbolic optimizer.

This module contains tests for the core functionality of the M.I.A.-simbolic optimizer.
"""

import unittest
import numpy as np
import time
import os
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.mia_simbolic import MIAOptimizer, Config
from src.mia_simbolic.core.multi_objective import QuadraticProblem
from src.mia_simbolic.benchmarks.problems import (
    SphereFunction, RosenbrockFunction, RastriginFunction, AckleyFunction
)


class TestMIAOptimizer(unittest.TestCase):
    """Test cases for the MIAOptimizer class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = Config(
            convergence_tolerance=1e-6,
            max_iterations=1000,
            auto_tune=False
        )
        self.optimizer = MIAOptimizer(config=self.config)
    
    def test_sphere_function(self):
        """Test optimization of the sphere function."""
        problem = SphereFunction(dimension=10)
        initial_point = np.ones(10)
        
        result = self.optimizer.optimize(problem.objective, initial_point)
        
        self.assertTrue(result.converged)
        self.assertLess(result.fun, 1e-5)
        self.assertLess(np.linalg.norm(result.x), 1e-2)
    
    def test_rosenbrock_function(self):
        """Test optimization of the Rosenbrock function."""
        problem = RosenbrockFunction(dimension=10)
        initial_point = np.zeros(10)
        
        result = self.optimizer.optimize(problem.objective, initial_point)
        
        self.assertTrue(result.converged)
        self.assertLess(result.fun, 1e-2)
        
        # Check if solution is close to [1, 1, ..., 1]
        expected_solution = np.ones(10)
        self.assertLess(np.linalg.norm(result.x - expected_solution), 1e-1)
    
    def test_quadratic_problem(self):
        """Test optimization of the quadratic problem."""
        problem = QuadraticProblem(dimension=10)
        initial_point = np.ones(10)
        
        result = self.optimizer.optimize(problem.objective, initial_point)
        
        self.assertTrue(result.converged)
        self.assertLess(result.gradient_norm, 1e-4)
    
    def test_with_bounds(self):
        """Test optimization with bounds."""
        problem = SphereFunction(dimension=5)
        initial_point = np.ones(5)
        bounds = [(-1, 1)] * 5
        
        result = self.optimizer.optimize(problem.objective, initial_point, bounds=bounds)
        
        self.assertTrue(result.converged)
        
        # Check if solution respects bounds
        for i in range(5):
            self.assertGreaterEqual(result.x[i], bounds[i][0])
            self.assertLessEqual(result.x[i], bounds[i][1])
    
    def test_auto_tuning(self):
        """Test auto-tuning functionality."""
        # Create optimizer with auto-tuning enabled
        auto_tune_config = Config(
            convergence_tolerance=1e-6,
            max_iterations=100,
            auto_tune=True,
            auto_tune_trials=5,  # Reduced for faster testing
            auto_tune_init_points=2
        )
        auto_tune_optimizer = MIAOptimizer(config=auto_tune_config)
        
        problem = SphereFunction(dimension=5)
        initial_point = np.ones(5)
        
        try:
            result = auto_tune_optimizer.optimize(problem.objective, initial_point)
            
            self.assertTrue(result.converged)
            self.assertLess(result.fun, 1e-4)
        except ImportError:
            # Skip test if scikit-optimize is not available
            self.skipTest("scikit-optimize not available")
    
    def test_performance(self):
        """Test performance of the optimizer."""
        problem = SphereFunction(dimension=100)
        initial_point = np.ones(100)
        
        start_time = time.time()
        result = self.optimizer.optimize(problem.objective, initial_point)
        end_time = time.time()
        
        self.assertTrue(result.converged)
        self.assertLess(end_time - start_time, 5.0)  # Should be fast for sphere function
    
    def test_multi_objective_weights(self):
        """Test multi-objective weight adjustment."""
        # Create a problem with multi-objective components
        problem = QuadraticProblem(dimension=10)
        initial_point = np.ones(10)
        
        # Create optimizer with custom weights
        custom_weights_config = Config(
            alpha=0.8,  # Higher weight on validation loss
            beta=0.1,   # Lower weight on computational cost
            gamma=0.1   # Lower weight on regularization
        )
        custom_optimizer = MIAOptimizer(config=custom_weights_config)
        
        result = custom_optimizer.optimize(problem.objective, initial_point)
        
        self.assertTrue(result.converged)
        self.assertLess(result.fun, 1e-4)


class TestAgents(unittest.TestCase):
    """Test cases for the agent components."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = Config(
            convergence_tolerance=1e-6,
            max_iterations=1000,
            auto_tune=False
        )
        self.optimizer = MIAOptimizer(config=self.config)
    
    def test_symbolic_generator(self):
        """Test the symbolic generator agent."""
        generator = self.optimizer.generator
        
        # Test gradient estimation
        problem = SphereFunction(dimension=5)
        x = np.ones(5)
        
        estimated_gradient = generator.estimate_gradient(problem.objective, x)
        exact_gradient = problem.gradient(x)
        
        # Check if estimated gradient is close to exact gradient
        self.assertLess(np.linalg.norm(estimated_gradient - exact_gradient), 1e-4)
        
        # Test step generation
        step = generator.generate_step(problem.objective, x, estimated_gradient)
        
        # Step should be in the opposite direction of the gradient
        self.assertLess(np.dot(step, estimated_gradient), 0)
    
    def test_orchestrator(self):
        """Test the orchestrator agent."""
        orchestrator = self.optimizer.orchestrator
        
        # Test step coordination
        step = np.array([10.0, -5.0, 8.0, -12.0, 7.0])
        x = np.zeros(5)
        gradient = np.array([1.0, -0.5, 0.8, -1.2, 0.7])
        bounds = [(-5, 5)] * 5
        
        coordinated_step = orchestrator.coordinate_step(step, x, gradient, bounds)
        
        # Check if step is clipped
        self.assertLessEqual(np.linalg.norm(coordinated_step), 1.0)
        
        # Check if bounds are respected
        for i in range(5):
            self.assertGreaterEqual(x[i] + coordinated_step[i], bounds[i][0])
            self.assertLessEqual(x[i] + coordinated_step[i], bounds[i][1])
    
    def test_validator(self):
        """Test the validation agent."""
        validator = self.optimizer.validator
        
        # Test convergence checking
        x = np.zeros(5)
        function_value = 0.0
        gradient = np.array([1e-7, 1e-7, 1e-7, 1e-7, 1e-7])
        iteration = 10
        patience_counter = 0
        
        # Should converge due to small gradient
        self.assertTrue(validator.check_convergence(x, function_value, gradient, iteration, patience_counter))
        
        # Test with larger gradient
        gradient = np.array([0.1, 0.1, 0.1, 0.1, 0.1])
        
        # Should not converge due to large gradient
        self.assertFalse(validator.check_convergence(x, function_value, gradient, iteration, patience_counter))
        
        # Test with patience
        patience_counter = 10
        
        # Should converge due to patience
        self.assertTrue(validator.check_convergence(x, function_value, gradient, iteration, patience_counter))


if __name__ == '__main__':
    unittest.main()