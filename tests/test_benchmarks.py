"""
Test module for the benchmark functionality of M.I.A.-simbolic.

This module contains tests for the benchmark and validation components.
"""

import unittest
import numpy as np
import os
import sys
import tempfile
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.mia_simbolic.benchmarks.problems import (
    SphereFunction, RosenbrockFunction, RastriginFunction, AckleyFunction
)
from src.mia_simbolic.benchmarks.baselines import (
    AdamOptimizer, SGDOptimizer, LBFGSOptimizer, RMSpropOptimizer, AdagradOptimizer
)
from src.mia_simbolic.utils.benchmarks import BenchmarkSuite, BenchmarkResults
from src.mia_simbolic.utils.validation import ValidationProtocol, ValidationResults


class TestBenchmarkProblems(unittest.TestCase):
    """Test cases for the benchmark problems."""
    
    def test_sphere_function(self):
        """Test the sphere function implementation."""
        problem = SphereFunction(dimension=5)
        
        # Test at origin (minimum)
        x = np.zeros(5)
        self.assertEqual(problem.objective(x), 0.0)
        
        # Test at unit vector
        x = np.ones(5)
        self.assertEqual(problem.objective(x), 5.0)
        
        # Test gradient
        gradient = problem.gradient(x)
        np.testing.assert_array_equal(gradient, 2 * x)
    
    def test_rosenbrock_function(self):
        """Test the Rosenbrock function implementation."""
        problem = RosenbrockFunction(dimension=5)
        
        # Test at minimum
        x = np.ones(5)
        self.assertEqual(problem.objective(x), 0.0)
        
        # Test at origin
        x = np.zeros(5)
        self.assertGreater(problem.objective(x), 0.0)
    
    def test_rastrigin_function(self):
        """Test the Rastrigin function implementation."""
        problem = RastriginFunction(dimension=5)
        
        # Test at origin (minimum)
        x = np.zeros(5)
        self.assertEqual(problem.objective(x), 0.0)
        
        # Test at unit vector
        x = np.ones(5)
        self.assertGreater(problem.objective(x), 0.0)
    
    def test_ackley_function(self):
        """Test the Ackley function implementation."""
        problem = AckleyFunction(dimension=5)
        
        # Test at origin (minimum)
        x = np.zeros(5)
        self.assertAlmostEqual(problem.objective(x), 0.0, places=5)
        
        # Test at unit vector
        x = np.ones(5)
        self.assertGreater(problem.objective(x), 0.0)
    
    def test_noise_addition(self):
        """Test noise addition to benchmark problems."""
        problem = SphereFunction(dimension=5, noise_std=0.1)
        
        # Run multiple evaluations to check if noise is added
        x = np.ones(5)
        values = [problem.objective(x) for _ in range(10)]
        
        # Values should be different due to noise
        self.assertGreater(np.std(values), 0.0)


class TestBaselineOptimizers(unittest.TestCase):
    """Test cases for the baseline optimizers."""
    
    def test_adam_optimizer(self):
        """Test the Adam optimizer implementation."""
        optimizer = AdamOptimizer(lr=0.1)
        problem = SphereFunction(dimension=5)
        initial_point = np.ones(5)
        
        result = optimizer.optimize(problem.objective, initial_point)
        
        self.assertTrue(result.success)
        self.assertLess(result.fun, 1e-4)
    
    def test_sgd_optimizer(self):
        """Test the SGD optimizer implementation."""
        optimizer = SGDOptimizer(lr=0.1, momentum=0.9)
        problem = SphereFunction(dimension=5)
        initial_point = np.ones(5)
        
        result = optimizer.optimize(problem.objective, initial_point)
        
        self.assertTrue(result.success)
        self.assertLess(result.fun, 1e-4)
    
    def test_lbfgs_optimizer(self):
        """Test the L-BFGS optimizer implementation."""
        optimizer = LBFGSOptimizer()
        problem = SphereFunction(dimension=5)
        initial_point = np.ones(5)
        
        result = optimizer.optimize(problem.objective, initial_point)
        
        self.assertTrue(result.success)
        self.assertLess(result.fun, 1e-4)


class TestBenchmarkSuite(unittest.TestCase):
    """Test cases for the benchmark suite."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create temporary directory for output
        self.temp_dir = tempfile.TemporaryDirectory()
        self.output_dir = self.temp_dir.name
        
        # Create benchmark suite with minimal settings for testing
        self.suite = BenchmarkSuite(
            n_runs=2,
            output_dir=self.output_dir
        )
    
    def tearDown(self):
        """Tear down test fixtures."""
        self.temp_dir.cleanup()
    
    def test_add_problem(self):
        """Test adding a problem to the benchmark suite."""
        # Add a custom problem
        self.suite.add_problem('custom_sphere', lambda dim: SphereFunction(dimension=dim))
        
        # Check if problem was added
        self.assertIn('custom_sphere', self.suite.problems)
    
    def test_add_optimizer(self):
        """Test adding an optimizer to the benchmark suite."""
        # Add a custom optimizer
        self.suite.add_optimizer('custom_adam', lambda: AdamOptimizer(lr=0.1))
        
        # Check if optimizer was added
        self.assertIn('custom_adam', self.suite.optimizers)
    
    def test_run_benchmark(self):
        """Test running a benchmark."""
        # Run a minimal benchmark
        results = self.suite.run(
            problem_class='sphere',
            dimensions=[2],
            optimizers=['mia_simbolic']
        )
        
        # Check if results were generated
        self.assertIsInstance(results, BenchmarkResults)
        self.assertGreater(len(results.results), 0)
        
        # Check if results were saved
        files = os.listdir(self.output_dir)
        self.assertGreater(len(files), 0)
    
    def test_compare_results(self):
        """Test comparing benchmark results."""
        # Run a benchmark with multiple optimizers
        self.suite.add_optimizer('adam', lambda: AdamOptimizer(lr=0.1))
        
        results = self.suite.run(
            problem_class='sphere',
            dimensions=[2],
            optimizers=['mia_simbolic', 'adam']
        )
        
        # Compare results
        comparison = self.suite.compare(results)
        
        # Check if comparison was generated
        self.assertIsInstance(comparison, dict)
        self.assertGreater(len(comparison), 0)


class TestValidationProtocol(unittest.TestCase):
    """Test cases for the validation protocol."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create temporary directory for output
        self.temp_dir = tempfile.TemporaryDirectory()
        self.output_dir = self.temp_dir.name
        
        # Create validation protocol with minimal settings for testing
        self.protocol = ValidationProtocol(
            n_runs=2,
            output_dir=self.output_dir
        )
    
    def tearDown(self):
        """Tear down test fixtures."""
        self.temp_dir.cleanup()
    
    def test_add_problem(self):
        """Test adding a problem to the validation protocol."""
        from src.mia_simbolic.utils.validation import SphereValidation
        
        # Add a custom problem
        self.protocol.add_problem('custom_sphere', lambda dim: SphereValidation(dimension=dim))
        
        # Check if problem was added
        self.assertIn('custom_sphere', self.protocol.problems)
    
    def test_run_validation(self):
        """Test running a validation."""
        # Run a minimal validation
        results = self.protocol.run(
            problem_class='sphere',
            dimensions=[2]
        )
        
        # Check if results were generated
        self.assertIsInstance(results, ValidationResults)
        self.assertGreater(len(results.results), 0)
        
        # Check if results were saved
        files = os.listdir(self.output_dir)
        self.assertGreater(len(files), 0)


if __name__ == '__main__':
    unittest.main()