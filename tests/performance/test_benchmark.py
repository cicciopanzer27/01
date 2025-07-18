"""
Performance tests for the M.I.A.-simbolic optimizer.
"""

import unittest
import time
import numpy as np
from src.mia_simbolic.core.optimizer import MIAOptimizer
from src.mia_simbolic.utils.benchmarks import BenchmarkSuite, SphereFunction, RosenbrockFunction
from src.mia_simbolic.config import Config


class TestBenchmarkPerformance(unittest.TestCase):
    """Test cases for benchmark performance."""

    def setUp(self):
        """Set up test fixtures."""
        self.config = Config()
        self.config.max_iterations = 100
        self.config.convergence_tolerance = 1e-5
        self.config.auto_tune = False
        self.optimizer = MIAOptimizer(config=self.config)

    def test_sphere_function_performance(self):
        """Test performance on sphere function."""
        dimensions = [10, 50, 100]
        
        for dim in dimensions:
            problem = SphereFunction(dimension=dim)
            initial_point = np.ones(dim)
            
            start_time = time.time()
            result = self.optimizer.optimize(problem.objective, initial_point)
            end_time = time.time()
            
            # Check results
            self.assertTrue(result.success)
            self.assertLess(result.fun, 1e-4)
            
            # Log performance
            print(f"Sphere function (dim={dim}): {end_time - start_time:.4f} seconds, {result.nit} iterations")

    def test_rosenbrock_function_performance(self):
        """Test performance on Rosenbrock function."""
        dimensions = [10, 50, 100]
        
        for dim in dimensions:
            problem = RosenbrockFunction(dimension=dim)
            initial_point = np.zeros(dim)
            
            start_time = time.time()
            result = self.optimizer.optimize(problem.objective, initial_point)
            end_time = time.time()
            
            # Check results
            self.assertTrue(result.success)
            
            # Log performance
            print(f"Rosenbrock function (dim={dim}): {end_time - start_time:.4f} seconds, {result.nit} iterations")

    def test_benchmark_suite(self):
        """Test the benchmark suite."""
        # Create benchmark suite with limited runs for testing
        suite = BenchmarkSuite(n_runs=1)
        
        # Run benchmarks on a subset of problems and dimensions
        results = suite.run(
            problem_class="sphere",
            dimensions=[10],
            optimizers=["mia_simbolic"]
        )
        
        # Check results
        self.assertGreater(len(results.results), 0)
        
        # Get summary
        summary = results.summary()
        self.assertGreater(len(summary), 0)
        
        # Compare optimizers (only one in this case)
        comparison = suite.compare(results)
        self.assertGreater(len(comparison), 0)


class TestScalabilityPerformance(unittest.TestCase):
    """Test cases for scalability performance."""

    def test_scalability_with_dimension(self):
        """Test how performance scales with problem dimension."""
        dimensions = [10, 50, 100, 200, 500]
        times = []
        iterations = []
        
        for dim in dimensions:
            config = Config()
            config.max_iterations = 100
            config.convergence_tolerance = 1e-5
            config.auto_tune = False
            optimizer = MIAOptimizer(config=config)
            
            problem = SphereFunction(dimension=dim)
            initial_point = np.ones(dim)
            
            start_time = time.time()
            result = optimizer.optimize(problem.objective, initial_point)
            end_time = time.time()
            
            times.append(end_time - start_time)
            iterations.append(result.nit)
            
            print(f"Dimension {dim}: {times[-1]:.4f} seconds, {iterations[-1]} iterations")
        
        # Check that time increases with dimension
        for i in range(1, len(dimensions)):
            self.assertGreaterEqual(times[i], times[i-1] * 0.5)  # Allow some fluctuation
        
        # Log scaling factor
        for i in range(1, len(dimensions)):
            scaling_factor = times[i] / times[i-1]
            dim_ratio = dimensions[i] / dimensions[i-1]
            print(f"Scaling factor from dim {dimensions[i-1]} to {dimensions[i]}: {scaling_factor:.2f}x (dimension ratio: {dim_ratio:.2f}x)")


if __name__ == '__main__':
    unittest.main()