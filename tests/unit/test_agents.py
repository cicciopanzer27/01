"""
Unit tests for the agent classes.
"""

import unittest
import numpy as np
from src.mia_simbolic.core.agents import SymbolicGenerator, Orchestrator, ValidationAgent
from src.mia_simbolic.config import Config


class TestSymbolicGenerator(unittest.TestCase):
    """Test cases for the SymbolicGenerator class."""

    def setUp(self):
        """Set up test fixtures."""
        self.config = Config()
        self.generator = SymbolicGenerator(self.config)

    def test_initialization(self):
        """Test generator initialization."""
        self.assertEqual(self.generator.learning_rate, self.config.generator_learning_rate)
        self.assertEqual(self.generator.momentum, 0.9)
        self.assertIsNone(self.generator.last_step)
        self.assertIsNone(self.generator.last_x)
        self.assertIsNone(self.generator.last_gradient)

    def test_update_params(self):
        """Test parameter update."""
        original_lr = self.generator.learning_rate
        original_momentum = self.generator.momentum
        
        new_params = {
            'learning_rate': 0.05,
            'momentum': 0.8
        }
        
        self.generator.update_params(new_params)
        
        self.assertEqual(self.generator.learning_rate, 0.05)
        self.assertEqual(self.generator.momentum, 0.8)
        self.assertNotEqual(self.generator.learning_rate, original_lr)
        self.assertNotEqual(self.generator.momentum, original_momentum)

    def test_estimate_gradient(self):
        """Test gradient estimation."""
        def sphere(x):
            return np.sum(x**2)
        
        x = np.array([1.0, 2.0])
        gradient = self.generator.estimate_gradient(sphere, x)
        
        self.assertEqual(gradient.shape, x.shape)
        self.assertAlmostEqual(gradient[0], 2.0)  # d/dx (x^2) = 2x
        self.assertAlmostEqual(gradient[1], 4.0)  # d/dx (x^2) = 2x
        
        # Check that last_x and last_gradient are updated
        np.testing.assert_array_equal(self.generator.last_x, x)
        np.testing.assert_array_equal(self.generator.last_gradient, gradient)

    def test_generate_step(self):
        """Test step generation."""
        def sphere(x):
            return np.sum(x**2)
        
        x = np.array([1.0, 2.0])
        gradient = np.array([2.0, 4.0])
        
        step = self.generator.generate_step(sphere, x, gradient)
        
        self.assertEqual(step.shape, x.shape)
        self.assertAlmostEqual(step[0], -self.generator.learning_rate * 2.0)
        self.assertAlmostEqual(step[1], -self.generator.learning_rate * 4.0)
        
        # Test with momentum
        self.generator.last_step = step.copy()
        new_step = self.generator.generate_step(sphere, x, gradient)
        
        # New step should include momentum component
        expected_step = -self.generator.learning_rate * gradient + self.generator.momentum * step
        np.testing.assert_array_almost_equal(new_step, expected_step)


class TestOrchestrator(unittest.TestCase):
    """Test cases for the Orchestrator class."""

    def setUp(self):
        """Set up test fixtures."""
        self.config = Config()
        self.orchestrator = Orchestrator(self.config)

    def test_initialization(self):
        """Test orchestrator initialization."""
        self.assertEqual(self.orchestrator.update_frequency, self.config.orchestrator_update_frequency)
        self.assertEqual(len(self.orchestrator.step_history), 0)
        self.assertEqual(len(self.orchestrator.gradient_history), 0)
        self.assertEqual(len(self.orchestrator.function_value_history), 0)

    def test_update_params(self):
        """Test parameter update."""
        original_freq = self.orchestrator.update_frequency
        
        new_params = {
            'update_frequency': 10
        }
        
        self.orchestrator.update_params(new_params)
        
        self.assertEqual(self.orchestrator.update_frequency, 10)
        self.assertNotEqual(self.orchestrator.update_frequency, original_freq)

    def test_coordinate_step(self):
        """Test step coordination."""
        step = np.array([0.1, 0.2])
        x = np.array([1.0, 1.0])
        gradient = np.array([0.5, 0.5])
        
        coordinated_step = self.orchestrator.coordinate_step(step, x, gradient)
        
        self.assertEqual(coordinated_step.shape, step.shape)
        np.testing.assert_array_equal(coordinated_step, step)  # No change for small step
        
        # Test with large step (should be clipped)
        large_step = np.array([2.0, 2.0])
        coordinated_step = self.orchestrator.coordinate_step(large_step, x, gradient)
        
        self.assertLess(np.linalg.norm(coordinated_step), np.linalg.norm(large_step))
        
        # Test with bounds
        step = np.array([0.5, 0.5])
        bounds = [(-1.0, 1.0), (-1.0, 1.0)]
        x = np.array([0.8, 0.8])
        
        coordinated_step = self.orchestrator.coordinate_step(step, x, gradient, bounds)
        
        # Step should be adjusted to respect bounds
        self.assertLessEqual(x[0] + coordinated_step[0], bounds[0][1])
        self.assertLessEqual(x[1] + coordinated_step[1], bounds[1][1])


class TestValidationAgent(unittest.TestCase):
    """Test cases for the ValidationAgent class."""

    def setUp(self):
        """Set up test fixtures."""
        self.config = Config()
        self.validator = ValidationAgent(self.config)

    def test_initialization(self):
        """Test validator initialization."""
        self.assertEqual(self.validator.validation_threshold, self.config.validation_threshold)
        self.assertEqual(len(self.validator.gradient_history), 0)
        self.assertEqual(len(self.validator.step_history), 0)
        self.assertEqual(len(self.validator.function_value_history), 0)

    def test_update_params(self):
        """Test parameter update."""
        original_threshold = self.validator.validation_threshold
        
        new_params = {
            'validation_threshold': 0.9
        }
        
        self.validator.update_params(new_params)
        
        self.assertEqual(self.validator.validation_threshold, 0.9)
        self.assertNotEqual(self.validator.validation_threshold, original_threshold)

    def test_check_convergence(self):
        """Test convergence checking."""
        x = np.array([0.001, 0.001])
        function_value = 0.000002
        gradient = np.array([0.001, 0.001])
        iteration = 10
        patience_counter = 0
        
        # Should not converge with small gradient but above tolerance
        self.config.convergence_tolerance = 1e-6
        self.validator = ValidationAgent(self.config)
        converged = self.validator.check_convergence(x, function_value, gradient, iteration, patience_counter)
        self.assertFalse(converged)
        
        # Should converge with gradient below tolerance
        gradient = np.array([1e-7, 1e-7])
        converged = self.validator.check_convergence(x, function_value, gradient, iteration, patience_counter)
        self.assertTrue(converged)
        
        # Should converge with patience counter >= patience
        gradient = np.array([0.1, 0.1])
        patience_counter = self.config.patience
        converged = self.validator.check_convergence(x, function_value, gradient, iteration, patience_counter)
        self.assertTrue(converged)


if __name__ == '__main__':
    unittest.main()