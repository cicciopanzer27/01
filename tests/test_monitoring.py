"""
Test module for the monitoring functionality of M.I.A.-simbolic.

This module contains tests for the monitoring and visualization components.
"""

import unittest
import numpy as np
import os
import sys
import tempfile
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.mia_simbolic.utils.monitoring import OptimizationMonitor


class TestOptimizationMonitor(unittest.TestCase):
    """Test cases for the OptimizationMonitor class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create temporary directory for output
        self.temp_dir = tempfile.TemporaryDirectory()
        self.output_dir = self.temp_dir.name
        
        # Create monitor
        self.monitor = OptimizationMonitor(
            metrics=['loss', 'gradient_norm', 'step_size'],
            update_frequency=1,
            save_plots=True,
            output_dir=self.output_dir
        )
    
    def tearDown(self):
        """Tear down test fixtures."""
        self.temp_dir.cleanup()
    
    def test_initialization(self):
        """Test initialization of the monitor."""
        # Check if metrics are set correctly
        self.assertIn('loss', self.monitor.metrics)
        self.assertIn('gradient_norm', self.monitor.metrics)
        self.assertIn('step_size', self.monitor.metrics)
        
        # Check if history is initialized
        self.assertIn('iteration', self.monitor.history)
        self.assertIn('time', self.monitor.history)
        self.assertIn('loss', self.monitor.history)
        self.assertIn('gradient_norm', self.monitor.history)
        self.assertIn('step_size', self.monitor.history)
        self.assertIn('x', self.monitor.history)
        
        # Check if output directory exists
        self.assertTrue(os.path.exists(self.output_dir))
    
    def test_update(self):
        """Test updating the monitor."""
        # Update monitor with some data
        for i in range(5):
            x = np.ones(3) * i
            loss = 1.0 / (i + 1)
            gradient = np.ones(3) * 0.1 / (i + 1)
            step_size = 0.01 / (i + 1)
            
            self.monitor.update(i, x, loss, gradient, step_size)
        
        # Check if history was updated
        self.assertEqual(len(self.monitor.history['iteration']), 5)
        self.assertEqual(len(self.monitor.history['loss']), 5)
        self.assertEqual(len(self.monitor.history['gradient_norm']), 5)
        self.assertEqual(len(self.monitor.history['step_size']), 5)
        self.assertEqual(len(self.monitor.history['x']), 5)
        
        # Check if values were stored correctly
        self.assertEqual(self.monitor.history['iteration'][0], 0)
        self.assertEqual(self.monitor.history['iteration'][4], 4)
        self.assertEqual(self.monitor.history['loss'][0], 1.0)
        self.assertEqual(self.monitor.history['loss'][4], 0.2)
    
    def test_save_history(self):
        """Test saving history to file."""
        # Update monitor with some data
        for i in range(5):
            x = np.ones(3) * i
            loss = 1.0 / (i + 1)
            gradient = np.ones(3) * 0.1 / (i + 1)
            step_size = 0.01 / (i + 1)
            
            self.monitor.update(i, x, loss, gradient, step_size)
        
        # Save history
        self.monitor.save_history()
        
        # Check if file was created
        files = os.listdir(self.output_dir)
        npz_files = [f for f in files if f.endswith('.npz')]
        self.assertGreater(len(npz_files), 0)
    
    def test_generate_report(self):
        """Test generating a report."""
        # Update monitor with some data
        for i in range(5):
            x = np.ones(3) * i
            loss = 1.0 / (i + 1)
            gradient = np.ones(3) * 0.1 / (i + 1)
            step_size = 0.01 / (i + 1)
            
            self.monitor.update(i, x, loss, gradient, step_size)
        
        # Generate report
        self.monitor.generate_report()
        
        # Check if file was created
        files = os.listdir(self.output_dir)
        html_files = [f for f in files if f.endswith('.html')]
        self.assertGreater(len(html_files), 0)
    
    def test_plot_contour(self):
        """Test plotting contour."""
        # Define a simple 2D function
        def sphere(x):
            return np.sum(x**2)
        
        # Update monitor with some data
        for i in range(5):
            x = np.array([1.0 - 0.2*i, 1.0 - 0.2*i])
            loss = sphere(x)
            gradient = 2 * x
            step_size = 0.2
            
            self.monitor.update(i, x, loss, gradient, step_size)
        
        # Plot contour
        try:
            import matplotlib
            
            self.monitor.plot_contour(
                objective_function=sphere,
                bounds=[(-2, 2), (-2, 2)],
                resolution=10
            )
            
            # Check if file was created
            files = os.listdir(self.output_dir)
            png_files = [f for f in files if f.endswith('.png') and 'contour' in f]
            self.assertGreater(len(png_files), 0)
            
        except ImportError:
            self.skipTest("matplotlib not available")
    
    def test_plot_3d_surface(self):
        """Test plotting 3D surface."""
        # Define a simple 2D function
        def sphere(x):
            return np.sum(x**2)
        
        # Update monitor with some data
        for i in range(5):
            x = np.array([1.0 - 0.2*i, 1.0 - 0.2*i])
            loss = sphere(x)
            gradient = 2 * x
            step_size = 0.2
            
            self.monitor.update(i, x, loss, gradient, step_size)
        
        # Plot 3D surface
        try:
            import matplotlib
            
            self.monitor.plot_3d_surface(
                objective_function=sphere,
                bounds=[(-2, 2), (-2, 2)],
                resolution=10
            )
            
            # Check if file was created
            files = os.listdir(self.output_dir)
            png_files = [f for f in files if f.endswith('.png') and 'surface' in f]
            self.assertGreater(len(png_files), 0)
            
        except ImportError:
            self.skipTest("matplotlib not available")


if __name__ == '__main__':
    unittest.main()