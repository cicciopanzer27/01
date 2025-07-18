"""
Monitoring module for M.I.A.-simbolic.

This module implements the OptimizationMonitor class, which provides
real-time monitoring and visualization of the optimization process.
"""

import logging
import time
import os
from typing import Dict, Any, Optional, List, Tuple, Union, Set
from datetime import datetime

import numpy as np

logger = logging.getLogger(__name__)

class OptimizationMonitor:
    """Monitor for optimization progress.
    
    This class provides real-time monitoring and visualization of the
    optimization process, including metrics tracking and plotting.
    
    Attributes:
        metrics: Set of metrics to track
        update_frequency: Frequency of updates
        save_plots: Whether to save plots
        output_dir: Directory for output files
        history: History of metrics
        start_time: Start time of optimization
    """
    
    def __init__(self, 
                metrics: Optional[List[str]] = None, 
                update_frequency: int = 10, 
                save_plots: bool = True, 
                output_dir: Optional[str] = None):
        """Initialize the optimization monitor.
        
        Args:
            metrics: List of metrics to track
            update_frequency: Frequency of updates
            save_plots: Whether to save plots
            output_dir: Directory for output files
        """
        self.metrics = set(metrics) if metrics is not None else {'convergence', 'loss', 'gradient_norm'}
        self.update_frequency = update_frequency
        self.save_plots = save_plots
        self.output_dir = output_dir or os.path.join(os.getcwd(), 'results', 'monitoring')
        
        # Create output directory if it doesn't exist
        if self.save_plots and not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        
        # Initialize history
        self.history = {
            'iteration': [],
            'time': [],
            'loss': [],
            'gradient_norm': [],
            'step_size': [],
            'x': []
        }
        
        self.start_time = time.time()
        
        logger.debug(f"Initialized OptimizationMonitor with metrics={self.metrics}, "
                    f"update_frequency={self.update_frequency}, save_plots={self.save_plots}")
    
    def update(self, 
              iteration: int, 
              x: np.ndarray, 
              loss: float, 
              gradient: np.ndarray, 
              step_size: Optional[float] = None) -> None:
        """Update the monitor with new data.
        
        Args:
            iteration: Current iteration
            x: Current point
            loss: Current loss value
            gradient: Current gradient
            step_size: Current step size
        """
        # Store data
        self.history['iteration'].append(iteration)
        self.history['time'].append(time.time() - self.start_time)
        self.history['loss'].append(loss)
        self.history['gradient_norm'].append(np.linalg.norm(gradient))
        self.history['step_size'].append(step_size if step_size is not None else 0.0)
        self.history['x'].append(x.copy())
        
        # Update plots if needed
        if iteration % self.update_frequency == 0:
            self._update_plots()
            self._print_status(iteration, loss, gradient)
    
    def _update_plots(self) -> None:
        """Update plots with current data."""
        if not self.save_plots:
            return
        
        try:
            import matplotlib.pyplot as plt
            
            # Create figure
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            
            # Plot loss
            if 'loss' in self.metrics:
                ax = axes[0, 0]
                ax.plot(self.history['iteration'], self.history['loss'])
                ax.set_xlabel('Iteration')
                ax.set_ylabel('Loss')
                ax.set_title('Loss vs. Iteration')
                ax.grid(True)
            
            # Plot gradient norm
            if 'gradient_norm' in self.metrics:
                ax = axes[0, 1]
                ax.plot(self.history['iteration'], self.history['gradient_norm'])
                ax.set_xlabel('Iteration')
                ax.set_ylabel('Gradient Norm')
                ax.set_title('Gradient Norm vs. Iteration')
                ax.grid(True)
            
            # Plot step size
            if 'step_size' in self.metrics:
                ax = axes[1, 0]
                ax.plot(self.history['iteration'], self.history['step_size'])
                ax.set_xlabel('Iteration')
                ax.set_ylabel('Step Size')
                ax.set_title('Step Size vs. Iteration')
                ax.grid(True)
            
            # Plot time
            if 'time' in self.metrics:
                ax = axes[1, 1]
                ax.plot(self.history['iteration'], self.history['time'])
                ax.set_xlabel('Iteration')
                ax.set_ylabel('Time (s)')
                ax.set_title('Time vs. Iteration')
                ax.grid(True)
            
            # Save figure
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, 'optimization_progress.png'))
            plt.close(fig)
            
            # Plot parameter trajectories if dimension is small
            if len(self.history['x'][0]) <= 10:
                fig, ax = plt.subplots(figsize=(10, 6))
                
                for i in range(len(self.history['x'][0])):
                    values = [x[i] for x in self.history['x']]
                    ax.plot(self.history['iteration'], values, label=f'x[{i}]')
                
                ax.set_xlabel('Iteration')
                ax.set_ylabel('Parameter Value')
                ax.set_title('Parameter Trajectories')
                ax.grid(True)
                ax.legend()
                
                plt.tight_layout()
                plt.savefig(os.path.join(self.output_dir, 'parameter_trajectories.png'))
                plt.close(fig)
        
        except ImportError:
            logger.warning("matplotlib not found, plotting is not available")
        except Exception as e:
            logger.error(f"Error updating plots: {e}")
    
    def _print_status(self, iteration: int, loss: float, gradient: np.ndarray) -> None:
        """Print current optimization status.
        
        Args:
            iteration: Current iteration
            loss: Current loss value
            gradient: Current gradient
        """
        elapsed_time = time.time() - self.start_time
        gradient_norm = np.linalg.norm(gradient)
        
        logger.info(f"Iteration {iteration}: loss={loss:.6e}, ||g||={gradient_norm:.6e}, time={elapsed_time:.2f}s")
    
    def save_history(self, filename: Optional[str] = None) -> None:
        """Save optimization history to file.
        
        Args:
            filename: Name of the file to save
        """
        if not self.save_plots:
            return
        
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"optimization_history_{timestamp}.npz"
        
        filepath = os.path.join(self.output_dir, filename)
        
        # Convert x history to array
        x_history = np.array(self.history['x'])
        
        # Save history
        np.savez(
            filepath,
            iteration=np.array(self.history['iteration']),
            time=np.array(self.history['time']),
            loss=np.array(self.history['loss']),
            gradient_norm=np.array(self.history['gradient_norm']),
            step_size=np.array(self.history['step_size']),
            x=x_history
        )
        
        logger.info(f"Saved optimization history to {filepath}")
    
    def generate_report(self, filename: Optional[str] = None) -> None:
        """Generate optimization report.
        
        Args:
            filename: Name of the file to save
        """
        if not self.save_plots:
            return
        
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"optimization_report_{timestamp}.html"
        
        filepath = os.path.join(self.output_dir, filename)
        
        try:
            # Generate HTML report
            with open(filepath, 'w') as f:
                f.write(f"""
                <!DOCTYPE html>
                <html>
                <head>
                    <title>M.I.A.-simbolic Optimization Report</title>
                    <style>
                        body {{ font-family: Arial, sans-serif; margin: 20px; }}
                        h1 {{ color: #2c3e50; }}
                        h2 {{ color: #3498db; }}
                        table {{ border-collapse: collapse; width: 100%; }}
                        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                        th {{ background-color: #f2f2f2; }}
                        tr:nth-child(even) {{ background-color: #f9f9f9; }}
                        .container {{ display: flex; flex-wrap: wrap; }}
                        .plot {{ margin: 10px; }}
                    </style>
                </head>
                <body>
                    <h1>M.I.A.-simbolic Optimization Report</h1>
                    <p>Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                    
                    <h2>Summary</h2>
                    <table>
                        <tr>
                            <th>Metric</th>
                            <th>Value</th>
                        </tr>
                        <tr>
                            <td>Total Iterations</td>
                            <td>{self.history['iteration'][-1] + 1}</td>
                        </tr>
                        <tr>
                            <td>Final Loss</td>
                            <td>{self.history['loss'][-1]:.6e}</td>
                        </tr>
                        <tr>
                            <td>Final Gradient Norm</td>
                            <td>{self.history['gradient_norm'][-1]:.6e}</td>
                        </tr>
                        <tr>
                            <td>Total Time</td>
                            <td>{self.history['time'][-1]:.2f} seconds</td>
                        </tr>
                    </table>
                    
                    <h2>Plots</h2>
                    <div class="container">
                        <div class="plot">
                            <img src="optimization_progress.png" alt="Optimization Progress">
                        </div>
                        <div class="plot">
                            <img src="parameter_trajectories.png" alt="Parameter Trajectories">
                        </div>
                    </div>
                    
                    <h2>Convergence Analysis</h2>
                    <p>
                        The optimization process {'converged' if self.history['gradient_norm'][-1] < 1e-5 else 'did not converge'}.
                        The final gradient norm is {self.history['gradient_norm'][-1]:.6e}.
                    </p>
                    
                    <h2>Performance Analysis</h2>
                    <p>
                        The optimization took {self.history['time'][-1]:.2f} seconds for {self.history['iteration'][-1] + 1} iterations,
                        with an average of {self.history['time'][-1] / (self.history['iteration'][-1] + 1):.6f} seconds per iteration.
                    </p>
                </body>
                </html>
                """)
            
            logger.info(f"Generated optimization report at {filepath}")
        
        except Exception as e:
            logger.error(f"Error generating report: {e}")
    
    def plot_contour(self, 
                    objective_function: callable, 
                    bounds: List[Tuple[float, float]], 
                    resolution: int = 100, 
                    filename: Optional[str] = None) -> None:
        """Plot contour of the objective function.
        
        Args:
            objective_function: Objective function
            bounds: Bounds for variables
            resolution: Resolution of the grid
            filename: Name of the file to save
        """
        if not self.save_plots:
            return
        
        if len(bounds) != 2:
            logger.warning("Contour plot is only available for 2D problems")
            return
        
        try:
            import matplotlib.pyplot as plt
            
            # Create grid
            x_min, x_max = bounds[0]
            y_min, y_max = bounds[1]
            
            x = np.linspace(x_min, x_max, resolution)
            y = np.linspace(y_min, y_max, resolution)
            
            X, Y = np.meshgrid(x, y)
            Z = np.zeros_like(X)
            
            # Evaluate function on grid
            for i in range(resolution):
                for j in range(resolution):
                    Z[i, j] = objective_function(np.array([X[i, j], Y[i, j]]))
            
            # Create figure
            fig, ax = plt.subplots(figsize=(10, 8))
            
            # Plot contour
            contour = ax.contourf(X, Y, Z, 50, cmap='viridis')
            fig.colorbar(contour, ax=ax)
            
            # Plot optimization trajectory
            x_history = np.array(self.history['x'])
            ax.plot(x_history[:, 0], x_history[:, 1], 'r-', linewidth=2)
            ax.plot(x_history[:, 0], x_history[:, 1], 'ro', markersize=4)
            
            # Mark start and end points
            ax.plot(x_history[0, 0], x_history[0, 1], 'go', markersize=8, label='Start')
            ax.plot(x_history[-1, 0], x_history[-1, 1], 'bo', markersize=8, label='End')
            
            ax.set_xlabel('x[0]')
            ax.set_ylabel('x[1]')
            ax.set_title('Objective Function Contour and Optimization Trajectory')
            ax.legend()
            
            # Save figure
            if filename is None:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                filename = f"contour_plot_{timestamp}.png"
            
            filepath = os.path.join(self.output_dir, filename)
            plt.savefig(filepath)
            plt.close(fig)
            
            logger.info(f"Saved contour plot to {filepath}")
        
        except ImportError:
            logger.warning("matplotlib not found, plotting is not available")
        except Exception as e:
            logger.error(f"Error plotting contour: {e}")
    
    def plot_3d_surface(self, 
                       objective_function: callable, 
                       bounds: List[Tuple[float, float]], 
                       resolution: int = 50, 
                       filename: Optional[str] = None) -> None:
        """Plot 3D surface of the objective function.
        
        Args:
            objective_function: Objective function
            bounds: Bounds for variables
            resolution: Resolution of the grid
            filename: Name of the file to save
        """
        if not self.save_plots:
            return
        
        if len(bounds) != 2:
            logger.warning("3D surface plot is only available for 2D problems")
            return
        
        try:
            import matplotlib.pyplot as plt
            from mpl_toolkits.mplot3d import Axes3D
            
            # Create grid
            x_min, x_max = bounds[0]
            y_min, y_max = bounds[1]
            
            x = np.linspace(x_min, x_max, resolution)
            y = np.linspace(y_min, y_max, resolution)
            
            X, Y = np.meshgrid(x, y)
            Z = np.zeros_like(X)
            
            # Evaluate function on grid
            for i in range(resolution):
                for j in range(resolution):
                    Z[i, j] = objective_function(np.array([X[i, j], Y[i, j]]))
            
            # Create figure
            fig = plt.figure(figsize=(12, 10))
            ax = fig.add_subplot(111, projection='3d')
            
            # Plot surface
            surface = ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8)
            fig.colorbar(surface, ax=ax, shrink=0.5, aspect=5)
            
            # Plot optimization trajectory
            x_history = np.array(self.history['x'])
            z_history = np.array(self.history['loss'])
            
            ax.plot(x_history[:, 0], x_history[:, 1], z_history, 'r-', linewidth=2)
            ax.plot(x_history[:, 0], x_history[:, 1], z_history, 'ro', markersize=4)
            
            # Mark start and end points
            ax.plot([x_history[0, 0]], [x_history[0, 1]], [z_history[0]], 'go', markersize=8, label='Start')
            ax.plot([x_history[-1, 0]], [x_history[-1, 1]], [z_history[-1]], 'bo', markersize=8, label='End')
            
            ax.set_xlabel('x[0]')
            ax.set_ylabel('x[1]')
            ax.set_zlabel('f(x)')
            ax.set_title('Objective Function Surface and Optimization Trajectory')
            ax.legend()
            
            # Save figure
            if filename is None:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                filename = f"surface_plot_{timestamp}.png"
            
            filepath = os.path.join(self.output_dir, filename)
            plt.savefig(filepath)
            plt.close(fig)
            
            logger.info(f"Saved 3D surface plot to {filepath}")
        
        except ImportError:
            logger.warning("matplotlib not found, plotting is not available")
        except Exception as e:
            logger.error(f"Error plotting 3D surface: {e}")