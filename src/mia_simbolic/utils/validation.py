"""
Validation module for M.I.A.-simbolic.

This module implements the ValidationProtocol class, which provides
validation functionality for the M.I.A.-simbolic optimizer.
"""

import logging
import time
import os
import json
from typing import Dict, Any, Optional, List, Tuple, Union, Callable
from dataclasses import dataclass, field, asdict
from datetime import datetime

import numpy as np

from ..core.optimizer import MIAOptimizer, OptimizationResult
from ..config import Config

logger = logging.getLogger(__name__)

@dataclass
class ValidationResult:
    """Result of a validation run.
    
    Attributes:
        problem_name: Name of the problem
        dimension: Dimension of the problem
        run_id: ID of the run
        converged: Whether the optimization converged
        final_value: Final function value
        iterations: Number of iterations
        time: Time taken for optimization
        success: Whether the optimization was successful
        gradient_norm: Norm of the gradient at the optimal point
        efficiency_score: Efficiency score of the optimization
        expected_value: Expected optimal value
        relative_error: Relative error from expected value
        passed: Whether the validation passed
    """
    problem_name: str
    dimension: int
    run_id: int
    converged: bool
    final_value: float
    iterations: int
    time: float
    success: bool
    gradient_norm: Optional[float] = None
    efficiency_score: Optional[float] = None
    expected_value: Optional[float] = None
    relative_error: Optional[float] = None
    passed: bool = False
    error: Optional[str] = None


@dataclass
class ValidationResults:
    """Collection of validation results.
    
    Attributes:
        results: List of validation results
        timestamp: Timestamp of the validation
        config: Configuration used for the validation
        overall_passed: Whether the overall validation passed
    """
    results: List[ValidationResult] = field(default_factory=list)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    config: Dict[str, Any] = field(default_factory=dict)
    overall_passed: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert results to dictionary.
        
        Returns:
            Dictionary representation of results
        """
        return {
            'results': [asdict(result) for result in self.results],
            'timestamp': self.timestamp,
            'config': self.config,
            'overall_passed': self.overall_passed
        }
    
    def save(self, filename: str) -> None:
        """Save results to file.
        
        Args:
            filename: Name of the file to save
        """
        with open(filename, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
        
        logger.info(f"Saved validation results to {filename}")
    
    @classmethod
    def load(cls, filename: str) -> 'ValidationResults':
        """Load results from file.
        
        Args:
            filename: Name of the file to load
            
        Returns:
            Loaded validation results
        """
        with open(filename, 'r') as f:
            data = json.load(f)
        
        results = cls(
            results=[ValidationResult(**result) for result in data['results']],
            timestamp=data['timestamp'],
            config=data['config'],
            overall_passed=data['overall_passed']
        )
        
        logger.info(f"Loaded validation results from {filename}")
        
        return results
    
    def summary(self) -> Dict[str, Any]:
        """Generate summary of results.
        
        Returns:
            Dictionary with summary statistics
        """
        # Group results by problem
        grouped = {}
        for result in self.results:
            key = (result.problem_name, result.dimension)
            if key not in grouped:
                grouped[key] = []
            grouped[key].append(result)
        
        # Compute statistics
        summary = {}
        for (problem, dimension), results in grouped.items():
            key = f"{problem}_{dimension}"
            summary[key] = {
                'problem': problem,
                'dimension': dimension,
                'num_runs': len(results),
                'pass_rate': sum(r.passed for r in results) / len(results),
                'convergence_rate': sum(r.converged for r in results) / len(results),
                'mean_time': np.mean([r.time for r in results]),
                'mean_iterations': np.mean([r.iterations for r in results]),
                'mean_final_value': np.mean([r.final_value for r in results]),
                'mean_relative_error': np.mean([r.relative_error for r in results if r.relative_error is not None]),
                'all_passed': all(r.passed for r in results)
            }
        
        return summary


class ValidationProblem:
    """Base class for validation problems.
    
    Attributes:
        name: Name of the problem
        dimension: Dimension of the problem
        bounds: Bounds for variables
        expected_value: Expected optimal value
        tolerance: Tolerance for validation
    """
    
    def __init__(self, 
                name: str, 
                dimension: int, 
                expected_value: float, 
                tolerance: float = 1e-6, 
                bounds: Optional[List[Tuple[float, float]]] = None):
        """Initialize the validation problem.
        
        Args:
            name: Name of the problem
            dimension: Dimension of the problem
            expected_value: Expected optimal value
            tolerance: Tolerance for validation
            bounds: Bounds for variables
        """
        self.name = name
        self.dimension = dimension
        self.expected_value = expected_value
        self.tolerance = tolerance
        self.bounds = bounds or [(-10.0, 10.0)] * dimension
        
        logger.debug(f"Initialized {self.name} validation problem with dimension {self.dimension}")
    
    def objective(self, x: np.ndarray) -> float:
        """Compute the objective function value.
        
        Args:
            x: Point at which to evaluate the objective
            
        Returns:
            Objective function value
        """
        raise NotImplementedError("Subclasses must implement this method")
    
    def gradient(self, x: np.ndarray) -> np.ndarray:
        """Compute the gradient of the objective function.
        
        Args:
            x: Point at which to evaluate the gradient
            
        Returns:
            Gradient vector
        """
        # Default implementation uses finite differences
        eps = 1e-6
        n = len(x)
        grad = np.zeros(n)
        
        for i in range(n):
            x_plus = x.copy()
            x_minus = x.copy()
            x_plus[i] += eps
            x_minus[i] -= eps
            
            f_plus = self.objective(x_plus)
            f_minus = self.objective(x_minus)
            
            grad[i] = (f_plus - f_minus) / (2 * eps)
        
        return grad
    
    def initial_point(self, seed: Optional[int] = None) -> np.ndarray:
        """Generate an initial point for optimization.
        
        Args:
            seed: Random seed
            
        Returns:
            Initial point
        """
        if seed is not None:
            np.random.seed(seed)
        
        # Generate random point within bounds
        x = np.zeros(self.dimension)
        for i in range(self.dimension):
            x[i] = np.random.uniform(self.bounds[i][0], self.bounds[i][1])
        
        return x
    
    def validate(self, result: OptimizationResult) -> bool:
        """Validate the optimization result.
        
        Args:
            result: Optimization result
            
        Returns:
            True if the result is valid, False otherwise
        """
        # Check if the optimization converged
        if not result.converged:
            logger.warning(f"Validation failed: optimization did not converge")
            return False
        
        # Check if the final value is close to the expected value
        relative_error = abs(result.fun - self.expected_value) / (abs(self.expected_value) + 1e-10)
        if relative_error > self.tolerance:
            logger.warning(f"Validation failed: relative error {relative_error:.6e} > tolerance {self.tolerance:.6e}")
            return False
        
        # Check if the gradient norm is small
        if result.gradient_norm is not None and result.gradient_norm > self.tolerance:
            logger.warning(f"Validation failed: gradient norm {result.gradient_norm:.6e} > tolerance {self.tolerance:.6e}")
            return False
        
        return True


class SphereValidation(ValidationProblem):
    """Sphere function validation problem.
    
    f(x) = sum(x_i^2)
    
    The expected optimal value is 0.0 at x = [0, 0, ..., 0].
    """
    
    def __init__(self, dimension: int = 10, tolerance: float = 1e-6):
        """Initialize the sphere function validation problem.
        
        Args:
            dimension: Dimension of the problem
            tolerance: Tolerance for validation
        """
        super().__init__("Sphere", dimension, expected_value=0.0, tolerance=tolerance)
    
    def objective(self, x: np.ndarray) -> float:
        """Compute the objective function value.
        
        Args:
            x: Point at which to evaluate the objective
            
        Returns:
            Objective function value
        """
        return np.sum(x**2)
    
    def gradient(self, x: np.ndarray) -> np.ndarray:
        """Compute the gradient of the objective function.
        
        Args:
            x: Point at which to evaluate the gradient
            
        Returns:
            Gradient vector
        """
        return 2 * x


class RosenbrockValidation(ValidationProblem):
    """Rosenbrock function validation problem.
    
    f(x) = sum(100 * (x_{i+1} - x_i^2)^2 + (x_i - 1)^2)
    
    The expected optimal value is 0.0 at x = [1, 1, ..., 1].
    """
    
    def __init__(self, dimension: int = 10, tolerance: float = 1e-6):
        """Initialize the Rosenbrock function validation problem.
        
        Args:
            dimension: Dimension of the problem
            tolerance: Tolerance for validation
        """
        super().__init__("Rosenbrock", dimension, expected_value=0.0, tolerance=tolerance)
    
    def objective(self, x: np.ndarray) -> float:
        """Compute the objective function value.
        
        Args:
            x: Point at which to evaluate the objective
            
        Returns:
            Objective function value
        """
        value = 0.0
        for i in range(self.dimension - 1):
            value += 100 * (x[i+1] - x[i]**2)**2 + (x[i] - 1)**2
        
        return value


class ValidationProtocol:
    """Protocol for validating the M.I.A.-simbolic optimizer.
    
    This class provides functionality for validating the optimizer
    on a variety of test problems.
    
    Attributes:
        problems: Dictionary of validation problems
        n_runs: Number of runs per problem
        output_dir: Directory for output files
        config: Configuration for the optimizer
    """
    
    def __init__(self, 
                n_runs: int = 10, 
                output_dir: Optional[str] = None, 
                config: Optional[Config] = None):
        """Initialize the validation protocol.
        
        Args:
            n_runs: Number of runs per problem
            output_dir: Directory for output files
            config: Configuration for the optimizer
        """
        self.n_runs = n_runs
        self.output_dir = output_dir or os.path.join(os.getcwd(), 'results', 'validation')
        self.config = config or Config()
        
        # Create output directory if it doesn't exist
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        
        # Initialize problems
        self.problems = {
            'sphere': lambda dim: SphereValidation(dimension=dim),
            'rosenbrock': lambda dim: RosenbrockValidation(dimension=dim)
        }
        
        logger.debug(f"Initialized ValidationProtocol with n_runs={n_runs}")
    
    def add_problem(self, name: str, problem_factory: Callable[[int], ValidationProblem]) -> None:
        """Add a problem to the validation protocol.
        
        Args:
            name: Name of the problem
            problem_factory: Factory function to create the problem
        """
        self.problems[name] = problem_factory
        logger.debug(f"Added problem {name} to validation protocol")
    
    def run(self, 
           problem_class: str = "all", 
           dimensions: Optional[List[int]] = None) -> ValidationResults:
        """Run the validation protocol.
        
        Args:
            problem_class: Class of problems to validate
            dimensions: List of dimensions to validate
            
        Returns:
            Validation results
        """
        # Set default values
        dimensions = dimensions or [10, 50, 100]
        
        # Select problems
        if problem_class == "all":
            problems = list(self.problems.keys())
        else:
            problems = [problem_class]
        
        # Initialize results
        results = ValidationResults(
            config={
                'n_runs': self.n_runs,
                'problems': problems,
                'dimensions': dimensions
            }
        )
        
        # Run validation
        total_validations = len(problems) * len(dimensions) * self.n_runs
        validation_count = 0
        
        logger.info(f"Starting validation protocol with {total_validations} validations")
        start_time = time.time()
        
        for problem_name in problems:
            for dimension in dimensions:
                for run_id in range(self.n_runs):
                    validation_count += 1
                    logger.info(f"Running validation {validation_count}/{total_validations}: "
                               f"{problem_name} (dim={dimension}, run={run_id})")
                    
                    # Create problem
                    problem = self.problems[problem_name](dimension)
                    
                    # Generate initial point
                    initial_point = problem.initial_point(seed=run_id)
                    
                    # Create optimizer
                    optimizer = MIAOptimizer(config=self.config)
                    
                    # Run optimization
                    try:
                        start_time_opt = time.time()
                        result = optimizer.optimize(
                            problem.objective,
                            initial_point,
                            bounds=problem.bounds
                        )
                        end_time_opt = time.time()
                        
                        # Validate result
                        passed = problem.validate(result)
                        
                        # Compute relative error
                        relative_error = abs(result.fun - problem.expected_value) / (abs(problem.expected_value) + 1e-10)
                        
                        # Create validation result
                        validation_result = ValidationResult(
                            problem_name=problem_name,
                            dimension=dimension,
                            run_id=run_id,
                            converged=result.converged,
                            final_value=result.fun,
                            iterations=result.nit,
                            time=end_time_opt - start_time_opt,
                            success=result.success,
                            gradient_norm=result.gradient_norm,
                            efficiency_score=result.efficiency_score,
                            expected_value=problem.expected_value,
                            relative_error=relative_error,
                            passed=passed
                        )
                        
                        results.results.append(validation_result)
                    
                    except Exception as e:
                        logger.error(f"Error in validation of {problem_name}: {e}")
                        
                        # Record error
                        validation_result = ValidationResult(
                            problem_name=problem_name,
                            dimension=dimension,
                            run_id=run_id,
                            converged=False,
                            final_value=float('inf'),
                            iterations=0,
                            time=0.0,
                            success=False,
                            expected_value=problem.expected_value,
                            relative_error=float('inf'),
                            passed=False,
                            error=str(e)
                        )
                        
                        results.results.append(validation_result)
        
        end_time = time.time()
        logger.info(f"Validation protocol completed in {end_time - start_time:.2f} seconds")
        
        # Check if all validations passed
        results.overall_passed = all(result.passed for result in results.results)
        
        # Save results
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = os.path.join(self.output_dir, f"validation_results_{timestamp}.json")
        results.save(filename)
        
        return results
    
    def generate_report(self, results: ValidationResults, filename: Optional[str] = None) -> None:
        """Generate validation report.
        
        Args:
            results: Validation results
            filename: Name of the file to save
        """
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = os.path.join(self.output_dir, f"validation_report_{timestamp}.html")
        else:
            filename = os.path.join(self.output_dir, filename)
        
        try:
            # Get summary statistics
            summary = results.summary()
            
            # Generate HTML report
            with open(filename, 'w') as f:
                f.write(f"""
                <!DOCTYPE html>
                <html>
                <head>
                    <title>M.I.A.-simbolic Validation Report</title>
                    <style>
                        body {{ font-family: Arial, sans-serif; margin: 20px; }}
                        h1 {{ color: #2c3e50; }}
                        h2 {{ color: #3498db; }}
                        table {{ border-collapse: collapse; width: 100%; }}
                        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                        th {{ background-color: #f2f2f2; }}
                        tr:nth-child(even) {{ background-color: #f9f9f9; }}
                        .passed {{ color: green; }}
                        .failed {{ color: red; }}
                    </style>
                </head>
                <body>
                    <h1>M.I.A.-simbolic Validation Report</h1>
                    <p>Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                    
                    <h2>Summary</h2>
                    <p class="{'passed' if results.overall_passed else 'failed'}">
                        Overall validation: {'PASSED' if results.overall_passed else 'FAILED'}
                    </p>
                    
                    <table>
                        <tr>
                            <th>Problem</th>
                            <th>Dimension</th>
                            <th>Pass Rate</th>
                            <th>Convergence Rate</th>
                            <th>Mean Time (s)</th>
                            <th>Mean Iterations</th>
                            <th>Mean Relative Error</th>
                            <th>Status</th>
                        </tr>
                """)
                
                for key, stats in summary.items():
                    f.write(f"""
                        <tr>
                            <td>{stats['problem']}</td>
                            <td>{stats['dimension']}</td>
                            <td>{stats['pass_rate']:.2f}</td>
                            <td>{stats['convergence_rate']:.2f}</td>
                            <td>{stats['mean_time']:.4f}</td>
                            <td>{stats['mean_iterations']:.1f}</td>
                            <td>{stats['mean_relative_error']:.6e}</td>
                            <td class="{'passed' if stats['all_passed'] else 'failed'}">
                                {'PASSED' if stats['all_passed'] else 'FAILED'}
                            </td>
                        </tr>
                    """)
                
                f.write("""
                    </table>
                    
                    <h2>Detailed Results</h2>
                    <table>
                        <tr>
                            <th>Problem</th>
                            <th>Dimension</th>
                            <th>Run</th>
                            <th>Converged</th>
                            <th>Final Value</th>
                            <th>Expected Value</th>
                            <th>Relative Error</th>
                            <th>Iterations</th>
                            <th>Time (s)</th>
                            <th>Status</th>
                        </tr>
                """)
                
                for result in results.results:
                    f.write(f"""
                        <tr>
                            <td>{result.problem_name}</td>
                            <td>{result.dimension}</td>
                            <td>{result.run_id}</td>
                            <td>{result.converged}</td>
                            <td>{result.final_value:.6e}</td>
                            <td>{result.expected_value:.6e if result.expected_value is not None else 'N/A'}</td>
                            <td>{result.relative_error:.6e if result.relative_error is not None else 'N/A'}</td>
                            <td>{result.iterations}</td>
                            <td>{result.time:.4f}</td>
                            <td class="{'passed' if result.passed else 'failed'}">
                                {'PASSED' if result.passed else 'FAILED'}
                            </td>
                        </tr>
                    """)
                
                f.write("""
                    </table>
                    
                    <h2>Configuration</h2>
                    <table>
                        <tr>
                            <th>Parameter</th>
                            <th>Value</th>
                        </tr>
                """)
                
                for key, value in results.config.items():
                    f.write(f"""
                        <tr>
                            <td>{key}</td>
                            <td>{value}</td>
                        </tr>
                    """)
                
                f.write("""
                    </table>
                </body>
                </html>
                """)
            
            logger.info(f"Generated validation report at {filename}")
        
        except Exception as e:
            logger.error(f"Error generating report: {e}")