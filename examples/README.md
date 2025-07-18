# M.I.A.-simbolic Examples

This directory contains example scripts demonstrating how to use the M.I.A.-simbolic optimizer.

## Simple Optimization

The `simple_optimization.py` script demonstrates how to use the MIAOptimizer to optimize a simple function (Rosenbrock function).

```bash
python simple_optimization.py
```

## Multi-Objective Optimization

The `multi_objective_optimization.py` script demonstrates how to create a custom multi-objective problem and optimize it with different weights.

```bash
python multi_objective_optimization.py
```

## PyTorch Integration

The `pytorch_integration.py` script demonstrates how to use the MIAPyTorchOptimizer with a simple neural network for regression.

```bash
python pytorch_integration.py
```

## Benchmark Comparison

The `benchmark_comparison.py` script demonstrates how to use the BenchmarkSuite to compare the M.I.A.-simbolic optimizer with other optimization algorithms.

```bash
python benchmark_comparison.py
```

## Validation Protocol

The `validation_protocol.py` script demonstrates how to use the ValidationProtocol to validate the M.I.A.-simbolic optimizer on standard test problems.

```bash
python validation_protocol.py
```

## Requirements

To run these examples, you need to install the M.I.A.-simbolic package and its dependencies:

```bash
pip install -e ..
```

For the PyTorch integration example, you also need to install PyTorch:

```bash
pip install torch
```

For the benchmark comparison example, you need to install SciPy:

```bash
pip install scipy
```

For visualization, you need to install Matplotlib:

```bash
pip install matplotlib
```

Or you can install all dependencies at once:

```bash
pip install -e ..[all]
```