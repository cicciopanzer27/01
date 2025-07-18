# M.I.A.-simbolic Tests

This directory contains tests for the M.I.A.-simbolic optimizer.

## Test Structure

The tests are organized into three categories:

- **Unit Tests**: Tests for individual components of the optimizer.
- **Integration Tests**: Tests for the interaction between components.
- **Performance Tests**: Tests for the performance and scalability of the optimizer.

## Running Tests

You can run all tests using the `run_tests.py` script:

```bash
python run_tests.py
```

You can also run specific types of tests:

```bash
python run_tests.py --type unit
python run_tests.py --type integration
python run_tests.py --type performance
```

For verbose output, use the `--verbose` flag:

```bash
python run_tests.py --verbose
```

## Test Requirements

To run the tests, you need to install the test dependencies:

```bash
pip install -e ..[dev]
```

Or install the specific test requirements:

```bash
pip install pytest pytest-cov
```

## Running Tests with pytest

You can also run the tests using pytest directly:

```bash
pytest tests/unit
pytest tests/integration
pytest tests/performance
```

To generate a coverage report:

```bash
pytest --cov=src/mia_simbolic tests/
```