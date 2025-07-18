#!/usr/bin/env python3
"""
M.I.A.-simbolic: Multi-Agent Threshold Optimization
The first multi-agent optimization system with guaranteed convergence.
"""

from setuptools import setup, find_packages
import os
import re

# Read version from __init__.py
def get_version():
    with open(os.path.join("src", "mia_simbolic", "__init__.py"), "r") as f:
        content = f.read()
        match = re.search(r"__version__ = ['\"]([^'\"]*)['\"]", content)
        if match:
            return match.group(1)
        raise RuntimeError("Unable to find version string.")

# Read long description from README
def get_long_description():
    with open("README.md", "r", encoding="utf-8") as f:
        return f.read()

# Read requirements
def get_requirements(filename):
    with open(filename, "r") as f:
        return [line.strip() for line in f if line.strip() and not line.startswith("#")]

setup(
    name="mia-simbolic",
    version=get_version(),
    author="M.I.A.-simbolic Team",
    author_email="contact@mia-simbolic.org",
    description="Multi-Agent Threshold Optimization with Guaranteed Convergence",
    long_description=get_long_description(),
    long_description_content_type="text/markdown",
    url="https://github.com/mia-simbolic/optimization",
    project_urls={
        "Documentation": "https://mia-simbolic.readthedocs.io/",
        "Source": "https://github.com/mia-simbolic/optimization",
        "Tracker": "https://github.com/mia-simbolic/optimization/issues",
        "Paper": "https://arxiv.org/abs/2025.xxxxx",
    },
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=get_requirements("requirements.txt"),
    extras_require={
        "dev": get_requirements("requirements-dev.txt"),
        "docs": get_requirements("requirements-docs.txt"),
        "distributed": ["horovod>=0.24.0", "mpi4py>=3.1.0"],
        "quantum": ["qiskit>=0.39.0", "cirq>=1.0.0"],
        "visualization": ["plotly>=5.0.0", "dash>=2.0.0"],
        "all": [
            "horovod>=0.24.0", "mpi4py>=3.1.0",
            "qiskit>=0.39.0", "cirq>=1.0.0",
            "plotly>=5.0.0", "dash>=2.0.0"
        ],
    },
    entry_points={
        "console_scripts": [
            "mia-optimize=mia_simbolic.cli:main",
            "mia-benchmark=mia_simbolic.benchmarks.cli:main",
            "mia-monitor=mia_simbolic.monitoring.cli:main",
        ],
    },
    include_package_data=True,
    package_data={
        "mia_simbolic": [
            "data/*.json",
            "data/*.yaml",
            "templates/*.html",
            "static/*",
        ],
    },
    zip_safe=False,
    keywords=[
        "optimization",
        "machine learning",
        "multi-agent",
        "convergence",
        "threshold",
        "neural networks",
        "deep learning",
        "bayesian optimization",
        "multi-objective",
    ],
    test_suite="tests",
    tests_require=get_requirements("requirements-test.txt"),
)

