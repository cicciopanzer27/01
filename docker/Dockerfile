# Multi-stage build for M.I.A.-simbolic
FROM python:3.9-slim as base

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd --create-home --shell /bin/bash mia
WORKDIR /home/mia

# Development stage
FROM base as development

# Install development dependencies
COPY requirements-dev.txt .
RUN pip install -r requirements-dev.txt

# Copy source code
COPY --chown=mia:mia . .
RUN pip install -e ".[dev]"

USER mia
CMD ["bash"]

# Production stage
FROM base as production

# Install only production dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy source code and install
COPY --chown=mia:mia . .
RUN pip install .

# Create directories for data and results
RUN mkdir -p /home/mia/{data,results,logs} && \
    chown -R mia:mia /home/mia

USER mia

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import mia_simbolic; print('OK')" || exit 1

# Default command
CMD ["mia-optimize", "--help"]

# Validation stage for testing
FROM production as validation

# Copy test data and scripts
COPY --chown=mia:mia tests/ tests/
COPY --chown=mia:mia benchmarks/ benchmarks/
COPY --chown=mia:mia scripts/ scripts/

# Install test dependencies
USER root
RUN pip install pytest pytest-cov
USER mia

# Run validation tests
RUN python -m pytest tests/integration/ -v

# Benchmark stage for performance testing
FROM validation as benchmark

# Run performance benchmarks
RUN python scripts/run_benchmarks.py --quick --output /home/mia/results/benchmark_results.json

# Distributed stage for multi-node deployment
FROM production as distributed

USER root

# Install MPI and distributed computing dependencies
RUN apt-get update && apt-get install -y \
    openmpi-bin \
    openmpi-common \
    libopenmpi-dev \
    && rm -rf /var/lib/apt/lists/*

# Install distributed dependencies
RUN pip install horovod mpi4py

USER mia

# Expose ports for distributed communication
EXPOSE 8080 8081 8082

# Default command for distributed mode
CMD ["mpirun", "-np", "1", "python", "-m", "mia_simbolic.distributed.main"]

