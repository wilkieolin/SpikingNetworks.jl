# syntax=docker/dockerfile:1
FROM julia:1.12

# Install system dependencies for Jupyter and optional GPU support
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-venv \
    && python3 -m venv /opt/venv \
    && /opt/venv/bin/pip install --no-cache-dir jupyter \
    && rm -rf /var/lib/apt/lists/*
ENV PATH="/opt/venv/bin:$PATH"

WORKDIR /app

# Copy package source and Project.toml first
COPY src/ ./src/
COPY Project.toml ./

# Install Julia dependencies (resolves latest compatible versions) including the local package
RUN julia --project=/app -e 'using Pkg; Pkg.instantiate(); Pkg.precompile()'

# Copy entry points
COPY scripts/demo_iaf.jl ./
COPY notebooks/ ./notebooks/
COPY docker-entrypoint.sh ./

RUN chmod +x docker-entrypoint.sh

EXPOSE 8888

ENTRYPOINT ["/app/docker-entrypoint.sh"]