#!/bin/bash
set -e

# Default to script mode
MODE="${1:-script}"
shift || true

case "$MODE" in
    script)
        echo "Running demo script..."
        exec julia --project=/app demo_iaf.jl "$@"
        ;;
    jupyter|notebook)
        echo "Starting Jupyter notebook server on port 8888..."
        exec julia --project=/app -e 'using IJulia; notebook(dir="/app/notebooks", port=8888)'
        ;;
    julia|repl)
        echo "Starting Julia REPL..."
        exec julia --project=/app -e 'push!(LOAD_PATH, "/app/src")' "$@"
        ;;
    *)
        echo "Usage: $0 {script|jupyter|notebook|julia|repl}"
        echo "  script   - Run demo_iaf.jl (default)"
        echo "  jupyter  - Start Jupyter notebook server"
        echo "  notebook - Alias for jupyter"
        echo "  julia    - Start Julia REPL (pass additional args after)"
        echo "  repl     - Alias for julia"
        exit 1
        ;;
esac