#!/usr/bin/env bash
set -euo pipefail

echo "Formatting with black..."
python -m black .

echo "Sorting imports with isort..."
python -m isort .

echo "Linting with flake8..."
python -m flake8 .

echo "All checks passed."
