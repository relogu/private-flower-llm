#!/bin/bash


echo "=== test.sh ==="

echo "- Start Python checks"

echo "- isort: start"
python -m isort --check-only .
echo "- isort: done"

echo "- black: start"
python -m black --check .
echo "- black: done"

echo "- docformatter: start"
python -m docformatter -c -r .
echo "- docformatter:  done"

echo "- ruff: start"
python -m ruff check .
echo "- ruff: done"

