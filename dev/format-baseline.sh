#!/bin/bash


echo "Formatting started"
echo "Run isort"
python -m isort .
echo "Run black"
python -m black -q .
echo "Run docformatter"
python -m docformatter -i -r .
echo "Run ruff"
python -m ruff check --fix .
echo "Formatting done"
