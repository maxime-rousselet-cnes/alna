#!/usr/bin/env bash

cd ..

module load python/3.11.10

python -m venv alna_venv

source alna_venv/bin/activate

python -m pip install --upgrade pip setuptools wheel

cd base_models

python -m pip install -r requirements.txt

python -m pip install -e .

python -m pytest test.py

cd ../alna

python -m pip install -r requirements.txt

python -m pip install -e .

python -m pytest tests_base.py