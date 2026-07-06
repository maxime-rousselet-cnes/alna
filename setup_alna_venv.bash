#!/usr/bin/env bash

cd ..

python3 -m venv alna_venv

source alna_venv/bin/activate

python3 -m pip install --upgrade pip setuptools wheel

cd base_models

python3 -m pip install -r requirements.txt

python3 -m pip install -e .

python3 -m pytest test.py

cd ../alna

python3 -m pip install -r requirements.txt

python3 -m pip install -e .

pthon3 -m pytest tests_base.py