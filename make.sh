#!/bin/bash

poetry run python ./build.py
poetry run pytest -v tests/
poetry run python ./bench.py
