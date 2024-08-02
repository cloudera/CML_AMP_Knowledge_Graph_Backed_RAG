#!/bin/bash

echo "installing GPU version of llama-cpp"
pip install llama-cpp-python --upgrade --force-reinstall --no-cache-dir --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu122
