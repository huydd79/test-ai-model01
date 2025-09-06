#!/bin/bash

# NOTE: before running, need to login to below web to create token
# https://huggingface.co/settings/tokens

set -x

source chatbot-env/bin/activate

hf auth login --token [YOUR-TOKEN]

python3 app.rag.chatbot.py

set +x
