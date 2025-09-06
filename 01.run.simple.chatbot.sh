#!/bin/bash

# NOTE: before running, need to login to below web to create token
# https://huggingface.co/settings/tokens

set -x

source chatbot-env/bin/activate

hf auth login --token [YOURTOKEN]

python3 app.simple.chatbot.py

set +x
