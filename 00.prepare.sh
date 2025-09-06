#!/bin/bash

set -x

sudo apt update
sudo apt install python3-venv

python3 -m venv chatbot-env
source chatbot-env/bin/activate

# This for simpe.chatbot
pip install llama-cpp-python gradio huggingface_hub

# This for rag.chatbot
pip install "packaging<25"
pip install langchain langchain-community beautifulsoup4 faiss-cpu sentence-transformers
pip install -U langchain-huggingface

set +x
