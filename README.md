# On-Premise RAG Chatbot with GPU Acceleration

This project provides a comprehensive guide to building, debugging, and deploying a sophisticated conversational AI application. The chatbot runs entirely on-premise, leverages Retrieval-Augmented Generation (RAG) to answer questions based on webpage content, and is accelerated by an NVIDIA GPU on a cloud virtual machine.

## Features

* **100% On-Premise:** Runs on your own hardware without relying on external APIs like OpenAI or Google.
* **Conversational Memory:** The chatbot remembers the context of the ongoing conversation.
* **Retrieval-Augmented Generation (RAG):** Connect the chatbot to any webpage, allowing it to answer questions based on that specific knowledge base.
* **GPU Acceleration:** Utilizes NVIDIA GPUs through `llama-cpp-python` for significantly faster response times.
* **Interactive UI:** Built with Gradio, providing a user-friendly interface with separate tabs for knowledge base management and chat.

---

## 🚀 Installation and Setup Guide

This guide details the steps to deploy the application on a Google Cloud Platform (GCP) VM with an NVIDIA L4 GPU, running Ubuntu 22.04.

### Phase 1: Basic Python Environment Setup

First, set up a clean Python environment.

1.  **Create and Activate a Virtual Environment:**
    ```bash
    python3 -m venv chatbot-env
    source chatbot-env/bin/activate
    ```

2.  **Authenticate with Hugging Face:**
    You need an account to download the Llama 3 model.
    * Create an Access Token with "read" permissions at `huggingface.co/settings/tokens`.
    * Log in via the terminal:
        ```bash
        hf auth login
        ```

### Phase 2: System & GPU Environment Setup (Critical)

This phase prepares the virtual machine with the necessary build tools, drivers, and CUDA Toolkit.

1.  **Install Build Tools:**
    A new Ubuntu VM often lacks essential compilers.
    ```bash
    sudo apt-get update
    sudo apt-get install -y build-essential g++-10
    ```

2.  **Clean and Install NVIDIA Driver & CUDA Toolkit:**
    The default Ubuntu drivers are often outdated. This process installs the correct version directly from NVIDIA.
    ```bash
    # 1. Thoroughly purge all old NVIDIA/CUDA installations
    sudo apt-get --purge remove -y "*cuda*" "*cublas*" "nvidia-*"
    sudo apt-get autoremove -y
