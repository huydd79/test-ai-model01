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
    sudo apt-get autoclean
    sudo rm -rf /usr/local/cuda*
    
    # 2. Install the cuda-12-5 meta-package from NVIDIA
    wget [https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb](https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb)
    sudo dpkg -i cuda-keyring_1.1-1_all.deb
    sudo apt-get update
    sudo apt-get -y install cuda-12-5
    
    # 3. Set up environment variables
    echo 'export PATH=/usr/local/cuda-12.5/bin${PATH:+:${PATH}}' | sudo tee /etc/profile.d/cuda.sh
    echo 'export LD_LIBRARY_PATH=/usr/local/cuda-12.5/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}' | sudo tee -a /etc/profile.d/cuda.sh
    
    # 4. Reboot the VM
    echo "System will now reboot to apply driver updates..."
    sudo reboot
    ```

3.  **Verify Installation Post-Reboot:**
    After the VM reboots, log back in and run these commands to confirm the setup:
    ```bash
    # Check the driver and GPU status
    nvidia-smi
    
    # Check the CUDA compiler version
    nvcc --version
    ```
    You should see `Driver Version: 555.xx` and `CUDA Version: 12.5`.

### Phase 3: Install GPU-Compatible Python Libraries

Now, install the Python dependencies, ensuring they are linked against the new CUDA environment.

1.  **Purge Pip Cache:**
    ```bash
    pip cache purge
    ```

2.  **Install PyTorch for CUDA 12.1:**
    This specific version is known to be compatible with the CUDA 12.5 environment.
    ```bash
    pip install torch torchvision torchaudio --index-url [https://download.pytorch.org/whl/cu121](https://download.pytorch.org/whl/cu121)
    ```

3.  **Install `llama-cpp-python` with GPU Support:**
    This command compiles the library from source with the correct flags for your environment.
    ```bash
    CMAKE_ARGS="-DGGML_CUDA=on -DCMAKE_CXX_COMPILER=/usr/bin/g++-10" FORCE_CMAKE=1 pip install llama-cpp-python
    ```

4.  **Install Remaining Libraries:**
    ```bash
    pip install gradio huggingface_hub requests beautifulsoup4 langchain langchain-huggingface langchain-community faiss-cpu sentence-transformers tqdm
    ```

---

## ▶️ Running the Application

1.  **Update the Code to Use GPU:**
    Ensure your Python script (e.g., `app-gpu.py`) has the `n_gpu_layers` parameter set to `-1` to offload the model to the GPU:
    ```python
    llm = Llama(
        # ... other parameters
        n_gpu_layers=-1 
    )
    ```

2.  **Launch the Application:**
    * The application requires `sudo` permissions to run on port 80.
    * In the first session after installation, you must specify the CUDA library path.
    ```bash
    sudo LD_LIBRARY_PATH=/usr/local/cuda-12.5/lib64 python app-gpu.py
    ```
    * In subsequent login sessions, the environment variable should be loaded automatically, so you may only need:
    ```bash
    sudo python app-gpu.py
    ```

3.  **Access the UI:**
    Open a web browser and navigate to the public IP address of your GCP virtual machine.
