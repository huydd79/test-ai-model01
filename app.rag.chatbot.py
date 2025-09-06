import gradio as gr
from llama_cpp import Llama
from huggingface_hub import hf_hub_download
import time
from tqdm import tqdm

# --- New Imports for RAG ---
import requests
from bs4 import BeautifulSoup
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# --- Constants for Llama 3 prompt construction ---
BOS_TOKEN = "<|begin_of_text|>"
EOT_TOKEN = "<|eot_id|>"
SYS_HEADER = "<|start_header_id|>system<|end_header_id|>\n\n"
USER_HEADER = "<|start_header_id|>user<|end_header_id|>\n\n"
ASSISTANT_HEADER = "<|start_header_id|>assistant<|end_header_id|>\n\n"

# --- Step 1: Download model from Hugging Face Hub ---
model_name = "MaziyarPanahi/Meta-Llama-3-8B-Instruct-GGUF"
model_file ="Meta-Llama-3-8B-Instruct.Q4_K_S.gguf" 

print("Downloading model (or getting from cache)...")
model_path = hf_hub_download(repo_id=model_name, filename=model_file)
print("Model path ready!")

# --- Step 2: Initialize LLM and Embeddings Model ---
print("Initializing LLM...")
llm = Llama(
    model_path=model_path,
    n_ctx=8192,
    n_threads=4,
    n_gpu_layers=0,
    verbose=False
)
print("LLM Initialized!")

print("Initializing Embeddings Model...")
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
print("Embeddings Model Initialized!")


# --- Step 3: RAG Processing and Chatbot Logic (UPDATED with gr.Progress) ---

def process_webpage_to_retriever(url, progress=gr.Progress()):
    """Fetches, chunks, and vectorizes a webpage using a Gradio progress tracker."""
    try:
        progress(0.1, desc="Fetching webpage content...")
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        
        text = ' '.join(p.get_text() for p in soup.find_all('p'))
        if not text:
            return "Failed: No paragraph text found on the page.", None

        progress(0.3, desc="Splitting text into chunks...")
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        chunks = text_splitter.split_text(text)
        
        if not chunks:
            return "Failed: Could not split text into chunks.", None

        progress(0.5, desc=f"Creating vector store from {len(chunks)} chunks...")
        
        # --- BATCH PROCESSING LOGIC ---
        batch_size = 100
        vector_store = None
        
        # Use tqdm to automatically track progress for the progress bar
        for i in progress.tqdm(range(0, len(chunks), batch_size), desc="Processing Batches"):
            batch = chunks[i:i + batch_size]
            if vector_store is None:
                vector_store = FAISS.from_texts(batch, embeddings)
            else:
                vector_store.add_texts(batch)
        
        retriever = vector_store.as_retriever(search_kwargs={"k": 3})
        final_status = f"Successfully loaded and processed {len(chunks)} chunks from {url}"
        return final_status, retriever

    except Exception as e:
        error_status = f"An error occurred: {e}"
        return error_status, None

# ... (Rest of the functions format_rag_prompt, format_standard_prompt, etc. remain the same) ...
def format_rag_prompt(message, context):
    """Formats the prompt for RAG, instructing the model to use the context."""
    prompt_parts = [
        BOS_TOKEN,
        SYS_HEADER,
        "You are a helpful AI assistant. Answer the user's question based ONLY on the context provided below. If the information is not in the context, say 'I do not have enough information to answer that.'.",
        "\n\n--- CONTEXT ---\n",
        context,
        "\n--- END CONTEXT ---",
        EOT_TOKEN,
        f"{USER_HEADER}{message}{EOT_TOKEN}",
        ASSISTANT_HEADER,
    ]
    return "".join(prompt_parts)

def format_standard_prompt(message, history):
    """Formats a standard conversational prompt."""
    prompt_parts = [BOS_TOKEN, SYS_HEADER, "You are a helpful AI assistant.", EOT_TOKEN]
    for user_turn, bot_turn in history:
        prompt_parts.extend([f"{USER_HEADER}{user_turn}{EOT_TOKEN}", f"{ASSISTANT_HEADER}{bot_turn}{EOT_TOKEN}"])
    prompt_parts.extend([f"{USER_HEADER}{message}{EOT_TOKEN}", ASSISTANT_HEADER])
    return "".join(prompt_parts)

def chatbot_response(message, history, temperature, retriever):
    final_prompt = ""
    if retriever:
        # RAG Mode
        relevant_docs = retriever.get_relevant_documents(message)
        context = "\n\n".join([doc.page_content for doc in relevant_docs])
        final_prompt = format_rag_prompt(message, context)
    else:
        # Standard Chat Mode
        final_prompt = format_standard_prompt(message, history)

    output = llm(final_prompt, max_tokens=2048, stop=[EOT_TOKEN], echo=False, temperature=temperature)
    return output['choices'][0]['text'].strip()

# --- UI and App Launch remain the same ---
with gr.Blocks(theme="soft", title="🤖 Chatbot AI On-Premise") as demo:
    retriever_state = gr.State(None)

    gr.Markdown("# 🤖 Chatbot AI On-Premise with RAG")
    
    with gr.Tabs():
        with gr.TabItem("Knowledge Base"):
            gr.Markdown("Load a webpage to use as a knowledge base for the chatbot.")
            url_input = gr.Textbox(label="Webpage URL", placeholder="https://en.wikipedia.org/wiki/Artificial_intelligence")
            load_button = gr.Button("Load and Process Webpage")
            status_output = gr.Label(value="Status: Not loaded")
            
            # The click event now implicitly uses the gr.Progress object
            load_button.click(process_webpage_to_retriever, inputs=[url_input], outputs=[status_output, retriever_state])

        with gr.TabItem("Chatbot"):
            chatbot = gr.Chatbot(height=500, label="Chat")
            msg = gr.Textbox(show_label=False, placeholder="Ask a question about the loaded webpage or start a normal chat...", container=False)
            
            with gr.Accordion("Advanced Options", open=False):
                temp_slider = gr.Slider(minimum=0.0, maximum=2.0, value=0.7, step=0.1, label="Temperature")

            def predict(message, history, temperature, retriever):
                response = chatbot_response(message, history, temperature, retriever)
                history.append((message, response))
                return "", history

            msg.submit(predict, [msg, chatbot, temp_slider, retriever_state], [msg, chatbot])
            gr.ClearButton([msg, chatbot], value="New Conversation")

if __name__ == "__main__":
    print("Launching Gradio UI...")
    demo.launch(server_name="0.0.0.0")

