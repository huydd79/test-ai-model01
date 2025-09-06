import gradio as gr
from llama_cpp import Llama
from huggingface_hub import hf_hub_download

# --- Constants for Llama 3 prompt construction ---
BOS_TOKEN = "<|begin_of_text|>"
EOT_TOKEN = "<|eot_id|>"
SYS_HEADER = "<|start_header_id|>system<|end_header_id|>\n\n"
USER_HEADER = "<|start_header_id|>user<|end_header_id|>\n\n"
ASSISTANT_HEADER = "<|start_header_id|>assistant<|end_header_id|>\n\n"

# --- Step 1: Download model from Hugging Face Hub ---
model_name = "MaziyarPanahi/Meta-Llama-3-8B-Instruct-GGUF"
# NOTE: Using a higher quality model for better accuracy
model_file ="Meta-Llama-3-8B-Instruct.Q4_K_S.gguf" 

print("Downloading model (or getting from cache)...")
model_path = hf_hub_download(
    repo_id=model_name,
    filename=model_file
)
print("Model path ready!")

# --- Step 2: Initialize LLM running on CPU ---
print("Initializing LLM...")
llm = Llama(
    model_path=model_path,
    n_ctx=4096,
    n_threads=4,  # Adjust to your CPU
    n_gpu_layers=0
)
print("LLM Initialized!")

# --- Step 3: Define processing function for chatbot ---
def chatbot_response(message, history, temperature):
    # Start the prompt with a system message and the beginning of text token
    prompt_parts = [
        BOS_TOKEN,
        SYS_HEADER,
        "You are a helpful, smart, kind, and efficient AI assistant.",
        EOT_TOKEN,
    ]

    # Loop through the history to reconstruct the conversation
    for user_turn, bot_turn in history:
        prompt_parts.append(f"{USER_HEADER}{user_turn}{EOT_TOKEN}")
        prompt_parts.append(f"{ASSISTANT_HEADER}{bot_turn}{EOT_TOKEN}")

    # Add the latest user message
    prompt_parts.append(f"{USER_HEADER}{message}{EOT_TOKEN}")

    # Add the token to signal the model to start responding
    prompt_parts.append(ASSISTANT_HEADER)
    
    # Join all parts into a single complete prompt
    final_prompt = "".join(prompt_parts)

    # Create response from LLM
    output = llm(
        final_prompt,
        max_tokens=2048,
        stop=[EOT_TOKEN], # Stop when the model generates the end token
        echo=False,
        temperature=temperature # Use temperature from the slider
    )

    response_text = output['choices'][0]['text'].strip()
    return response_text

# --- Step 4: Create UI with Gradio, including a temperature slider ---
with gr.Blocks(theme="soft", title="🤖 Chatbot AI On-Premise") as demo:
    gr.Markdown(
        """
        # 🤖 Chatbot AI On-Premise
        *A simple chatbot that runs 100% on your machine, no GPU required.*
        """
    )
    
    chatbot = gr.Chatbot(height=500)
    
    with gr.Row():
        msg = gr.Textbox(
            show_label=False,
            placeholder="Enter your question...",
            container=False,
            scale=7
        )
    
    with gr.Row():
        submit_btn = gr.Button("Submit")
        clear_btn = gr.ClearButton([msg, chatbot], value="New Conversation")

    with gr.Accordion("Advanced Options", open=False):
        temp_slider = gr.Slider(
            minimum=0.0,
            maximum=2.0,
            value=0.7,
            step=0.1,
            label="Temperature",
            info="Higher values make the output more random and creative. Lower values make it more deterministic."
        )

    # Define the logic for handling chat interactions
    def predict(message, history, temperature):
        response = chatbot_response(message, history, temperature)
        # Append the new interaction to the history
        history.append((message, response))
        # Clear the input box and return the updated history
        return "", history

    # Set up event listeners for the submit button and textbox
    submit_btn.click(predict, [msg, chatbot, temp_slider], [msg, chatbot])
    msg.submit(predict, [msg, chatbot, temp_slider], [msg, chatbot])

# --- Step 5: Start app ---
if __name__ == "__main__":
    print("Launching Gradio UI...")
    demo.launch(server_name="0.0.0.0")


