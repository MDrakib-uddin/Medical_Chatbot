import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Load model and tokenizer
model_name = "Qwen-Medical-Model"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    trust_remote_code=True,
    load_in_8bit=True,
    device_map='auto',
    torch_dtype=torch.bfloat16
)

# Chat function
def chat(instruction):
    prompt = f"You are a helpful assistant. {instruction}"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=200)
    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    response = decoded[len(prompt):].strip()
    return response

# Streamlit UI
st.set_page_config(page_title="Medical Chatbot")
st.title("Medical Chatbot")
st.markdown("Trained on alternative medicine and healthcare Q&A using LoRa fine-tuning.")

user_input = st.text_area("Ask Health related question", height=150)

if st.button("Get Answer"):
    if user_input.strip():
        with st.spinner("Generating answer..."):
            answer = chat(user_input)
            st.text_area("Answer", value=answer, height=150)
    else:
        st.warning("Please enter a question.")

