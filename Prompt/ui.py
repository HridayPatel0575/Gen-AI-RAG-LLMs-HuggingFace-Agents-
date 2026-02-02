import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain_huggingface import HuggingFacePipeline, ChatHuggingFace

MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

@st.cache_resource
def load_llm():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,   
        device_map="cuda"            
    )

    text_gen_pipeline = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=300,
        temperature=0.7,
        top_p=0.9,
        do_sample=True,
    )

    lc_llm = HuggingFacePipeline(pipeline=text_gen_pipeline)

    return ChatHuggingFace(llm=lc_llm)

model = load_llm()

st.header("Research Tool (GPU)")

user_input = st.text_area("Enter text", height=200)

if st.button("Summarize"):
    if user_input.strip():
        with st.spinner("Running on GPU"):
            prompt = f"Summarize the following text:\n\n{user_input}"
            result = model.invoke(prompt)
            st.write(result.content)
    else:
        st.warning("Please enter text")
