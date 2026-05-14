import os
import zipfile
import gdown
import torch
import streamlit as st
from transformers import MarianMTModel, MarianTokenizer

st.set_page_config(page_title="English → Urdu Translator", page_icon="🌐")

GDRIVE_ID = "1Zvs35RIOA7RrVEwQFiShYar6UvmhXC51"
ZIP_PATH  = "model.zip"

@st.cache_resource
def load_model():
    # Step 1: Download zip if not already there
    if not os.path.exists(ZIP_PATH):
        with st.spinner("⏳ Downloading model... (~545MB, first run only)"):
            gdown.download(f"https://drive.google.com/uc?id={GDRIVE_ID}", ZIP_PATH, quiet=False)

    # Step 2: Extract and auto-detect the model folder
    with st.spinner("📦 Extracting model..."):
        with zipfile.ZipFile(ZIP_PATH, "r") as z:
            z.extractall(".")
            all_names = z.namelist()

    # Step 3: Find the folder that contains config.json
    model_dir = None
    for name in all_names:
        if "config.json" in name:
            model_dir = os.path.dirname(name)
            break

    # If config.json is in root of zip, use current directory
    if model_dir == "" or model_dir is None:
        model_dir = "."

    st.write(f"📁 Model found at: `{model_dir}`")  # helpful debug line

    model     = MarianMTModel.from_pretrained(model_dir)
    tokenizer = MarianTokenizer.from_pretrained(model_dir)
    return model, tokenizer

model, tokenizer = load_model()

st.title("🌐 English → اردو Translator")
st.markdown("Fine-tuned MarianMT · Helsinki-NLP/opus-mt-en-ur")

input_text = st.text_area("Enter English text:", height=150, placeholder="Type something in English...")

if st.button("Translate 🔁"):
    if input_text.strip():
        with st.spinner("Translating..."):
            encoded = tokenizer(
                input_text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=128,
            )
            with torch.no_grad():
                output = model.generate(
                    **encoded,
                    max_new_tokens=128,
                    num_beams=4,
                    early_stopping=True,
                )
            result = tokenizer.decode(output[0], skip_special_tokens=True)

        st.subheader("Urdu Translation:")
        st.markdown(
            f"<p style='font-size:26px; direction:rtl; text-align:right;'>{result}</p>",
            unsafe_allow_html=True
        )
    else:
        st.warning("⚠️ Please enter some text first.")
