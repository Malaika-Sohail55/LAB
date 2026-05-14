import os
import zipfile
import gdown
import torch
import streamlit as st
from transformers import MarianMTModel, MarianTokenizer

st.set_page_config(page_title="English → Urdu Translator", page_icon="🌐")

MODEL_DIR = "./en_ur_translation_model"
GDRIVE_ID = "1Zvs35RIOA7RrVEwQFiShYar6UvmhXC51"

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_DIR):
        with st.spinner("⏳ Downloading model... (first run only, ~545MB)"):
            gdown.download(f"https://drive.google.com/uc?id={GDRIVE_ID}", "model.zip", quiet=False)
            with zipfile.ZipFile("model.zip", "r") as z:
                z.extractall(".")
            os.remove("model.zip")
    model = MarianMTModel.from_pretrained(MODEL_DIR)
    tokenizer = MarianTokenizer.from_pretrained(MODEL_DIR)
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
            f"<p style='font-size:26px; direction:rtl; text-align:right; font-family:Noto Nastaliq Urdu;'>{result}</p>",
            unsafe_allow_html=True
        )
    else:
        st.warning("⚠️ Please enter some text first.")