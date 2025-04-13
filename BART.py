import streamlit as st
from transformers import BartTokenizer, BartForConditionalGeneration
import PyPDF2
import docx2txt
import io
import textstat
import re

# Load model and tokenizer once
@st.cache_resource
def load_model():
    tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")
    model = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn")
    return tokenizer, model

tokenizer, model = load_model()

st.title("📘 Advanced AI Text Summarizer")

st.write("Upload a file or enter text to generate a smart, customizable summary using the BART model.")

# File uploader
uploaded_file = st.file_uploader("📄 Upload a .txt, .pdf, or .docx file", type=["txt", "pdf", "docx"])
file_text = ""

if uploaded_file is not None:
    if uploaded_file.type == "text/plain":
        file_text = uploaded_file.read().decode("utf-8")
    elif uploaded_file.type == "application/pdf":
        reader = PyPDF2.PdfReader(uploaded_file)
        for page in reader.pages:
            file_text += page.extract_text()
    elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        file_text = docx2txt.process(uploaded_file)

# Text input area
user_input = st.text_area("✍️ Or type/paste your text here:", value=file_text, height=250)

# Tone selection
tone = st.selectbox("🎨 Choose Summary Tone/Style", ["Default", "Formal", "Informal", "Academic", "Concise"])

# Manual keyword emphasis
manual_keywords = st.text_input("🔍 Enter keywords/phrases to highlight (comma-separated):")

# Summary length
summary_length = st.slider("📏 Desired summary length (in words)", 30, 200, 60)
token_length = int(summary_length * 1.33)  # Adjust as per model's token-to-word ratio

# 3-line summary logic
three_line_mode = st.checkbox("📏 Generate a 3-line summary")

if st.button("🚀 Generate Summary"):
    if user_input.strip():
        with st.spinner("Generating summary..."):

            input_text = user_input.strip().replace("\n", " ")

            # Apply tone instruction if selected
            tone_instruction = ""
            if tone != "Default":
                tone_instruction = f"Summarize in a {tone.lower()} tone: "
            full_input = tone_instruction + input_text

            # Tokenization
            inputs = tokenizer([full_input], max_length=1024, truncation=True, return_tensors="pt")

            if three_line_mode:
                # Generate summary for 3-line mode
                summary_ids = model.generate(inputs["input_ids"],
                                             max_length=60,
                                             min_length=30,
                                             num_beams=4,
                                             length_penalty=2.0,
                                             early_stopping=True)
                summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
                
                # Split summary into sentences
                summary_sentences = re.split(r'(?<=[.!?]) +', summary)
                
                # Get the first 3 sentences (or less if the summary is shorter)
                summary = ' '.join(summary_sentences[:3])
            else:
                # Standard summary generation
                summary_ids = model.generate(inputs["input_ids"],
                                             max_length=token_length + 40,
                                             min_length=token_length,
                                             num_beams=4,
                                             length_penalty=2.0,
                                             early_stopping=True)
                summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

            # Highlight keywords
            if manual_keywords:
                keywords = [kw.strip() for kw in manual_keywords.split(',')]
                for kw in keywords:
                    summary = re.sub(f"(?i)({re.escape(kw)})", r"**\1**", summary)

            # Display summary
            st.success("✨ Summary:")
            st.markdown(summary)

            # Download
            st.download_button("📥 Download Summary as .txt", summary, file_name="summary.txt")

            # Word stats
            original_len = len(user_input.split())
            summary_len = len(summary.split())
            compression = round(100 * (1 - summary_len / original_len), 1) if original_len > 0 else 0
            st.markdown(f"📊 **Original Words:** {original_len} | **Summary Words:** {summary_len} | **Compression:** {compression}%")

            # Readability
            st.markdown("🧠 **Readability Analysis**")
            st.write(f"• Flesch Reading Ease: `{textstat.flesch_reading_ease(user_input):.2f}`")
            st.write(f"• Flesch-Kincaid Grade: `{textstat.flesch_kincaid_grade(user_input):.2f}`")
            st.write(f"• Gunning Fog Index: `{textstat.gunning_fog(user_input):.2f}`")
    else:
        st.warning("Please enter or upload some text.")
