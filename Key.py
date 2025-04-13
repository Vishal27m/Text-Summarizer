import streamlit as st
from transformers import BartTokenizer, BartForConditionalGeneration
import PyPDF2
import docx2txt
import io
import textstat
import re
from googletrans import Translator

# Load model and tokenizer once
@st.cache_resource
def load_model():
    tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")
    model = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn")
    return tokenizer, model

tokenizer, model = load_model()
translator = Translator()

st.title("🌍 Multilingual AI Text Summarizer")

st.write("Upload a file or enter text to generate a summary using the BART model. Supports multiple languages!")

# Language selection
lang_dict = {
    "English": "en", "Hindi": "hi", "French": "fr", "Spanish": "es",
    "German": "de", "Chinese (Simplified)": "zh-cn", "Tamil": "ta", "Arabic": "ar"
}
selected_lang = st.selectbox("🌐 Select Language of Input Text", list(lang_dict.keys()))
selected_lang_code = lang_dict[selected_lang]

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
token_length = int(summary_length * 1.33)

# 3-line summary logic
three_line_mode = st.checkbox("📏 Generate a 3-line summary")

if st.button("🚀 Generate Summary"):
    if user_input.strip():
        with st.spinner("🔄 Translating & Summarizing..."):

            # Translate input to English
            input_translated = translator.translate(user_input.strip(), src=selected_lang_code, dest="en").text

            # Add tone instruction if selected
            tone_instruction = ""
            if tone != "Default":
                tone_instruction = f"Summarize in a {tone.lower()} tone: "
            full_input = tone_instruction + input_translated.replace("\n", " ")

            # Tokenize
            inputs = tokenizer([full_input], max_length=1024, truncation=True, return_tensors="pt")

            if three_line_mode:
                summary_ids = model.generate(inputs["input_ids"],
                                             max_length=60,
                                             min_length=30,
                                             num_beams=4,
                                             length_penalty=2.0,
                                             early_stopping=True)
            else:
                summary_ids = model.generate(inputs["input_ids"],
                                             max_length=token_length + 40,
                                             min_length=token_length,
                                             num_beams=4,
                                             length_penalty=2.0,
                                             early_stopping=True)

            summary_en = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

            # Keep first 3 sentences (optional)
            if three_line_mode:
                summary_sentences = re.split(r'(?<=[.!?]) +', summary_en)
                summary_en = ' '.join(summary_sentences[:3])

            # Highlight keywords
            if manual_keywords:
                keywords = [kw.strip() for kw in manual_keywords.split(',')]
                for kw in keywords:
                    summary_en = re.sub(f"(?i)({re.escape(kw)})", r"**\1**", summary_en)

            # Translate back to original language
            summary_translated = translator.translate(summary_en, src="en", dest=selected_lang_code).text

            st.success("✨ Summary:")
            st.markdown(summary_translated)

            # Download
            st.download_button("📥 Download Summary as .txt", summary_translated, file_name="summary.txt")

            # Stats
            original_len = len(user_input.split())
            summary_len = len(summary_translated.split())
            compression = round(100 * (1 - summary_len / original_len), 1) if original_len > 0 else 0
            st.markdown(f"📊 **Original Words:** {original_len} | **Summary Words:** {summary_len} | **Compression:** {compression}%")

            # Readability (only in English)
            translated_for_score = translator.translate(user_input, src=selected_lang_code, dest="en").text
            st.markdown("🧠 **Readability Analysis (English Equivalent)**")
            st.write(f"• Flesch Reading Ease: `{textstat.flesch_reading_ease(translated_for_score):.2f}`")
            st.write(f"• Flesch-Kincaid Grade: `{textstat.flesch_kincaid_grade(translated_for_score):.2f}`")
            st.write(f"• Gunning Fog Index: `{textstat.gunning_fog(translated_for_score):.2f}`")

    else:
        st.warning("Please enter or upload some text.")
