---

# 📝 BART-based Text Summarizer Multilingual

This is a simple AI utility that takes a long piece of text as input and returns a **3-line summary** using the `facebook/bart-large-cnn` model from Hugging Face Transformers.

---

## 📌 Project Overview

Built as part of the **AI & ML Internship at Senthuron**, this summarizer tool demonstrates how state-of-the-art transformer models can be used to enhance internal workflows and educational platforms by condensing large blocks of text into concise summaries.

---

## 🚀 How It Works

1. User inputs a block of text.
2. Text is tokenized using `BartTokenizer`.
3. Tokens are passed to the `BartForConditionalGeneration` model.
4. The model returns a 3-line summary.

---

## 🧠 Model & Tools Used

- **Model:** `facebook/bart-large-cnn`
- **Library:** Hugging Face Transformers
- **Framework:** PyTorch
- **Optional UI:** Streamlit

---

## 🧪 Example

**Input:**  
"Natural Language Processing (NLP) is a field of Artificial Intelligence that enables computers to understand, interpret, and generate human language..."

**Output:**  
- NLP enables machines to understand and interact using human language.  
- It powers tools like chatbots, translators, and summarizers.  
- This utility summarizes long texts into concise, readable insights.

---

## 💻 Setup Instructions

1. **Install dependencies:**

```bash
pip install transformers torch streamlit
```

2. **Run the script (CLI/Notebook):**  
- For Python script:  
  ```bash
  python Key.py
  ```
- For Streamlit UI (optional):  
  ```bash
  streamlit run Key.py
  ```

---



