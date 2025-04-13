📝 BART-based Text Summarizer

A simple AI utility that generates a 3-line summary from any large block of text using the powerful facebook/bart-large-cnn model from Hugging Face.

🔍 About the Project
This project is developed as part of the AI & ML Internship at Senthuron, aimed at showcasing the use of modern NLP models to enhance productivity through micro-tools. The summarizer is built using Python and Hugging Face Transformers.

🚀 How It Works
The user inputs a long piece of text.

The text is tokenized using BartTokenizer.

It is passed to the BartForConditionalGeneration model.

The model returns a short, readable summary.

🧠 Model Used
Model: facebook/bart-large-cnn

Library: Hugging Face Transformers

Framework: PyTorch

✨ Features
Summarizes long content into a 3-line output.

Easy-to-use script with Streamlit for UI (optional).

Can be extended into educational or client-facing tools.

📈 Scope for Improvement
Add multi-language summarization support.

Let users select summary length (short/medium/long).

Add keyword extraction or topic detection.

Deploy as a web app with Gradio or Streamlit Cloud.

🧪 Usage
bash
Copy
Edit
# Install dependencies
pip install transformers torch streamlit

# Run the app (if using Streamlit UI)
streamlit run BART.py
Or run the script directly in a notebook (Colab or local).

📁 Deliverables
.ipynb or .py file with summarization logic.

A write-up explaining the model and improvement scope.

📌 Example
Input:
"Natural Language Processing (NLP) is a field of Artificial Intelligence focused on the interaction between computers and humans through natural language..."

Output:
"Natural Language Processing (NLP) enables computers to understand and respond to human language.
It is a core field in AI with applications in chatbots, translation, and summarization.
This tool provides a quick 3-line summary from any input text."
