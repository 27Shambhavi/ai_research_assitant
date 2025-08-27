#Research Tool

## 🔍 Description

The **Universal Research Tool** is an innovative **Streamlit-based web application** designed to empower researchers, students, and professionals to interact seamlessly with any research paper. Users can **input a paper's content or abstract**, ask targeted questions, and receive **context-aware answers** in multiple styles such as **concise summaries, bullet points, detailed explanations, or simplified versions**. This tool leverages the power of **Hugging Face Transformers** and **semantic search** for accurate, insightful responses.

## ✨ Features

* 📝 **Flexible Input:** Paste any research paper text or abstract.
* 🎨 **Response Styles:** Choose how the answers are delivered — concise, bullet points, detailed, or simplified.
* ❓ **Interactive Q\&A:** Ask specific questions about the paper.
* 🔍 **Semantic Search:** Ensures answers are based on the most relevant sections of the paper.
* ⚡ **CPU-Friendly:** Optimized to avoid tensor errors, even on CPU.

## 🛠 Requirements

* Python 3.9+
* Streamlit
* Transformers
* Sentence-Transformers
* PyTorch (CPU or GPU)

## 🚀 Optional Improvements

* 📄 Add **PDF upload support** using `PyPDF2` or `pdfplumber`.
* 💬 Enable **conversation memory** to maintain multi-turn interactions.
* 🌐 Integrate **Hugging Face Inference API** for larger models.

## 📄 License

This project is licensed under the **MIT License**.

##Acknowledgements

* Hugging Face Transformers
* Sentence-Transformers
* Streamlit
