Research Tool
Description:- 

The Universal Research Tool is a Streamlit-based web application designed to help researchers, students, and professionals interact with any research paper. Users can input a paper's content (or abstract), ask questions, and receive answers in different styles such as concise summaries, bullet points, detailed explanations, or simplified versions. The application leverages Hugging Face Transformers and semantic search for accurate, context-aware responses.

Features

Input any research paper text or abstract.

Choose response style (concise, bullet points, detailed, simplified).

Ask questions related to the paper.

Semantic search ensures relevant sections are used for answering.

CPU-friendly model usage to prevent tensor errors.
Requirements

Python 3.9+

Streamlit

Transformers

Sentence-Transformers

PyTorch (CPU or GPU)

Optional Improvements

Add PDF upload support using PyPDF2 or pdfplumber.

Enable conversation memory to maintain multi-turn interactions.

Integrate Hugging Face Inference API for larger models.

License

MIT License

Acknowledgements

Hugging Face Transformers

Sentence-Transformers

Streamlit
