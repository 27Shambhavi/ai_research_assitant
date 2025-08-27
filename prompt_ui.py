import streamlit as st
from transformers import pipeline
from sentence_transformers import SentenceTransformer, util
import torch

st.set_page_config(page_title="Universal Research Tool", page_icon="🔍")
st.header("Research Tool")

# Inputs
paper_text = st.text_area("Paste the paper text or abstract here:")
style_options = ["Concise summary", "Detailed explanation", "Bullet points", "Simplified version"]
style_choice = st.selectbox("Select the response style:", style_options)
user_query = st.text_area("Enter your research question here:")

# Initialize Hugging Face model
generator = pipeline(
    "text2text-generation",
    model="google/flan-t5-small",  # smaller CPU-friendly model
    max_new_tokens=200,
    device=-1
)

# Initialize embedding model for semantic search
embedder = SentenceTransformer('all-MiniLM-L6-v2')

def chunk_text(text, max_len=500):
    words = text.split()
    chunks = []
    for i in range(0, len(words), max_len):
        chunks.append(" ".join(words[i:i+max_len]))
    return chunks

if st.button("Submit"):
    if not paper_text.strip() or not user_query.strip():
        st.warning("Please enter both paper content and your query.")
    else:
        try:
            # Chunk paper
            chunks = chunk_text(paper_text)

            # Embed chunks and query
            chunk_embeddings = embedder.encode(chunks, convert_to_tensor=True)
            query_embedding = embedder.encode(user_query, convert_to_tensor=True)

            # Semantic search: find most relevant chunk
            scores = util.pytorch_cos_sim(query_embedding, chunk_embeddings)
            best_idx = torch.argmax(scores)
            relevant_chunk = chunks[best_idx]

            # Build prompt for model
            prompt = (
                f"Paper content: {relevant_chunk}\n"
                f"Style: {style_choice}\n"
                f"Question: {user_query}\n"
                f"Answer the question according to the paper and style."
            )

            response = generator(prompt)
            st.success("Response generated successfully!")
            st.write(response[0]['generated_text'])

        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
