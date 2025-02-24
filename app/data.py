import numpy as np
from sentence_transformers import SentenceTransformer
import torch
import faiss  # We'll use this in Step 3

# For demonstration, we still have our small FAQ examples.
faqs = [
    {"question": "Where is my order?", "answer": "Your order is on the way. It will arrive within 3-5 business days."},
    {"question": "How do I return an item?", "answer": "To return an item, visit our returns page and follow the instructions."},
    {"question": "Do you offer refunds?", "answer": "Yes, refunds are available within 30 days of purchase."}
]

# Load large historical data from PDFs (simulate this by reading from our PDFs)
from app.pdf_loader import load_historical_data
pdf_corpus_text = load_historical_data()  # Returns a long string
pdf_documents = [doc.strip() for doc in pdf_corpus_text.split("\n") if doc.strip()]

# Combine FAQ questions and PDF documents for the retrieval corpus
retrieval_corpus = [faq["question"] for faq in faqs] + pdf_documents

# Initialize a pre-trained SentenceTransformer for embeddings
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Compute embeddings for the entire corpus
# We convert them to numpy array for FAISS (Step 3)
corpus_embeddings = embedding_model.encode(retrieval_corpus, convert_to_tensor=False)
corpus_embeddings = np.array(corpus_embeddings).astype("float32")

# Build a FAISS index (see Step 3 for details)
d = corpus_embeddings.shape[1]  # dimension of embeddings
faiss_index = faiss.IndexFlatL2(d)
faiss_index.add(corpus_embeddings)

def prepare_input(query: str, d_model: int = 64):
    """
    Given a query, compute its embedding and use FAISS to retrieve the most similar context.
    Also return a simulated accuracy measure.
    """
    # Compute the embedding for the query using SentenceTransformer
    query_embedding = embedding_model.encode([query], convert_to_tensor=False)
    query_embedding = np.array(query_embedding).astype("float32")
    
    # Use FAISS to find the closest match
    k = 1
    D, I = faiss_index.search(query_embedding, k)
    top_idx = int(I[0][0])
    retrieved_text = retrieval_corpus[top_idx]
    
    # Simulated accuracy: based on cosine similarity (we can compute it here)
    # For simplicity, we normalize and compute dot product as cosine similarity.
    query_norm = query_embedding / np.linalg.norm(query_embedding)
    doc_norm = corpus_embeddings[top_idx] / np.linalg.norm(corpus_embeddings[top_idx])
    cosine_sim = float(np.dot(query_norm, doc_norm))
    
    # Prepare a dummy tensor input for the model if needed (here just for compatibility)
    # In this step, the generation model uses text, so we don't need tensor inputs.
    dummy_tensor = torch.tensor([[0.0]])  # Not used by generation, kept for interface consistency.
    
    return dummy_tensor, retrieved_text, cosine_sim
