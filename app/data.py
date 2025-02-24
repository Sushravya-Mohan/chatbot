import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from app.pdf_loader import load_historical_data

# Simulated small FAQ examples (these can be augmented by PDF content)
faqs = [
    {"question": "Where is my order?", "answer": "Your order is on the way. It will arrive within 3-5 business days."},
    {"question": "How do I return an item?", "answer": "To return an item, visit our returns page and follow the instructions."},
    {"question": "Do you offer refunds?", "answer": "Yes, refunds are available within 30 days of purchase."}
]

# Load large historical data from PDFs (simulate large text data)
pdf_corpus_text = load_historical_data()  # This can be a long string containing historical FAQs

# Split the PDF text into pseudo-documents (e.g., by line or delimiter)
pdf_documents = pdf_corpus_text.split("\n")
# Combine with the FAQ questions for a more extensive retrieval corpus
retrieval_corpus = [faq["question"] for faq in faqs] + pdf_documents

# Fit TF-IDF on the extended corpus
vectorizer = TfidfVectorizer(stop_words="english").fit(retrieval_corpus)
retrieval_embeddings = vectorizer.transform(retrieval_corpus)

def text_to_embedding(text: str, d_model: int = 64) -> np.ndarray:
    """Generate a synthetic embedding for a token. Replace with a real model in production."""
    np.random.seed(abs(hash(text)) % (2**32))
    return np.random.rand(d_model)

def prepare_input(query: str, d_model: int = 64):
    """Retrieve the most relevant context from the large corpus and prepare the input tensor."""
    from sklearn.metrics.pairwise import cosine_similarity
    import torch

    query_vec = vectorizer.transform([query])
    similarities = cosine_similarity(query_vec, retrieval_embeddings).flatten()
    top_idx = int(np.argmax(similarities))
    retrieved_text = retrieval_corpus[top_idx]
    
    # Simulated accuracy measure: if the query and retrieved text share common words, assume 'accurate'
    accuracy = len(set(query.lower().split()) & set(retrieved_text.lower().split())) / len(query.split())
    
    # Combine query with retrieved context
    combined_text = query + " " + retrieved_text
    tokens = combined_text.split()
    embeddings = [text_to_embedding(token, d_model) for token in tokens]
    embeddings_tensor = torch.tensor(embeddings, dtype=torch.float32).unsqueeze(0)
    return embeddings_tensor, retrieved_text, accuracy
