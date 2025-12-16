import json
import faiss
import numpy as np
import re
from langchain.docstore.document import Document
from langchain_huggingface import HuggingFaceEmbeddings
from nltk.corpus import stopwords
import nltk

nltk.download('stopwords')

embeddings_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

def load_catalogue(file_path="shl_catalogue.json"):
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    documents = []
    for item in data:
        name = item.get("Assessment Name", "Unnamed Assessment")
        description = item.get("Description", "No description available.")
        job_levels = item.get("Job Levels", "General")
        test_type = item.get("Test Type", "Unknown")
        length = item.get("Assessment Length", "Not specified")
        remote = item.get("Remote Testing", "No")
        adaptive = item.get("Adaptive / IRT", "No")
        url = item.get("URL", "")

        full_text = f"""
        Assessment: {name}
        Description: {description}
        Suitable For: {job_levels}
        Test Type: {test_type}
        Duration: {length}
        Remote Testing: {remote}
        Adaptive/IRT: {adaptive}
        URL: {url}
        """

        metadata = {
            "name": name,
            "description": description,
            "job_levels": job_levels,
            "test_type": test_type,
            "duration": length,
            "remote": remote,
            "adaptive": adaptive,
            "url": url,
        }

        documents.append(Document(page_content=full_text, metadata=metadata))

    return documents

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z\s]", "", text)
    stop_words = set(stopwords.words("english"))
    return " ".join(word for word in text.split() if word not in stop_words)

def generate_embeddings(documents):
    preprocessed = [preprocess_text(doc.page_content) for doc in documents]
    return embeddings_model.embed_documents(preprocessed)

def create_faiss_index(embeddings):
    embeddings = np.array(embeddings).astype("float32")
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    return index

def get_query_embedding(query):
    return embeddings_model.embed_query(query)

def search_similar_assessments(query_embedding, faiss_index, k=5):
    query_embedding = np.array(query_embedding).reshape(1, -1).astype("float32")
    distances, indices = faiss_index.search(query_embedding, k)
    return distances, indices

def get_recommendations(user_input, faiss_index, documents, k=5):
    query = preprocess_text(user_input)
    relevant_documents = []
    for doc in documents:
        if any(keyword in query for keyword in ['spanish', 'language proficiency', 'beginner', 'foreign language']):
            if 'spanish' in doc.page_content.lower():
                relevant_documents.append(doc)
    if not relevant_documents:
        query_embedding = get_query_embedding(user_input)
        distances, indices = search_similar_assessments(query_embedding, faiss_index, k)
        relevant_documents = [documents[idx] for idx in indices[0]]
    return relevant_documents[:k]

def filter_documents_by_keywords(documents, keywords):
    filtered_docs = []
    for doc in documents:
        content = doc.page_content.lower()
        if any(keyword in content for keyword in keywords):
            filtered_docs.append(doc)
    return filtered_docs