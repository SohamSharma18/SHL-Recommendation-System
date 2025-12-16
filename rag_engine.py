import os
import asyncio
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.docstore.document import Document
from langchain.text_splitter import CharacterTextSplitter
import json
from typing import List, Any
from langchain_community.document_loaders import JSONLoader

try:
    asyncio.get_running_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())

MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.1"
embeddings_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

async def load_catalogue(file_path="catalogue.json"):
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    documents = []
    for item in data:
        name = item.get("name", "")
        description = item.get("description", "")
        tags = " ".join(item.get("tags", []))
        content = f"{name} {description} {tags}"
        documents.append(Document(page_content=content))
    
    return documents

async def index_documents(documents: List[Document]) -> FAISS:
    splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = splitter.split_documents(documents)
    db = FAISS.from_documents(docs, embeddings)
    return db

async def load_llm():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        device_map="auto",
        torch_dtype="auto",
        low_cpu_mem_usage=True,
        offload_folder="offload",
        trust_remote_code=True
    )
    return pipeline("text-generation", model=model, tokenizer=tokenizer)

async def query_catalogue(query: str, db: FAISS, generator: Any):
    similar_docs = db.similarity_search(query, k=3)
    context = "\n---\n".join([doc.page_content for doc in similar_docs])
    prompt = f"You are an SHL product advisor. Based on the context below, recommend suitable assessments for: \n\n" \
             f"{query}\n\nContext:\n{context}\n\nAnswer:"

    response = await asyncio.to_thread(generator, prompt, max_new_tokens=512, do_sample=True, temperature=0.7)
    return response[0]["generated_text"].replace(prompt, "")

async def initialize_rag_engine():
    documents = await load_catalogue("shl_catalogue.json")
    db = await index_documents(documents)
    generator = await load_llm()
    return db, generator

async def get_recommendation(user_input: str):
    db, generator = await initialize_rag_engine()
    return await query_catalogue(user_input, db, generator)