
import os
import pickle
import faiss
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI
from sentence_transformers import SentenceTransformer


class RAGPipeline:
    def __init__(
        self,
        index_path="index.faiss",
        chunks_path="chunks.pkl",
        embedding_model="all-MiniLM-L6-v2",
        llm_model="gpt-4.1-mini"
    ):
        load_dotenv()

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found.")

        self.client = OpenAI(api_key=api_key)
        self.llm_model = llm_model

        # Load embedding model
        self.embedder = SentenceTransformer(embedding_model)

        # Load FAISS index
        if not os.path.exists(index_path):
            raise FileNotFoundError("FAISS index not found.")
        self.index = faiss.read_index(index_path)

        # Load chunks
        if not os.path.exists(chunks_path):
            raise FileNotFoundError("Chunks file not found.")
        with open(chunks_path, "rb") as f:
            self.chunks = pickle.load(f)

    def retrieve(self, query, k=3):
        query_embedding = self.embedder.encode(
            [query],
            normalize_embeddings=True
        )

        scores, indices = self.index.search(
            query_embedding.astype("float32"),
            k
        )

        results = []
        for idx, score in zip(indices[0], scores[0]):
            metadata = self.chunks[idx].metadata

            results.append({
                "score": float(score),
                "source": metadata.get("source"),
                "page": metadata.get("page_label"),
                "content": self.chunks[idx].page_content
            })

        return results

    def generate_answer(self, query, k=3):
        retrieved_docs = self.retrieve(query, k=k)

        if not retrieved_docs:
            return {
                "answer": "No relevant documents were retrieved.",
                "sources": []
            }

        context = "\n\n---\n\n".join(
            doc["content"] for doc in retrieved_docs
        )

        prompt = f"""
You are a legal assistant answering strictly from the context.

Rules:
- Only use provided context.
- If answer not present, say:
  "The guide does not provide this information."
- Do not invent laws or numbers.

Context:
{context}

Question:
{query}

Answer:
"""

        response = self.client.chat.completions.create(
            model=self.llm_model,
            messages=[
                {"role": "system", "content": "You are a precise legal assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=0,
            max_tokens=500
        )

        answer = response.choices[0].message.content.strip()

        if "The guide does not provide this information." in answer:
            return {
                "answer": answer,
                "sources": []
            }

        sources = list(set(
            f"{doc['source']} (Page {doc['page']})"
            for doc in retrieved_docs
        ))

        return {
            "answer": answer,
            "sources": sources
        }
