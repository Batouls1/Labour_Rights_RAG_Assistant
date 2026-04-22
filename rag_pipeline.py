import os
import re
import pickle
import logging
import faiss
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI
from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi


# Logging Setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

# Configuration
CONFIG = {
    "top_k_retrieval": 10,
    "top_k_final": 3,
    "candidate_multiplier": 5,
    "use_reranker": True,
    "dense_weight": {"en": 0.7, "ar": 0.5},
    "sparse_weight": {"en": 0.3, "ar": 0.5},
}

class RAGPipeline:
    def __init__(
        self,
        index_path="index.faiss",
        chunks_path="chunks.pkl",
        embed_model="intfloat/multilingual-e5-base",
        rerank_model="cross-encoder/mmarco-mMiniLMv2-L12-H384-v1",
        llm_model="gpt-4.1-mini"
    ):
        load_dotenv()

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in .env file.")

        self.client = OpenAI(api_key=api_key)
        self.llm_model = llm_model

        logger.info("Loading embedding model: %s", embed_model)
        self.embedder = SentenceTransformer(embed_model)

        logger.info("Loading reranker model: %s", rerank_model)
        self.reranker = CrossEncoder(rerank_model)

        # Load FAISS index
        if not os.path.exists(index_path):
            raise FileNotFoundError(f"FAISS index not found at: {index_path}")
        self.index = faiss.read_index(index_path)
        logger.info("FAISS index loaded: %d vectors", self.index.ntotal)

        # Load chunks
        if not os.path.exists(chunks_path):
            raise FileNotFoundError(f"Chunks file not found at: {chunks_path}")
        with open(chunks_path, "rb") as f:
            self.chunks = pickle.load(f)
        logger.info("Chunks loaded: %d total", len(self.chunks))

        # Build BM25 index from loaded chunks
        logger.info("Building BM25 index...")
        tokenized_corpus = [
            self._tokenize(chunk.page_content, lang=chunk.metadata.get("language", "en"))
            for chunk in self.chunks
        ]
        self.bm25 = BM25Okapi(tokenized_corpus)
        logger.info("BM25 index built.")

    # Langauge Detection
    def _detect_language(self, text: str) -> str:
        arabic_chars = sum(1 for c in text if "\u0600" <= c <= "\u06FF")
        latin_chars = sum(1 for c in text if c.isascii() and c.isalpha())
        return "ar" if arabic_chars > latin_chars else "en"
    
    # BM25 Tokenizer (language-aware)
    def _tokenize(self, text: str, lang: str = "en") -> list:
        if lang == "ar":
            text = re.sub(r"[^\w\s\u0600-\u06FF]", " ", text)
            return text.strip().split()
        return text.lower().split()
    
    # Embedding Helper - applies E5 prefix
    def _embed(self, texts: list, prefix: str = "passage") -> np.ndarray:
        prefixed = [f"{prefix}: {t}" for t in texts]
        embeddings = self.embedder.encode(
            prefixed,
            normalize_embeddings=True,
            show_progress_bar=False
        )
        return np.array(embeddings).astype("float32")

    # Retrieval - hybrid dense + sparse with language filter
    def retrieve(self, query: str, k: int = 5, candidate_multiplier: int = 5) -> list:
        query_lang = self._detect_language(query)
        logger.info("Retrieving — query_lang=%s, k=%d", query_lang, k)

        # Dense retrieval
        query_embedding = self._embed([query], prefix="query")
        dense_scores, dense_ids = self.index.search(query_embedding, k * candidate_multiplier)

        dense_map = {
            int(idx): float(score)
            for idx, score in zip(dense_ids[0], dense_scores[0])
            if idx != -1
        }

        # Sparse retrieval
        tokenized_query = self._tokenize(query, lang=query_lang)
        sparse_scores = self.bm25.get_scores(tokenized_query)

        results = []
        for i, chunk in enumerate(self.chunks):
            chunk_lang = chunk.metadata.get("language", "en")

            if chunk_lang != query_lang:
                continue

            dense_score = dense_map.get(i, 0.0)
            sparse_score = float(sparse_scores[i])

            combined_score = (
                CONFIG["dense_weight"][query_lang] * dense_score +
                CONFIG["sparse_weight"][query_lang] * sparse_score
            )

            results.append({
                "score": combined_score,
                "source": os.path.basename(chunk.metadata.get("source", "")),
                "page": chunk.metadata.get("page_label"),
                "content": chunk.page_content,
                "language": chunk_lang
            })

        results.sort(key=lambda x: x["score"], reverse=True)
        retrieved = results[:k * 2]
        logger.info("Retrieved %d candidates after language filter", len(retrieved))
        return retrieved

    # Reranking
    def rerank(self, query: str, candidates: list, top_n: int = 3) -> list:
        if not candidates:
            return []

        pairs = [[query, doc["content"]] for doc in candidates]
        scores = self.reranker.predict(pairs)

        for doc, score in zip(candidates, scores):
            doc["rerank_score"] = float(score)

        reranked = sorted(candidates, key=lambda x: x["rerank_score"], reverse=True)
        logger.info("Reranked — returning top %d chunks", top_n)
        return reranked[:top_n]

    # Prompt Builder - language-aware
    def _build_prompt(self, query: str, retrieved_docs: list, query_lang: str) -> tuple:
        context = "\n\n".join(
            f"[Source {i+1}]\n{doc['content']}"
            for i, doc in enumerate(retrieved_docs)
        )

        if query_lang == "ar":
            system_msg = "أنت مساعد قانوني دقيق. أجب فقط باللغة العربية ولا تستخدم أي لغة أخرى."
            refusal_phrase = "الدليل لا يوفر هذه المعلومات"
            user_prompt = f"""أنت مساعد قانوني تجيب على الأسئلة بناءً على السياق فقط.

القواعد:
- استخدم فقط المعلومات الموجودة في السياق.
- إذا لم تجد الإجابة، قل: "{refusal_phrase}"
- لا تخترع معلومات.

السياق:
{context}

السؤال:
{query}

الجواب:"""
        else:
            system_msg = "You are a precise legal assistant. Answer ONLY in English."
            refusal_phrase = "The guide does not provide this information"
            user_prompt = f"""You are a legal assistant answering strictly from context.

Rules:
- Use only the provided context.
- If missing, say: "{refusal_phrase}"
- Do not hallucinate.

Context:
{context}

Question:
{query}

Answer:"""

        return system_msg, user_prompt, refusal_phrase
    
    # LLM Call
    def _call_llm(self, system_msg: str, user_prompt: str) -> str:
        response = self.client.chat.completions.create(
            model=self.llm_model,
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0,
            max_tokens=500
        )
        return response.choices[0].message.content.strip()
    
    # Full Pipeline
    def generate_answer(self, query: str) -> dict:
        if not query or not query.strip():
            return {"answer": "Please provide a valid question.", "sources": []}
        
        logger.info("generate_answer called — query: %s", query[:80])
        query_lang = self._detect_language(query)

        candidates = self.retrieve(
            query,
            k=CONFIG["top_k_retrieval"],
            candidate_multiplier=CONFIG["candidate_multiplier"]
        )

        top_n = 5 if query_lang == "ar" else CONFIG["top_k_final"]

        if CONFIG["use_reranker"]:
            retrieved_docs = self.rerank(query, candidates, top_n=top_n)
        else:
            retrieved_docs = candidates[:top_n]

        if not retrieved_docs:
            msg = "لم يتم العثور على وثائق ذات صلة." if query_lang == "ar" else "No relevant documents were retrieved."
            logger.warning("No documents retrieved for query: %s", query[:80])
            return {"answer": msg, "sources": []}

        system_msg, user_prompt, refusal_phrase = self._build_prompt(
            query, retrieved_docs, query_lang
        )
        answer = self._call_llm(system_msg, user_prompt)

        if refusal_phrase in answer:
            logger.info("Refusal triggered for query: %s", query[:80])
            return {"answer": answer, "sources": []}

        seen = set()
        sources = []
        for doc in retrieved_docs:
            key = f"{doc['source']} (Page {doc['page']})"
            if key not in seen:
                seen.add(key)
                sources.append(key)

        logger.info("Answer generated — %d sources cited", len(sources))
        return {"answer": answer, "sources": sources}
