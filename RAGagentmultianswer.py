import json
import os
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from openai import OpenAI
import chromadb


class JSONRAGChroma:
    def __init__(
        self,
        kb_file: str,
        answer_fields: list[str] | None = None,
        mode: str = "single",  # "single", "concat", "multi"
        model_name: str = "multi-qa-mpnet-base-dot-v1",
        top_k: int = 3,
        urls: list[tuple[str, str]] | None = None,
        use_llm: bool = False,
        llm_model: str = "gpt-4o-mini",
        collection_name: str = "rag_collection",
        persist_path: str = "./multianswer"
    ):
        if answer_fields is None:
            answer_fields = ["answer_chatgpt"]

        # Load KB from JSON
        with open(kb_file, "r", encoding="utf-8") as f:
            raw = json.load(f)

        records, sources = [], []

        for row in raw:
            q = row.get("input", "").strip()
            if not q:
                continue

            if mode == "single":
                a = row.get(answer_fields[0], "").strip()
                if a:
                    records.append(f"Q: {q}\nA: {a}")
                    sources.append("json")

            elif mode == "concat":
                answers = [row.get(f, "").strip() for f in answer_fields if row.get(f)]
                if answers:
                    a = " | ".join(answers)
                    records.append(f"Q: {q}\nA: {a}")
                    sources.append("json")

            elif mode == "multi":
                for f in answer_fields:
                    a = row.get(f, "").strip()
                    if a:
                        records.append(f"Q: {q}\nA ({f}): {a}")
                        sources.append(f)

        # Add URLs as extra entries
        if urls:
            for text, url in urls:
                records.append(f"{text} (Source: {url})")
                sources.append(url)

        # Embedding model
        self.model = SentenceTransformer(model_name)

        # Init Chroma
        self.client = chromadb.PersistentClient(path=persist_path)
        self.collection = self.client.get_or_create_collection(name=collection_name)

        # Only populate once
        if self.collection.count() == 0:
            embeddings = self.model.encode(records, convert_to_numpy=True).tolist()
            self.collection.add(
                documents=records,
                embeddings=embeddings,
                metadatas=[{"source": s} for s in sources],
                ids=[str(i) for i in range(len(records))]
            )

        self.top_k = top_k
        self.use_llm = use_llm
        self.llm_model = llm_model

        if use_llm:
            load_dotenv()
            self.client_llm = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    def retrieve(self, query: str):
        """Retrieve top_k relevant docs for a single query."""
        q_vec = self.model.encode([query], convert_to_numpy=True)[0].tolist()
        results = self.collection.query(query_embeddings=[q_vec], n_results=self.top_k)

        docs = results["documents"][0] # type: ignore
        metas = results["metadatas"][0] # type: ignore
        sims = results["distances"][0] # type: ignore
        return list(zip(docs, metas, sims))

    def retrieve_batch(self, queries: list[str]):
        """Retrieve results for multiple queries at once."""
        q_vecs = self.model.encode(queries, convert_to_numpy=True).tolist()
        results = self.collection.query(query_embeddings=q_vecs, n_results=self.top_k)
        all_hits = []
        for i, q in enumerate(queries):
            docs = results["documents"][i] # pyright: ignore[reportOptionalSubscript]
            metas = results["metadatas"][i] # pyright: ignore[reportOptionalSubscript]
            sims = results["distances"][i] # pyright: ignore[reportOptionalSubscript]
            hits = list(zip(docs, metas, sims))
            all_hits.append((q, hits))
        return all_hits

    def answer(self, query: str):
        hits = self.retrieve(query)
        if not hits:
            return "No relevant entries found."

        if not self.use_llm:
            return "\n---\n".join(
                f"{doc}\n(source: {meta['source']}, distance: {dist:.2f})"
                for doc, meta, dist in hits
            )

        # --- LLM synthesis ---
        context_blocks, source_links = [], []
        for doc, meta, _ in hits:
            context_blocks.append(doc)
            if meta["source"].startswith("http"):
                source_links.append(meta["source"])

        context = "\n\n".join(context_blocks)
        sources_text = "\n".join(source_links) if source_links else "No external URLs retrieved."

        prompt = f"""
You are a helpful medical assistant.
Use the retrieved information to answer the user's question clearly and concisely.
- Summarize key points from Q&A.
- If there are URLs, include them as citations at the end.
- Do not copy text verbatim; synthesize into a clean short answer.
- Always include a disclaimer: "This information is for educational purposes only and not a substitute for professional medical advice."

User question:
{query}

Retrieved knowledge:
{context}

Sources:
{sources_text}

Final answer:
"""

        resp = self.client_llm.chat.completions.create(
            model=self.llm_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=250,
        )
        return resp.choices[0].message.content.strip() # type: ignore


# -----------------------
# DEMO
# -----------------------
if __name__ == "__main__":
    urls = [
        ("Morning-after pill guide", "https://www.drugs.com/mtm/morning-after.html"),
        ("Emergency contraception info", "https://www.drugs.com/condition/postcoital-contraception.html"),
    ]

    agent = JSONRAGChroma(
        "meddialog.json",
        answer_fields=["answer_chatgpt", "answer_icliniq", "answer_chatdoctor"],
        mode="concat",
        urls=urls,
        use_llm=True
    )

    queries = [
        "What are the side effects of the morning after pill?",
        "Is it safe to use emergency contraception if I have PCOD?",
        "Difference between regular contraceptive pills and i-pill?"
    ]

    for q in queries:
        print(f"\n Query: {q}")
        print(agent.answer(q))
        print("=" * 50)
