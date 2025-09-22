import os
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from openai import OpenAI
import chromadb
from typing import Optional


class URLRAGChroma:
    def __init__(
        self,
        urls: list[tuple[str, str]],   # list of (text, url)
        model_name: str = "multi-qa-mpnet-base-dot-v1",
        top_k: int = 3,
        use_llm: bool = False,
        llm_model: str = "gpt-4o-mini",
        collection_name: str = "rag_urls",
        persist_path: str = "./rag"
    ):
        # Init embedding model
        self.model = SentenceTransformer(model_name)

        # Init Chroma
        self.client = chromadb.PersistentClient(path=persist_path)
        self.collection = self.client.get_or_create_collection(name=collection_name)

        # Only populate if empty
        if self.collection.count() == 0:
            records = [f"{desc} (Source: {url})" for desc, url in urls]
            sources = [url for _, url in urls]
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
        q_vec = self.model.encode([query], convert_to_numpy=True)[0].tolist()
        results = self.collection.query(query_embeddings=[q_vec], n_results=self.top_k)

        docs = results["documents"][0]
        metas = results["metadatas"][0]
        sims = results["distances"][0]
        return list(zip(docs, metas, sims))

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
- Summarize key points from the retrieved docs.
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
        return resp.choices[0].message.content.strip()


# -----------------------
# DEMO
# -----------------------
if __name__ == "__main__":
    urls = [
        ("Morning-after pill guide", "https://www.drugs.com/mtm/morning-after.html"),
        ("Emergency contraception info", "https://www.drugs.com/condition/postcoital-contraception.html"),
        ("Planned Parenthood emergency contraception", "https://www.plannedparenthood.org/learn/morning-after-pill-emergency-contraception")
    ]

    agent = URLRAGChroma(
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
