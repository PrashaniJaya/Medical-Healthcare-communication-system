import json
import os
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

class RAGAgent:
    def __init__(self, kb_file, answer_field="answer_chatgpt",
                 top_k=3, threshold=0.3, model_name="multi-qa-mpnet-base-dot-v1",
                 use_llm=False, llm_model="gpt-4o-mini"):
        """
        kb_file: JSON file with [{"input": "...", "answer_chatgpt": "...", ...}]
        answer_field: which answer field to use ("answer_chatgpt", "answer_icliniq", etc.)
        use_llm: whether to synthesize a final answer using LLM
        llm_model: which OpenAI model to use for synthesis
        """
        with open(kb_file, "r", encoding="utf-8") as f:
            raw_entries = json.load(f)

        self.kb_entries = []
        for row in raw_entries:
            q = row.get("input", "").strip()
            a = row.get(answer_field, "").strip()
            if q and a:
                self.kb_entries.append(f"Q: {q}\nA: {a}")

        # Embedding model
        self.model = SentenceTransformer(model_name)
        self.embedded_kb = self.model.encode(
            self.kb_entries, convert_to_numpy=True, show_progress_bar=True
        )

        self.top_k = top_k
        self.threshold = threshold
        self.use_llm = use_llm
        self.llm_model = llm_model
        if use_llm:
            self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))  

    def retrieve(self, query):
        """Retrieve top-k relevant KB entries"""
        q_vec = self.model.encode([query], convert_to_numpy=True)
        sims = cosine_similarity(q_vec, self.embedded_kb)[0]
        top_indices = [i for i in np.argsort(sims)[::-1] if sims[i] >= self.threshold][:self.top_k]
        return [(self.kb_entries[i], sims[i]) for i in top_indices]

    def handle(self, query):
        retrieved = self.retrieve(query)

        if not retrieved:
            return "No relevant entries found."

        if not self.use_llm:
            # Return raw retrieved Q&A
            return "\n---\n".join(f"{text} (score: {score:.2f})" for text, score in retrieved)

        # Prepare prompt for synthesis
        context = "\n\n".join(text for text, _ in retrieved)
        prompt = f"""
You are a helpful medical assistant.
Use the following retrieved Q&A information to answer the user’s question clearly and concisely.
Do not copy verbatim; synthesize into a coherent short answer.
If something is uncertain, state it explicitly.
Always include a disclaimer: "This information is for educational purposes only and not a substitute for professional medical advice."

User question:
{query}

Retrieved info:
{context}

Final answer:
"""

        response = self.client.chat.completions.create(
            model=self.llm_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=250
        )
        return response.choices[0].message.content.strip() # type: ignore


# -----------------------
# DEMO
# -----------------------
if __name__ == "__main__":
    # Use JSON KB file you showed earlier
    agent = RAGAgent("meddialog.json", answer_field="answer_chatgpt", use_llm=True)

    queries = [
        "What are the side effects of Unwanted 72?",
        "Is it safe to take the morning after pill multiple times in one month?",
        "How effective is the i-pill compared to regular contraceptive pills?"
    ]

    for q in queries:
        print(f"\n Query: {q}")
        print(agent.handle(q))
        print("=" * 50)
