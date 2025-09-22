import os
from openai import OpenAI
from sentence_transformers import SentenceTransformer, util

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def unpack_hits(hits):
    """Normalize hits to always return (doc, meta, score)."""
    normalized = []
    for h in hits:
        if len(h) == 3:
            doc, meta, score = h
        elif len(h) == 2:
            doc, meta = h
            score = 1.0   # default score if missing
        else:
            doc = h
            meta = {"source": "unknown"}
            score = 1.0
        normalized.append((doc, meta, score))
    return normalized


def eval_retrieval(query, retrieved_docs, gold_answer=None, embed_model=None):
    """Evaluate retrieval quality."""
    if gold_answer:
        sim_scores = [
            util.cos_sim(embed_model.encode(gold_answer), embed_model.encode(doc))[0].item()
            for doc in retrieved_docs
        ]
        recall = max(sim_scores) > 0.7
        return {"recall@k": float(recall)}
    else:
        context = "\n".join(retrieved_docs)
        prompt = f"""
You are a judge. Query: "{query}"
Retrieved docs: {context}

Are the retrieved docs relevant to the query?
Return ONLY 1 if relevant, 0 if not.
"""
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=5,
        )
        return {"relevance": int(resp.choices[0].message.content.strip())}


def eval_hallucination(context, answer):
    """Estimate hallucination rate using LLM-as-judge."""
    prompt = f"""
You are an evaluator.
Here is retrieved context:
{context}

Here is the model's answer:
{answer}

Task:
- Split the answer into statements.
- For each statement, check if it is directly supported by the context.
- Count unsupported statements as hallucinations.
- Return ONLY a number between 0 and 1 = fraction of hallucinated statements.
"""
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=10,
    )
    try:
        return float(resp.choices[0].message.content.strip())
    except:
        return None


def evaluate_agent(agent, queries, gold_answers=None, embed_model_name="all-MiniLM-L6-v2"):
    """Evaluate a RAG agent across queries."""
    embed_model = SentenceTransformer(embed_model_name)
    results = []

    for q in queries:
        hits = agent.retrieve(q)
        hits = unpack_hits(hits)   # ✅ normalize here

        docs = [doc for doc, _, _ in hits]
        answer = agent.answer(q) if hasattr(agent, "answer") else agent.handle(q)

        # Retrieval metrics
        if gold_answers and q in gold_answers:
            retrieval_metrics = eval_retrieval(q, docs, gold_answers[q], embed_model)
        else:
            retrieval_metrics = eval_retrieval(q, docs, None, embed_model)

        # Hallucination metric
        context = "\n".join(docs)
        halluc_rate = eval_hallucination(context, answer)

        results.append({
            "query": q,
            "retrieval": retrieval_metrics,
            "hallucination_rate": halluc_rate,
            "answer": answer
        })

    return results
