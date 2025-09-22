from Ragwithwebscraping import ScrapeChroma
from RAGagentmultianswer import JSONRAGChroma
from RAGagentwithmeddialog import RAGAgent
from ragwithoutmeddialog import URLRAGChroma
from evaluateRAG import evaluate_agent


if __name__ == "__main__":
    # Example queries
    queries = [
        "Hi doctor,My girlfriend had morning after pill and got her period a few days later. Then we had sex after 15 days of her periods and I did not ejaculate inside her but still no period. Would she be pregnant?",
        "Hello doctor,I took an I-pill after 48 hours of intercourse. Now I am experiencing brown spotting a week before my periods. Is it a sign of pregnancy?"
    ]

    # Gold answers for MedDialog evaluation (add more if available)
    gold_answers = {
        "Hi doctor,My girlfriend had morning after pill and got her period a few days later. Then we had sex after 15 days of her periods and I did not ejaculate inside her but still no period. Would she be pregnant?":
            "Hi, Welcome to Chat Doctor forum. Considering that her with ChatDoctor.  So, having unprotected sex during this time, even though you ejaculated outside, still the chances of precum inside her vagina remains. Precum contains live sperms and hence there may be a chance that she may get pregnant. So, the best way to rule out pregnancy at present would be to get a serum beta hCG test. If the hCG levels less than 5 ng/dL, that indicates negative result and she is not pregnant. In that case, the deal in menses is due to the morning after pill as it is a side effect. Menses shall resume in next few days and one can wait safely. So, at present please get a serum beta hCG test done."
        "Hello doctor,I took an I-pill after 48 hours of intercourse. Now I am experiencing brown spotting a week before my periods. Is it a sign of pregnancy?"
            "Hi, Welcome to Chat Doctor forum. As you took an emergency contraceptive pill within 48 hours of intercourse, you can expect around 75 % protection towards pregnancy. The protection rate decreases with a delay in the intake of pill from intercourse. The spotting you had could be due to with ChatDoctor.  So, to be sure, if you miss regular bleeding during your expected date of periods, please go for urine pregnancy test once. For more information consult an obstetrician and gynaecologist online "
    }

    # Instantiate agents
    agent_med = RAGAgent("meddialog.json", use_llm=True)
    agent_url = URLRAGChroma([
        ("Morning-after pill guide", "https://www.drugs.com/mtm/morning-after.html"),
        ("Emergency contraception info", "https://www.drugs.com/condition/postcoital-contraception.html"),
    ], use_llm=True)

    agent_hybrid = JSONRAGChroma(
        "meddialog.json",
        answer_fields=["answer_chatgpt", "answer_icliniq", "answer_chatdoctor"], 
        mode="concat",
        urls=[("Morning-after pill guide", "https://www.drugs.com/mtm/morning-after.html")],
        use_llm=True
    )

    agent_scraping = ScrapeChroma([
        "https://www.drugs.com/mtm/morning-after.html",
        "https://www.plannedparenthood.org/learn/morning-after-pill-emergency-contraception"
    ], use_llm=True)

    # Evaluate all
    print("\n=== Evaluating MedDialog Agent ===")
    results_med = evaluate_agent(agent_med, queries, gold_answers)
    print(results_med)

    print("\n=== Evaluating URL Agent ===")
    results_url = evaluate_agent(agent_url, queries)
    print(results_url)

    print("\n=== Evaluating Hybrid Agent ===")
    results_hybrid = evaluate_agent(agent_hybrid, queries)
    print(results_hybrid)

    print("\n=== Evaluating Scraping Agent ===")
    results_scraping = evaluate_agent(agent_scraping, queries)
    print(results_scraping)
