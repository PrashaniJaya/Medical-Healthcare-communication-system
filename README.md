# Medical-Healthcare-communication-system
The medical communication system first starts with creation of multiagents to dynamically process the query. 
The First agent is RAG and in the repo there are 4 types
  1. Rag agent with medical dialog from hugging face
  2. Rag agent with URLS directly inserted to RAG
  3. Rag with a webscraper
  4. Rag with both a medical dialog and URLs

Each of these agents provided answers for the same three questions buidling different vector databases. The answers are stored in Testing phase 001
Each of these agents were also evaluated by taking two test questions out of the medical dialog.json and verified the hallucinations and the retrieval index and is stored in evaluatorphase001.txt



