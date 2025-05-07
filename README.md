Zomato Restaurant Q&A Chatbot
A simple chatbot that answers questions about restaurants using a Zomato CSV file. It runs locally using a language model via Ollama and a RAG pipeline.

Main Features:

Ask about cuisine, location, price, etc.

Uses your dataset to give accurate answers

Runs offline using Streamlit and a local LLM

Requirements:

Python 3.9+

Ollama (with Llama 3.1 8B or similar)

Streamlit, LangChain, ChromaDB, Sentence Transformers, Pandas

Setup:

Clone the repo and go to the folder.

Create and activate a virtual environment.

Install dependencies with: pip install -r requirements.txt

Install and run Ollama, then pull the model: ollama pull llama3.1:8b

Add your Zomato CSV to the project folder.

Update config.py with the correct file name and Ollama host if needed.

Run:

Start Ollama (ollama serve)

Activate your environment

Run: streamlit run main.py

Open the browser link shown (usually http://localhost:8501)

Notes:

Data is stored in a vector DB (Chroma)

Your CSV is required for responses

The project ignores venv, cache files, and DB folders

