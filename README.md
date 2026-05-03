# DocMind
**A local multi-agent RAG chatbot for querying your documents with Groq + Llama 3.1.**

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Groq](https://img.shields.io/badge/Groq-API-orange)
![LangChain](https://img.shields.io/badge/LangChain-TextSplitters-brightgreen)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

## Features
- Multi-agent planning, retrieval, and synthesis pipeline
- Local embeddings with `sentence-transformers` and persistent ChromaDB
- Streamlit UI with document upload, chat history, sources, and caching
- SQLite memory cache for similar questions
- Standalone ingestion script for batch indexing

## Architecture
```
User Question
     |
     v
Planner Agent  ---> Sub-questions
     |
     v
RAG Agent(s)  ---> ChromaDB (local) + Embeddings
     |
     v
Synthesizer  ---> Final Answer
     |
     v
Memory Store (SQLite)
```

## Prerequisites
- Python 3.10+
- A Groq API key from console.groq.com

## Installation
1. Clone or download this repository.
2. Create and activate a virtual environment.
3. Install dependencies:

```bash
pip install -r requirements.txt
```

## Environment Setup
1. Copy the example environment file:

```bash
copy .env.example .env
```

2. Add your Groq API key to `.env`:

```
GROQ_API_KEY=your_groq_key_here
```

## How to Get a Free Groq API Key
1. Visit console.groq.com
2. Create an account or sign in.
3. Navigate to API keys and generate a new key.
4. Paste the key into your `.env` file.

## Run the Project
### Option A: Index from the command line
Place PDF, CSV, JSON, or TXT files into `data/` and run:

```bash
python ingestion/loader.py
```

### Option B: Upload inside the app
The UI can upload and index documents directly. Files are saved to `data/` and indexed automatically.

### Start the Streamlit app
```bash
streamlit run ui/app.py
```

## Supported Document Types
- PDF
- CSV
- JSON
- TXT / MD

## Project Structure
```
docmind/
├── ingestion/
│   └── loader.py
├── agents/
│   ├── planner.py
│   ├── rag_agent.py
│   └── synthesizer.py
├── memory/
│   └── store.py
├── ui/
│   └── app.py
├── data/
├── .env.example
├── .gitignore
├── requirements.txt
└── README.md
```

## Tech Stack
| Layer | Technology |
| --- | --- |
| LLM | Groq API (llama-3.1-8b-instant) |
| Embeddings | sentence-transformers (all-MiniLM-L6-v2) |
| Vector DB | ChromaDB (persistent) |
| UI | Streamlit |
| Memory | SQLite |
| Runtime | PyTorch + Transformers |

## Contributing
Contributions are welcome. Open an issue or submit a pull request.

## License
MIT
