# LangChain Examples

Modular, runnable Python examples for LangChain topics. Uses **Ollama** (local LLM, no API key).

## Setup

1. **Install and run Ollama** ([ollama.com](https://ollama.com))

   ```bash
   ollama pull llama3.2
   ollama pull nomic-embed-text   # for RAG embeddings in 03_rag_pipeline.py
   ```

2. **Create virtual environment (recommended)**

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

4. **Optional – `.env` for Pinecone** (only if using Pinecone in 03)

   Copy `.env.example` to `.env` and add `PINECONE_API_KEY` if needed.

## Files

| File | Topic |
|------|-------|
| `01_prompt_templates.py` | PromptTemplate, FewShotPromptTemplate |
| `02_memory_types.py` | Buffer, Window, Summary memory |
| `03_rag_pipeline.py` | RAG with RecursiveCharacterTextSplitter, OpenAI Embeddings, Pinecone |
| `04_agents.py` | Zero-shot ReAct agent with LLMMath tool |
| `05_lcel_basics.py` | Pipe operator, RunnablePassthrough, RunnableLambda |

## Run

```bash
python 01_prompt_templates.py
python 02_memory_types.py
python 03_rag_pipeline.py   # needs Pinecone index
python 04_agents.py
python 05_lcel_basics.py
```

## Pinecone (03_rag_pipeline.py)

Uses FAISS by default (in-memory). For Pinecone: create an index with dimensions matching your embedding model (e.g. 768 for `nomic-embed-text`), set `PINECONE_API_KEY` and `PINECONE_INDEX_NAME` in `.env`.
