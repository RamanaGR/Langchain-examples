# LangChain Examples

Modular, runnable Python examples for LangChain topics. Uses **OpenAI** models only.

> **Setup:** Copy `.env.example` to `.env` and add your `OPENAI_API_KEY`. Never commit `.env`.

## Setup

1. **Create virtual environment (recommended)**

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

3. **Configure environment**

   Copy `.env.example` to `.env` and add your keys:

   ```bash
   cp .env.example .env
   ```

   Required in `.env`:

   - `OPENAI_API_KEY` – for all scripts
   - `PINECONE_API_KEY` – for `03_rag_pipeline.py` only

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

Create a Pinecone index with:

- Dimensions: `1536` (for `text-embedding-3-small`)
- Metric: `cosine`

Or set `PINECONE_INDEX_NAME` in `.env` to match your index name.
