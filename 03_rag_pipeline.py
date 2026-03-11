"""
03 - RAG Pipeline (Retrieval Augmented Generation)
==================================================
Document chunking with RecursiveCharacterTextSplitter, OpenAI Embeddings,
and Pinecone vector store for semantic search and retrieval.
"""

import os
from dotenv import load_dotenv

load_dotenv()

from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

try:
    from langchain_pinecone import PineconeVectorStore
    HAS_PINECONE = True
except ImportError:
    HAS_PINECONE = False

from langchain_community.vectorstores import FAISS

# -----------------------------------------------------------------------------
# 1. Document Chunking with RecursiveCharacterTextSplitter
# -----------------------------------------------------------------------------
# Splits text recursively by separators: ["\\n\\n", "\\n", " ", ""]
# chunk_overlap: Overlap between chunks prevents losing context at boundaries.
# Larger overlap = more redundancy but better continuity; typical 10-20% of chunk_size.

def create_text_splitter(
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    ):
    """Create a text splitter with sensible defaults for RAG."""
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,      # Max chars per chunk
        chunk_overlap=chunk_overlap,  # Overlap to preserve context at boundaries
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )


# -----------------------------------------------------------------------------
# 2. Build Vector Store (Pinecone or FAISS + OpenAI Embeddings)
# -----------------------------------------------------------------------------
# OpenAI embeddings produce dense vectors. Pinecone for cloud; FAISS for local fallback.

def build_rag_chain(vectorstore):
    """Build a RAG chain: retrieve -> format -> generate."""
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    
    template = """Answer based only on the following context:
    {context}

    Question: {question}

    Answer:"""
    
    prompt = ChatPromptTemplate.from_template(template)
    llm = ChatOllama(model="llama3.2", temperature=0)
    
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)
    
    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return chain


# -----------------------------------------------------------------------------
# 3. Index Documents (Pinecone or FAISS fallback)
# -----------------------------------------------------------------------------
# Uses Pinecone when available with a valid index; otherwise FAISS (in-memory).

def index_documents(index_name: str = "langchain-demo"):
    """Chunk sample documents and add them to a vector store. Falls back to FAISS if Pinecone unavailable."""
    
    sample_docs = [
        Document(
            page_content="LangChain is a framework for developing applications powered by language models. "
            "It enables applications that are context-aware and connect a language model to other sources of data.",
            metadata={"source": "intro"},
        ),
        Document(
            page_content="RAG stands for Retrieval Augmented Generation. It combines retrieval of relevant documents "
            "with generation to produce accurate, grounded answers.",
            metadata={"source": "rag"},
        ),
    ]
    
    splitter = create_text_splitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(sample_docs)
    
    print("Chunks:")
    for i, c in enumerate(chunks):
        print(f"  [{i+1}] {c.page_content}")
    
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    
    # Print one embedding (first chunk)
    one_embedding = embeddings.embed_query(chunks[0].page_content)
    print(f"\nOne embedding (chunk 1, dim={len(one_embedding)}): {one_embedding[:5]}...")
    
    if HAS_PINECONE and os.getenv("PINECONE_API_KEY"):
        try:
            vectorstore = PineconeVectorStore.from_documents(
                documents=chunks,
                embedding=embeddings,
                index_name=index_name,
            )
            print(f"Indexed {len(chunks)} chunks to Pinecone index '{index_name}'.")
            return vectorstore
        except ValueError as e:
            if "No active indexes" in str(e) or "index" in str(e).lower():
                print("Pinecone has no indexes. Using FAISS (in-memory) instead.")
            else:
                raise
    
    # Fallback: FAISS (no Pinecone or no API key)
    vectorstore = FAISS.from_documents(documents=chunks, embedding=embeddings)
    print(f"Indexed {len(chunks)} chunks to FAISS (in-memory).")
    return vectorstore


# -----------------------------------------------------------------------------
# 4. Run RAG Query
# -----------------------------------------------------------------------------

def run_rag_query(chain, question: str):
    """Execute a RAG query."""
    return chain.invoke(question)


if __name__ == "__main__":
    # Ollama runs locally - no API key. Ensure: ollama pull llama3.2 && ollama pull nomic-embed-text
    index = os.getenv("PINECONE_INDEX_NAME", "langchain-demo")
    print("Indexing sample documents...")
    vectorstore = index_documents(index)
    print("\nBuilding RAG chain...")
    chain = build_rag_chain(vectorstore)
    print("\nQuerying: What is RAG?")
    answer = run_rag_query(chain, "What is RAG?")
    print(f"Answer: {answer}")
    print("\nDone.")
