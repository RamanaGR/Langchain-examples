"""
02 - Memory Types
=================
Demonstrates Buffer, Window, and Summary memory for conversation persistence.
- Buffer: Stores full conversation (can hit token limits)
- Window: Keeps last K turns (sliding window to control context size)
- Summary: Compresses history into a summary (reduces tokens for long chats)
"""

import os
from dotenv import load_dotenv

load_dotenv()

from langchain_openai import ChatOpenAI
from langchain_classic.chains import ConversationChain
from langchain_classic.memory import (
    ConversationBufferMemory,
    ConversationBufferWindowMemory,
    ConversationSummaryMemory,
)

# -----------------------------------------------------------------------------
# 1. ConversationBufferMemory
# -----------------------------------------------------------------------------
# Stores the entire conversation history as-is.
# Risk: Long conversations can exceed model's max_token_limit and cause errors.
# Use when: Short conversations or when full context is critical.

def demo_buffer_memory():
    """Full conversation history stored in memory."""
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    
    memory = ConversationBufferMemory(
        return_messages=True,  # Return as message objects (for chat models)
        memory_key="history",
    )
    
    chain = ConversationChain(llm=llm, memory=memory, verbose=True)
    
    print("=== Buffer Memory (full history) ===")
    chain.invoke({"input": "My name is Alice."})
    chain.invoke({"input": "What's my name?"})
    
    print("\nStored messages:", memory.chat_memory.messages)


# -----------------------------------------------------------------------------
# 2. ConversationBufferWindowMemory
# -----------------------------------------------------------------------------
# Keeps only the last K exchanges (sliding window).
# k=2 means only the 2 most recent human-AI pairs are retained.
# Why: Prevents unbounded growth; older context is dropped to stay within token limits.

def demo_window_memory():
    """Sliding window of last k interactions."""
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    
    memory = ConversationBufferWindowMemory(
        k=3,  # Keep only last 2 exchanges; older ones are forgotten
        return_messages=True,
        memory_key="history",
    )
    
    chain = ConversationChain(llm=llm, memory=memory, verbose=True)
    
    print("\n=== Window Memory (k=2, last 2 exchanges only) ===")
    chain.invoke({"input": "My name is Bob."})
    chain.invoke({"input": "I like pizza."})
    chain.invoke({"input": "I like burger also."})
    r1 = chain.invoke({"input": "What's my name?"})  # May not remember "Bob" - it was >2 turns ago
    print(r1["response"])


# -----------------------------------------------------------------------------
# 3. ConversationSummaryMemory
# -----------------------------------------------------------------------------
# Uses an LLM to summarize the conversation as it grows.
# Reduces token usage for long chats by compressing history into a short summary.
# Why: Enables long conversations without hitting max_token_limit.

def demo_summary_memory():
    """Summarizes conversation history to save tokens."""
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    
    memory = ConversationSummaryMemory(
        llm=llm,  # Same LLM used to generate summaries
        return_messages=True,
        memory_key="history",
    )
    
    chain = ConversationChain(llm=llm, memory=memory, verbose=True)
    
    print("\n=== Summary Memory (compresses history) ===")
    chain.invoke({"input": "I'm Carol. I work as a data scientist at a fintech company."})
    chain.invoke({"input": "I enjoy hiking and reading science fiction."})
    r1 = chain.invoke({"input": "What do you know about me?"})
    print(r1["response"])
    


if __name__ == "__main__":
    print("Running Memory demos...\n")
    # demo_buffer_memory()
    # demo_window_memory()
    demo_summary_memory()
    print("\nDone.")
