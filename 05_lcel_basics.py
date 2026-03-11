"""
05 - LCEL Basics (LangChain Expression Language)
================================================
Demonstrates the Pipe operator (|) and custom Runnables for composable chains.
LCEL makes it easy to connect components in a linear or branching flow.
"""

import os
from typing import Any
from dotenv import load_dotenv

load_dotenv()

from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda,RunnableParallel

# -----------------------------------------------------------------------------
# 1. Pipe Operator (|)
# -----------------------------------------------------------------------------
# The pipe operator chains components: output of one becomes input of next.
# Equivalent to: prompt | llm | parser  =>  parser(llm(prompt))
# Benefits: Composable, streamable, batchable, and async-native.

def demo_pipe_operator():
    """Chain components: regular (modular) vs pipe syntax comparison."""
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a concise assistant."),
        ("human", "{question}"),
    ])
    llm = ChatOllama(model="llama3.2", temperature=0)
    parser = StrOutputParser()
    question = "What is 2 + 2? Reply in one word."

    # --- REGULAR SYNTAX (modular, step-by-step) ---
    # Each step: invoke with previous output. Verbose but explicit.
    step1_formatted = prompt.invoke({"question": question})
    step2_llm_out = llm.invoke(step1_formatted)
    step3_parsed = parser.invoke(step2_llm_out)
    response_regular = step3_parsed

    print("=== Regular Syntax (modular) ===")
    print("  step1: prompt.invoke(...)  -> formatted messages")
    print("  step2: llm.invoke(step1)   -> LLM response")
    print("  step3: parser.invoke(step2)-> final string")
    print(f"  Output: {response_regular}")

    # --- PIPE SYNTAX (shortcut, same result) ---
    # One line chains all steps. Simpler and easier to read.
    chain = prompt | llm | parser
    response_pipe = chain.invoke({"question": question})

    print("\n=== Pipe Syntax (shortcut) ===")
    print("  chain = prompt | llm | parser")
    print("  response = chain.invoke({'question': question})")
    print(f"  Output: {response_pipe}")



# -----------------------------------------------------------------------------
# 2. Custom RunnableLambda
# -----------------------------------------------------------------------------
# RunnableLambda wraps a function to make it a Runnable. Use for simple
# transformations (e.g., string formatting, filtering) in the pipeline.

def demo_custom_runnable():
    """Use RunnableLambda for custom logic in the chain."""
    
    def add_prefix(text: str) -> str:
        return f"PREFIX: {text}"
    
    def extract_first_line(text: str) -> str:
        return text.split("\n")[0] if text else ""
    
    prompt = ChatPromptTemplate.from_template("List 3 colors. One per line.")
    llm = ChatOllama(model="llama3.2", temperature=0)
    parser = StrOutputParser()
    
    chain = (
        prompt
        | llm
        | parser
        | RunnableLambda(extract_first_line)
        | RunnableLambda(add_prefix)
    )
    
    response = chain.invoke({})
    print("\n=== Custom Runnable Output ===")
    print(response)

# 1. RunnableParallel
# -----------------------------------------------------------------------------
# Runs multiple runnables in parallel and merges their outputs into a single dict.
# Keys in RunnableParallel become keys in the output. Speeds up multi-branch flows.

def demo_runnable_parallel():
    """Run multiple prompt+LLM branches in parallel and merge results."""
    
    llm = ChatOllama(model="llama3.2", temperature=0)
    parser = StrOutputParser()
    
    prompt_summary = ChatPromptTemplate.from_template("Summarize in one sentence: {text}")
    prompt_sentiment = ChatPromptTemplate.from_template("Classify sentiment of: {text}")
    
    summary_chain = prompt_summary | llm | parser
    sentiment_chain = prompt_sentiment | llm | parser
    
    # Both chains receive the same {"text": ...} input and run in parallel
    parallel = RunnableParallel[dict](
        summary=summary_chain,
        sentiment=sentiment_chain,
    )
    
    result = parallel.invoke({"text": "The new AI model exceeded all expectations."})
    print("=== RunnableParallel Output ===")
    print("Summary:", result["summary"])
    print("Sentiment:", result["sentiment"])


if __name__ == "__main__":
    print("Running LCEL basics demos...\n")
    # demo_pipe_operator()
    # demo_custom_runnable()
    demo_runnable_parallel()
    print("\nDone.")
