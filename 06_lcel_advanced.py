"""
06 - LCEL Advanced
==================
Using RunnableParallel and RunnablePassthrough for complex, branching data flow.
- RunnableParallel: Run multiple branches in parallel and merge outputs.
- RunnablePassthrough: Pass through input or assign to keys in a dict.
"""

import os
from dotenv import load_dotenv

load_dotenv()

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda

# -----------------------------------------------------------------------------
# 1. RunnableParallel
# -----------------------------------------------------------------------------
# Runs multiple runnables in parallel and merges their outputs into a single dict.

def demo_runnable_parallel():
    """Run multiple prompt+LLM branches in parallel and merge results."""
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    parser = StrOutputParser()

    prompt_summary = ChatPromptTemplate.from_template("Summarize in one sentence: {text}")
    prompt_sentiment = ChatPromptTemplate.from_template("Classify sentiment of: {text}")

    summary_chain = prompt_summary | llm | parser
    sentiment_chain = prompt_sentiment | llm | parser

    parallel = RunnableParallel(
        summary=summary_chain,
        sentiment=sentiment_chain,
    )

    result = parallel.invoke({"text": "The new AI model exceeded all expectations."})
    print("=== RunnableParallel Output ===")
    print("Summary:", result["summary"])
    print("Sentiment:", result["sentiment"])


# -----------------------------------------------------------------------------
# 2. RunnablePassthrough.assign
# -----------------------------------------------------------------------------
# Passes input through and adds new keys from runnable outputs.

def demo_passthrough_assign():
    """Use assign to add new keys while passing through original input."""
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    parser = StrOutputParser()

    prompt_expand = ChatPromptTemplate.from_template("Expand this into a full sentence: {topic}")
    expand_chain = prompt_expand | llm | parser

    assign_chain = RunnablePassthrough.assign(expanded=expand_chain)

    result = assign_chain.invoke({"topic": "LLMs"})
    print("\n=== RunnablePassthrough.assign Output ===")
    print("Original:", result["topic"])
    print("Expanded:", result["expanded"])


# -----------------------------------------------------------------------------
# 3. Complex Flow: Parallel + Sequential
# -----------------------------------------------------------------------------

def demo_complex_flow():
    """Parallel retrieval then synthesis step."""
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    parser = StrOutputParser()

    def perspective_a(x):
        return "Optimistic: Technology will solve major problems."

    def perspective_b(x):
        return "Cautious: We must regulate AI carefully."

    parallel = RunnableParallel(
        opt=RunnableLambda(perspective_a),
        cautious=RunnableLambda(perspective_b),
    )

    synthesis_prompt = ChatPromptTemplate.from_template(
        "Given these two views:\nOpt: {opt}\nCautious: {cautious}\n\n"
        "Write one balanced sentence synthesizing both."
    )
    synthesis_chain = synthesis_prompt | llm | parser
    full_chain = parallel | synthesis_chain

    result = full_chain.invoke({})
    print("\n=== Complex Flow Output ===")
    print(result)


if __name__ == "__main__":
    print("Running LCEL advanced demos...\n")
    demo_runnable_parallel()
    demo_passthrough_assign()
    demo_complex_flow()
    print("\nDone.")
