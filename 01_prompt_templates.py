"""
01 - Prompt Templates
=====================
Demonstrates PromptTemplate and FewShotPromptTemplate for structured LLM inputs.
Uses OpenAI via langchain_openai.
"""

import os
from dotenv import load_dotenv

load_dotenv()

from langchain_core.prompts import PromptTemplate, FewShotPromptTemplate
from langchain_openai import ChatOpenAI

# -----------------------------------------------------------------------------
# 1. Basic PromptTemplate
# -----------------------------------------------------------------------------
# PromptTemplate allows you to define reusable prompts with placeholders.
# Variables are filled at runtime, enabling dynamic content injection.

def demo_prompt_template():
    """Basic PromptTemplate with single and multiple variables."""
    # Single variable template
    simple_template = PromptTemplate(
        input_variables=["topic"],
        template="Explain {topic} in 3 sentences for a beginner.",
    )
    
    formatted1 = simple_template.format(topic="quantum computing")
    print("=== Simple PromptTemplate Output ===")
    print(formatted1)
    
    # Multi-variable template
    multi_template = PromptTemplate(
        input_variables=["role", "task", "constraint"],
        template="You are a {role}. {task} Constraint: {constraint}",
    )
    
    formatted = multi_template.format(
        role="Python tutor",
        task="Write a function to reverse a string.",
        constraint="Use no built-in reverse methods.",
    )
    print("\n=== Multi-Variable PromptTemplate Output ===")
    print(formatted)
    
    # Invoke LLM with the formatted prompt
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    response = llm.invoke(formatted1)
    print("\n=== LLM Response ===")
    print(response.content)


# -----------------------------------------------------------------------------
# 2. FewShotPromptTemplate
# -----------------------------------------------------------------------------
# Few-shot prompting provides examples to the LLM for in-context learning.
# The model learns the expected format from examples before tackling new inputs.

def demo_few_shot_template():
    """FewShotPromptTemplate for sentiment classification with examples."""
    
    # Example format: input -> output pairs
    examples = [
        {"input": "This product is amazing! Best purchase ever.", "output": "Positive"},
        {"input": "Terrible experience. Would not recommend.", "output": "Negative"},
        {"input": "It's okay, nothing special.", "output": "Neutral"},
    ]
    
    example_template = """
    Input: {input}
    Output: {output}
    """
    
    example_prompt = PromptTemplate(
        input_variables=["input", "output"],
        template=example_template,
    )
    
    few_shot_prompt = FewShotPromptTemplate(
        examples=examples,
        example_prompt=example_prompt,
        prefix="Classify the sentiment of each review as Positive, Negative, or Neutral.",
        suffix="Input: {user_input}\nOutput:",
        input_variables=["user_input"],
    )
    
    formatted = few_shot_prompt.format(user_input="The delivery was fast but the product was damaged.")
    print("\n=== FewShotPromptTemplate Output ===")
    print(formatted)
    
    # Invoke LLM
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    response = llm.invoke(formatted)
    print("\n=== Classification Result ===")
    print(response.content)


if __name__ == "__main__":
    print("Running PromptTemplate demos...\n")
    # demo_prompt_template()
    demo_few_shot_template()
    print("\nDone.")
