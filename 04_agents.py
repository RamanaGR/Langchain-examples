"""
04 - Zero-Shot ReAct Agent
==========================
ReAct (Reason + Act) agent that uses tools to solve tasks.
Demonstrates LLMMathChain as a tool for mathematical reasoning.
"""

import os
from dotenv import load_dotenv

load_dotenv()

from langchain_openai import ChatOpenAI
from langchain_classic.agents import AgentExecutor, create_react_agent
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import Tool

# -----------------------------------------------------------------------------
# 1. Define Tools
# -----------------------------------------------------------------------------
# Tools give the agent capabilities (e.g., math, search, code execution).
# We use LLMMathChain when available, else a numexpr-based calculator.
# LLMMathChain: parses natural language math -> Python expr -> numexpr eval.

def _calculator_numexpr(expr: str) -> str:
    """Evaluate math expression using numexpr (safe, fast)."""
    import re
    import numexpr  # type: ignore
    # Strip non-math chars; allow digits, ops, parens
    cleaned = re.sub(r"[^\d\s\+\-\*\/\.\(\)\%\*\*]", "", expr)
    if not cleaned.strip():
        return "Invalid expression"
    try:
        return str(numexpr.evaluate(cleaned))
    except Exception as e:
        return f"Error: {e}"


def _get_calculator_tool_llm_math(llm):
    """Use LLMMathChain (from langchain) if available."""
    try:
        from langchain.chains.llm_math.base import LLMMathChain
        chain = LLMMathChain.from_llm(llm=llm)

        def calc(expr: str) -> str:
            r = chain.invoke({"question": expr})
            return str(r.get("answer", r))

        return Tool(
            name="Calculator",
            description="Useful for answering math questions. Input should be a valid mathematical expression.",
            func=calc,
        )
    except ImportError:
        return Tool(
            name="Calculator",
            description="Useful for answering math questions. Input should be a valid mathematical expression.",
            func=_calculator_numexpr,
        )


def get_tools(llm):
    """Create tools for the agent. Prefer LLMMathChain, fallback to numexpr."""
    return [_get_calculator_tool_llm_math(llm)]


# -----------------------------------------------------------------------------
# 2. ReAct Prompt
# -----------------------------------------------------------------------------
# The prompt instructs the agent to think (Reason) and take actions (Act).
# Placeholders: {input}, {agent_scratchpad}, {tools}, {tool_names}

REACT_PROMPT = """You are a helpful assistant. Use the following tools:

{tools}

Use the following format:

Question: the input question
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {input}
Thought: {agent_scratchpad}
"""


# -----------------------------------------------------------------------------
# 3. Create and Run Agent
# -----------------------------------------------------------------------------

def create_agent():
    """Create a zero-shot ReAct agent with LLMMath tool."""
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    tools = get_tools(llm)
    
    prompt = PromptTemplate.from_template(REACT_PROMPT)
    
    agent = create_react_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        handle_parsing_errors=True,
    )
    
    return agent_executor


def run_agent(agent_executor, question: str):
    """Invoke the agent with a question."""
    return agent_executor.invoke({"input": question})


if __name__ == "__main__":
    print("Creating zero-shot ReAct agent with Calculator tool...\n")
    agent = create_agent()
    
    result = run_agent(agent, "What is (15 * 3) + (100 / 4)?")
    print("\n=== Result ===")
    print(result["output"])
    
    print("\nDone.")
