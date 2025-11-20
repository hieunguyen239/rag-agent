"""
Simple Tool System for Agentic RAG

This module provides basic tools that the agent can use:
1. RAGTool - Search the knowledge base (text-only)
2. CalculatorTool - Perform mathematical calculations
"""

from typing import Dict, Any
import re
import math
import os
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from lang_chain_llm import call_llama


class RAGTool:
    """Tool for searching the knowledge base using text-based RAG."""
    
    name = "search_knowledge_base"
    description = "Search the knowledge base of documents for information about returns, products, policies, and processes. Use this when you need to find factual information from documents."
    
    def __init__(self, vectorstore: FAISS, embeddings):
        """
        Initialize the RAG tool.
        
        Args:
            vectorstore: The FAISS vectorstore instance
            embeddings: The embeddings model
        """
        self.vectorstore = vectorstore
        self.embeddings = embeddings
        self.retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    
    def execute(self, query: str) -> str:
        """
        Execute a search query against the knowledge base.
        
        Args:
            query: The search query
            
        Returns:
            The answer from the RAG system
        """
        try:
            # Retrieve relevant documents
            retrieved_docs = self.retriever.invoke(query)
            
            if not retrieved_docs:
                return "No relevant information found in the knowledge base."
            
            # Build context from retrieved documents
            context_parts = []
            for i, doc in enumerate(retrieved_docs, 1):
                context_parts.append(f"{i}. {doc.page_content}")
            
            context = "\n".join(context_parts)
            
            # Create prompt for LLM
            prompt = f"""You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question.
If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.

<context>
{context}
</context>

Question: {query}

Helpful Answer:"""
            
            # Generate answer using LLM
            answer = call_llama(prompt)
            
            # Add source count
            answer += f"\n\n(Based on {len(retrieved_docs)} document(s))"
            
            return answer
            
        except Exception as e:
            return f"Error searching knowledge base: {str(e)}"


class CalculatorTool:
    """Tool for performing mathematical calculations."""
    
    name = "calculate"
    description = "Perform mathematical calculations. Supports basic arithmetic (+, -, *, /), exponents (**), and common math functions (sqrt, sin, cos, log, etc.). Input should be a valid mathematical expression."
    
    def execute(self, expression: str) -> str:
        """
        Execute a mathematical calculation.
        
        Args:
            expression: A mathematical expression to evaluate
            
        Returns:
            The result of the calculation as a string
        """
        try:
            # Clean the expression
            expression = expression.strip()
            
            # Create a safe environment with math functions
            safe_dict = {
                "__builtins__": {},
                "abs": abs,
                "round": round,
                "min": min,
                "max": max,
                "sum": sum,
                "sqrt": math.sqrt,
                "sin": math.sin,
                "cos": math.cos,
                "tan": math.tan,
                "log": math.log,
                "log10": math.log10,
                "exp": math.exp,
                "pi": math.pi,
                "e": math.e,
            }
            
            # Evaluate the expression
            result = eval(expression, safe_dict, {})
            
            # Format the result
            if isinstance(result, float):
                # Round to reasonable precision
                if result.is_integer():
                    return str(int(result))
                else:
                    return f"{result:.6f}".rstrip('0').rstrip('.')
            else:
                return str(result)
                
        except ZeroDivisionError:
            return "Error: Division by zero"
        except Exception as e:
            return f"Error calculating '{expression}': {str(e)}"


def get_tool_descriptions(tools: list) -> str:
    """
    Generate a formatted description of all available tools for the LLM prompt.
    
    Args:
        tools: List of tool instances
        
    Returns:
        Formatted string describing all tools
    """
    descriptions = []
    for tool in tools:
        descriptions.append(f"- {tool.name}: {tool.description}")
    return "\n".join(descriptions)


def get_tool_by_name(tools: list, name: str):
    """
    Find a tool by its name.
    
    Args:
        tools: List of tool instances
        name: Name of the tool to find
        
    Returns:
        The tool instance or None if not found
    """
    for tool in tools:
        if tool.name == name:
            return tool
    return None


if __name__ == "__main__":
    # Simple test
    print("Testing Calculator Tool...")
    calc = CalculatorTool()
    
    test_cases = [
        "2 + 2",
        "10 * 5",
        "100 / 4",
        "2 ** 8",
        "sqrt(16)",
        "sin(pi/2)",
    ]
    
    for expr in test_cases:
        result = calc.execute(expr)
        print(f"  {expr} = {result}")
    
    print("\nTesting RAG Tool...")
    print("(Requires faiss_index to exist)")
    try:
        FAISS_INDEX_PATH = "faiss_index"
        if os.path.exists(FAISS_INDEX_PATH):
            embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
            vectorstore = FAISS.load_local(
                FAISS_INDEX_PATH, embeddings, allow_dangerous_deserialization=True
            )
            rag_tool = RAGTool(vectorstore, embeddings)
            result = rag_tool.execute("What is the return process?")
            print(f"  Query result: {result[:150]}...")
        else:
            print(f"  FAISS index not found at {FAISS_INDEX_PATH}")
    except Exception as e:
        print(f"  Could not test RAG tool: {e}")
