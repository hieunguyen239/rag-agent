#!/usr/bin/env python3
"""
Simple Agent CLI

Interactive command-line interface for the agentic RAG system.
Allows users to chat with the agent and see its reasoning process.
"""

import sys
import os
from simple_agent import SimpleAgent
from tools import RAGTool, CalculatorTool
from lang_chain_llm import LLaMa
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings


def print_banner():
    """Print welcome banner."""
    print("\n" + "="*60)
    print("  SIMPLE AGENTIC RAG - Interactive CLI")
    print("="*60)
    print("\nThis agent can:")
    print("  • Search the knowledge base for information")
    print("  • Perform mathematical calculations")
    print("  • Reason through multi-step tasks")
    print("\nCommands:")
    print("  'quit' or 'exit' - Exit the program")
    print("  'trace' - Show reasoning trace from last query")
    print("  'help' - Show this help message")
    print("\n" + "="*60 + "\n")


def print_help():
    """Print help message."""
    print("\nAvailable commands:")
    print("  quit, exit - Exit the program")
    print("  trace - Show detailed reasoning from last query")
    print("  help - Show this message")
    print("\nJust type your question to ask the agent!\n")


def print_trace(agent):
    """Print the reasoning trace from the last query."""
    trace = agent.get_reasoning_trace()
    
    if not trace:
        print("\nNo reasoning trace available yet. Ask a question first!\n")
        return
    
    print("\n" + "-"*60)
    print("REASONING TRACE")
    print("-"*60)
    
    for i, step in enumerate(trace, 1):
        print(f"\nStep {i}:")
        print(f"  Thought: {step['thought'][:200]}...")
        print(f"  Action: {step['action']}")
        print(f"  Input: {step['input']}")
        print(f"  Result: {step['observation'][:200]}...")
    
    print("-"*60 + "\n")


def main():
    """Main CLI loop."""
    
    # Check for verbose flag
    verbose = "--verbose" in sys.argv or "-v" in sys.argv
    
    # Print banner
    print_banner()
    
    # Initialize components
    print("Loading RAG system...")
    try:
        FAISS_INDEX_PATH = "faiss_index"
        if not os.path.exists(FAISS_INDEX_PATH):
            print(f"Error: FAISS index not found at '{FAISS_INDEX_PATH}'")
            print("\nThe text-based RAG index doesn't exist yet.")
            print("Run the following to create it:")
            print("  python lang_chain_llm.py")
            print("\nOr make sure you have a PDF file and it will create the index automatically.\n")
            return 1
        
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        vectorstore = FAISS.load_local(
            FAISS_INDEX_PATH, embeddings, allow_dangerous_deserialization=True
        )
        print("✓ FAISS index loaded")
        
    except Exception as e:
        print(f"Error: Could not load RAG system: {e}\n")
        return 1
    
    print("Initializing LLM...")
    try:
        llm = LLaMa()
        print("✓ LLM ready")
    except Exception as e:
        print(f"Error: Could not initialize LLM: {e}")
        print("\nMake sure Ollama is running with the llama3.1 model.")
        print("Run 'ollama serve' and 'ollama pull llama3.1'\n")
        return 1
    
    # Create tools
    tools = [
        RAGTool(vectorstore, embeddings),
        CalculatorTool()
    ]
    
    # Create agent
    agent = SimpleAgent(tools, llm)
    
    print("Agent ready! Type your question or 'help' for commands.\n")
    
    # Main interaction loop
    while True:
        try:
            # Get user input
            user_input = input("You: ").strip()
            
            # Handle empty input
            if not user_input:
                continue
            
            # Handle commands
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("\nGoodbye! 👋\n")
                break
            
            if user_input.lower() == 'help':
                print_help()
                continue
            
            if user_input.lower() == 'trace':
                print_trace(agent)
                continue
            
            # Process query with agent
            print("\nAgent thinking...\n")
            answer = agent.run(user_input, verbose=verbose)
            
            # Print answer
            print(f"\n{'='*60}")
            print("Agent:", answer)
            print(f"{'='*60}\n")
            
            # Optionally show trace hint
            if not verbose:
                print("(Type 'trace' to see reasoning steps)\n")
            
        except KeyboardInterrupt:
            print("\n\nGoodbye! 👋\n")
            break
        except Exception as e:
            print(f"\nError: {e}\n")
            continue
    
    return 0


if __name__ == "__main__":
    exit(main())
