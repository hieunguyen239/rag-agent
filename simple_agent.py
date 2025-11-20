"""
Simple Agent with ReAct Pattern

This module implements a minimal agentic system using the ReAct (Reasoning + Acting) pattern.
The agent can think, choose tools, execute them, and observe results in a loop.
"""

from typing import List, Dict, Any, Optional, Tuple
from tools import get_tool_descriptions, get_tool_by_name
import re


class SimpleAgent:
    """
    A simple agent that uses the ReAct pattern to solve tasks.
    
    The agent follows this loop:
    1. THINK - Reason about what to do next
    2. ACT - Choose and execute a tool
    3. OBSERVE - Process the tool's result
    4. Repeat until task is complete
    """
    
    def __init__(self, tools: list, llm):
        """
        Initialize the agent.
        
        Args:
            tools: List of tool instances the agent can use
            llm: Language model instance (must have _call method)
        """
        self.tools = tools
        self.llm = llm
        self.memory = []  # Stores reasoning history
        self.max_iterations = 2
        
    def run(self, user_query: str, verbose: bool = False) -> str:
        """
        Run the agent to answer a user query.
        
        Args:
            user_query: The user's question or task
            verbose: If True, print reasoning steps
            
        Returns:
            The final answer
        """
        self.memory = []  # Reset memory for new query
        
        if verbose:
            print(f"\n{'='*60}")
            print(f"USER QUERY: {user_query}")
            print(f"{'='*60}\n")
        
        for iteration in range(self.max_iterations):
            if verbose:
                print(f"--- Iteration {iteration + 1} ---")
            
            # THINK: Generate reasoning about next action
            thought = self.think(user_query)
            
            if verbose:
                print(f"THOUGHT: {thought}")
            
            # ACT: Parse the thought to get action and input
            action, tool_input, is_final = self.choose_action(thought)
            
            if verbose:
                print(f"ACTION: {action}")
                if not is_final:
                    print(f"INPUT: {tool_input}")
            
            # Check if we're done
            if is_final:
                final_answer = self.extract_final_answer(thought)
                if verbose:
                    print(f"\n{'='*60}")
                    print(f"FINAL ANSWER: {final_answer}")
                    print(f"{'='*60}\n")
                return final_answer
            
            # OBSERVE: Execute the tool and get result
            observation = self.execute_tool(action, tool_input)
            
            if verbose:
                print(f"OBSERVATION: {observation[:200]}{'...' if len(observation) > 200 else ''}")
                print()
            
            # REMEMBER: Store this step
            self.memory.append({
                "thought": thought,
                "action": action,
                "input": tool_input,
                "observation": observation
            })
        
        # Max iterations reached
        return self.generate_fallback_answer()
    
    def think(self, user_query: str) -> str:
        """
        Use the LLM to reason about what to do next.
        
        Args:
            user_query: The original user query
            
        Returns:
            The LLM's reasoning (thought)
        """
        prompt = self.build_prompt(user_query)
        response = self.llm._call(prompt)
        return response
    
    def build_prompt(self, user_query: str) -> str:
        """
        Build the ReAct prompt for the LLM.
        
        Args:
            user_query: The user's question
            
        Returns:
            Formatted prompt string
        """
        tool_descriptions = get_tool_descriptions(self.tools)
        
        # Build memory context
        memory_text = ""
        if self.memory:
            memory_text = "\n\nPrevious steps:\n"
            for i, step in enumerate(self.memory, 1):
                memory_text += f"{i}. ACTION: {step['action']}\n"
                memory_text += f"   INPUT: {step['input']}\n"
                memory_text += f"   RESULT: {step['observation'][:150]}...\n"
        
        prompt = f"""You are an AI agent that can use tools to answer questions.

Available Tools:
{tool_descriptions}

Instructions:
- Think step by step about what you need to do
- Choose the most appropriate tool for each step
- When you have enough information, provide the final answer

Format your response EXACTLY as follows:

THOUGHT: [your reasoning about what to do next]
ACTION: [tool_name OR "FINISH"]
INPUT: [input for the tool, or leave empty if ACTION is FINISH]
ANSWER: [only include this if ACTION is FINISH - provide the final answer to the user]

Example 1:
THOUGHT: I need to search for information about the return process
ACTION: search_knowledge_base
INPUT: return process policy

Example 2:
THOUGHT: I have all the information needed to answer the question
ACTION: FINISH
INPUT: 
ANSWER: The return process requires customers to initiate returns within 30 days.

Question: {user_query}{memory_text}

Your response:"""
        
        return prompt
    
    def choose_action(self, thought: str) -> Tuple[str, str, bool]:
        """
        Parse the LLM's thought to extract action and input.
        
        Args:
            thought: The LLM's response
            
        Returns:
            Tuple of (action_name, tool_input, is_final)
        """
        # Extract ACTION
        action_match = re.search(r'ACTION:\s*(.+?)(?:\n|$)', thought, re.IGNORECASE)
        action = action_match.group(1).strip() if action_match else "FINISH"
        
        # Check if this is the final action
        if action.upper() == "FINISH":
            return "FINISH", "", True
        
        # Extract INPUT
        input_match = re.search(r'INPUT:\s*(.+?)(?:\n|$)', thought, re.IGNORECASE)
        tool_input = input_match.group(1).strip() if input_match else ""
        
        return action, tool_input, False
    
    def execute_tool(self, action: str, tool_input: str) -> str:
        """
        Execute the chosen tool with the given input.
        
        Args:
            action: Name of the tool to execute
            tool_input: Input for the tool
            
        Returns:
            The tool's output
        """
        tool = get_tool_by_name(self.tools, action)
        
        if tool is None:
            return f"Error: Tool '{action}' not found. Available tools: {', '.join([t.name for t in self.tools])}"
        
        try:
            result = tool.execute(tool_input)
            return result
        except Exception as e:
            return f"Error executing {action}: {str(e)}"
    
    def extract_final_answer(self, thought: str) -> str:
        """
        Extract the final answer from the LLM's thought.
        
        Args:
            thought: The LLM's final response
            
        Returns:
            The extracted answer
        """
        # Try to extract ANSWER field
        answer_match = re.search(r'ANSWER:\s*(.+)', thought, re.IGNORECASE | re.DOTALL)
        if answer_match:
            return answer_match.group(1).strip()
        
        # Fallback: return the whole thought
        return thought
    
    def generate_fallback_answer(self) -> str:
        """
        Generate an answer when max iterations are reached.
        
        Returns:
            A fallback answer based on observations
        """
        if not self.memory:
            return "I apologize, but I couldn't find an answer to your question."
        
        # Use the last observation as the answer
        last_observation = self.memory[-1]['observation']
        return f"Based on my search: {last_observation}"
    
    def get_reasoning_trace(self) -> List[Dict[str, Any]]:
        """
        Get the full reasoning trace for debugging/transparency.
        
        Returns:
            List of reasoning steps
        """
        return self.memory


if __name__ == "__main__":
    # Simple test
    print("Testing Simple Agent...")
    print("(This requires the FAISS RAG system to be set up)")
    
    try:
        from lang_chain_llm import LLaMa
        from tools import RAGTool, CalculatorTool
        from langchain_community.vectorstores import FAISS
        from langchain_huggingface import HuggingFaceEmbeddings
        import os
        
        # Initialize components
        FAISS_INDEX_PATH = "faiss_index"
        if not os.path.exists(FAISS_INDEX_PATH):
            print(f"\nError: FAISS index not found at '{FAISS_INDEX_PATH}'")
            print("Run 'python lang_chain_llm.py' first to create it.\n")
            exit(1)
        
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        vectorstore = FAISS.load_local(
            FAISS_INDEX_PATH, embeddings, allow_dangerous_deserialization=True
        )
        llm = LLaMa()
        tools = [RAGTool(vectorstore, embeddings), CalculatorTool()]
        
        # Create agent
        agent = SimpleAgent(tools, llm)
        
        # Test query
        print("\nTest 1: Simple RAG query")
        answer = agent.run("What is the return process?", verbose=True)
        print(f"\nFinal Answer: {answer}\n")
        
        print("\n" + "="*60 + "\n")
        
        # Test calculation
        print("Test 2: Calculator")
        answer = agent.run("What is 25 * 4?", verbose=True)
        print(f"\nFinal Answer: {answer}\n")
        
    except Exception as e:
        print(f"Could not run test: {e}")
        print("\nTo test the agent, make sure:")
        print("1. The faiss_index exists (run lang_chain_llm.py first)")
        print("2. Ollama is running with llama3.1 model")
