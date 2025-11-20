# Simple Agentic RAG System

A minimal implementation of an agentic AI system using the ReAct (Reasoning + Acting) pattern. This is a **learning project** to understand how modern AI agents work.

## What is This?

This project converts a traditional RAG (Retrieval Augmented Generation) system into an **agentic system** that can:
- 🧠 **Reason** through problems step-by-step
- 🛠️ **Use tools** to gather information and perform tasks
- 🔄 **Iterate** until it has enough information to answer
- 📝 **Show its work** with transparent reasoning traces

## Quick Start

### Prerequisites

1. **Ollama** running with Llama 3.1:
   ```bash
   ollama serve
   ollama pull llama3.1
   ```

2. **Multimodal RAG index** (if using RAG tool):
   ```bash
   python multimodal_rag.py
   ```

3. **Python dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

### Run the Agent

```bash
# Interactive mode
python agent_cli.py

# With verbose reasoning traces
python agent_cli.py --verbose
```

## Example Usage

```
You: What is 25 * 4?

Agent thinking...

Agent: 100

---

You: What is the return process?

Agent thinking...

Agent: The return process must be initiated within 30 days of purchase...
```

## How It Works

### The ReAct Pattern

```
User Query → THINK → ACT → OBSERVE → THINK → ... → FINISH
```

1. **THINK**: Agent reasons about what to do next
2. **ACT**: Agent chooses and executes a tool
3. **OBSERVE**: Agent sees the result
4. **Repeat** until task is complete

### Available Tools

| Tool | Description | Example |
|------|-------------|---------|
| `search_knowledge_base` | Search documents and images | "What is the return policy?" |
| `calculate` | Perform math calculations | "What is 15 * 8?" |

## Files

- **`tools.py`** - Tool definitions (RAG + Calculator)
- **`simple_agent.py`** - Core agent with ReAct loop
- **`agent_cli.py`** - Interactive command-line interface

**Total**: ~350 lines of code

## Commands

While in the CLI:
- Type any question to ask the agent
- `trace` - Show reasoning steps from last query
- `help` - Display help message
- `quit` - Exit

## Programmatic Usage

```python
from simple_agent import SimpleAgent
from tools import RAGTool, CalculatorTool
from multimodal_rag import MultimodalRAG
from lang_chain_llm import LLaMa

# Setup
rag = MultimodalRAG()
llm = LLaMa()
tools = [RAGTool(rag), CalculatorTool()]

# Create agent
agent = SimpleAgent(tools, llm)

# Ask a question
answer = agent.run("What is 25 * 4?", verbose=True)
print(answer)

# Get reasoning trace
trace = agent.get_reasoning_trace()
```

## Testing

```bash
# Test tools independently
python tools.py

# Test agent (requires Ollama)
python simple_agent.py
```

## Troubleshooting

### "500 Server Error" from Ollama

**Solution**: Make sure Ollama is running
```bash
ollama serve
```

### "No index found"

**Solution**: Create the multimodal index first
```bash
python multimodal_rag.py
```

### Agent doesn't follow format

**Solution**: The LLM might need better prompting. Try:
- Using a larger model (llama3.1:70b)
- Adding more examples to the prompt
- Adjusting temperature settings

## Learning Resources

- **[ReAct Paper](https://arxiv.org/abs/2210.03629)** - Original research
- **[Implementation Plan](implementation_plan.md)** - Detailed design doc
- **[Walkthrough](walkthrough.md)** - Complete implementation guide

## Next Steps

Once you understand the basics:

1. **Add new tools** (web search, image generation, etc.)
2. **Improve memory** (vector store for conversations)
3. **Add planning** (agent creates multi-step plans)
4. **Multi-agent** (multiple agents working together)

## Architecture

```
User Query
    ↓
SimpleAgent (ReAct Loop)
    ↓
LLM (Llama 3.1) ←→ Memory
    ↓
Tool Selection
    ↓
┌─────────────┬──────────────┐
│  RAG Tool   │ Calculator   │
└─────────────┴──────────────┘
    ↓
Observation → Back to LLM
```

## License

This is a learning project. Feel free to use and modify as needed.

## Credits

Built following the ReAct pattern from the paper "ReAct: Synergizing Reasoning and Acting in Language Models" (Yao et al., 2022).
