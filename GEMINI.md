# Project Overview

This project is a Python-based application that demonstrates a Retrieval-Augmented Generation (RAG) pipeline using the LangChain library. It's designed to answer user queries based on a predefined set of documents.

The project uses the following technologies:
- **Ollama:** For running a local LLaMA model (`llama3.1`).
- **LangChain:** For building the RAG pipeline, including prompt templates, document chains, and retrieval mechanisms.
- **FAISS:** For efficient similarity search in a vector store of document embeddings.
- **HuggingFace Sentence-Transformers:** For generating text embeddings.

The core logic is contained in `lang_chain_llm.py`, which sets up the entire RAG pipeline from document loading to answer generation. The `call_llama.py` script provides a standalone function for interacting with the Ollama API.

# Building and Running

## 1. Install Dependencies

First, install the required Python packages using pip:

```bash
pip install -r requirements.txt
```

## 2. Set up Ollama on Windows

This project is configured to connect to an Ollama instance running on the Windows host machine, from within the WSL environment.

First, ensure Ollama is running on your Windows machine. You can start it by finding "Ollama" in the Start Menu or by running `ollama serve` in a Windows Command Prompt or PowerShell.

If you don't have the `llama3.1` model, you can pull it using the following command on Windows:
```bash
ollama pull llama3.1
```

The Python scripts in this project are hardcoded to connect to Ollama at `http://172.20.96.1:11434`. This IP address is assumed to be the address of your Windows host, as seen from WSL. If your configuration is different, you may need to update the IP address in `lang_chain_llm.py` and `call_llama.py`.

## 3. Run the Application

The main application logic is in `lang_chain_llm.py`. To run the example, execute the script directly:

```bash
python3 lang_chain_llm.py
```

This will run a predefined query through the RAG pipeline and print the model's answer to the console.

# Development Conventions

- **Coding Style:** The code is written in Python and generally follows standard Python conventions. It includes type hints for function signatures.
- **Modularity:** The `lang_chain_llm.py` script is organized into logical sections for different parts of the RAG pipeline, such as API interaction, LLM wrapping, retriever setup, prompt templating, and chain orchestration.
- **Testing:** The `if __name__ == "__main__"` block in `lang_chain_llm.py` serves as a simple test case to demonstrate the functionality of the RAG pipeline.
