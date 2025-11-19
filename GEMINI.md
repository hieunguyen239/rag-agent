# Project Overview

This project is a Python-based application that demonstrates a Retrieval-Augmented Generation (RAG) pipeline using the LangChain library. It's designed to answer user queries based on a predefined set of documents.

The project uses the following technologies:
- **Hugging Face Transformers:** For running LLMs locally (currently using TinyLlama, can be switched to Llama 3 with access)
- **LangChain:** For building the RAG pipeline, including prompt templates, document chains, and retrieval mechanisms.
- **FAISS:** For efficient similarity search in a vector store of document embeddings.
- **HuggingFace Sentence-Transformers:** For generating text embeddings.
- **PyTorch:** For running the transformer models locally.

The core logic is contained in `lang_chain_llm.py`, which sets up the entire RAG pipeline from document loading to answer generation.

# Building and Running

## 1. Install Dependencies

First, install the required Python packages using pip:

```bash
pip install -r requirements.txt
```

## 2. Set up Hugging Face API Token

Create a `.env` file in the project root with your Hugging Face API token:

```bash
HUGGINGFACEHUB_API_TOKEN=your_token_here
```

You can get a token from https://huggingface.co/settings/tokens

## 3. Run the Application

The main application logic is in `lang_chain_llm.py`. To run the example, execute the script directly:

```bash
python3 lang_chain_llm.py
```

This will run a predefined query through the RAG pipeline and print the model's answer to the console.

**Note:** The first run will download the model (TinyLlama, ~2GB) which may take several minutes. Subsequent runs will use the cached model.

For instructions on using Llama 3 models instead, see `HUGGINGFACE_SETUP.md`.

# Development Conventions

- **Coding Style:** The code is written in Python and generally follows standard Python conventions. It includes type hints for function signatures.
- **Modularity:** The `lang_chain_llm.py` script is organized into logical sections for different parts of the RAG pipeline, such as API interaction, LLM wrapping, retriever setup, prompt templating, and chain orchestration.
- **Testing:** The `if __name__ == "__main__"` block in `lang_chain_llm.py` serves as a simple test case to demonstrate the functionality of the RAG pipeline.
