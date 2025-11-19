# Hugging Face Integration - Summary

## What Was Changed

Successfully migrated from local Ollama (Llama 3) to **Hugging Face Transformers** for running LLMs.

### Key Changes:

1. **Environment Configuration** (`.env` file):
   - Created `.env` file to store `HUGGINGFACEHUB_API_TOKEN`
   - Added `python-dotenv` to load environment variables automatically

2. **Dependencies** (`requirements.txt`):
   - Added: `python-dotenv`, `transformers`, `torch`, `accelerate`
   - These enable local model execution via Hugging Face

3. **Code Changes** (`lang_chain_llm.py`):
   - Removed: Custom `call_llama()` function and `LLaMa` wrapper class
   - Added: Hugging Face Transformers integration using `HuggingFacePipeline`
   - Currently using: `TinyLlama/TinyLlama-1.1B-Chat-v1.0` (ungated, publicly available)

## Current Status

✅ **Working**: The application successfully runs with TinyLlama model from Hugging Face
✅ **Local Execution**: Model downloads and runs locally (no API calls needed after download)
✅ **RAG Pipeline**: Document retrieval and question-answering working

## How to Use Llama 3 Instead

To use actual Llama 3 models, you need to:

### Step 1: Request Access on Hugging Face
Visit one of these pages and click "Request Access":
- https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct (smaller, faster)
- https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct (larger, better quality)

### Step 2: Get Your Hugging Face Token
1. Go to https://huggingface.co/settings/tokens
2. Create a new token (or use existing one)
3. Update your `.env` file with the token

### Step 3: Update the Model in Code
Once you have access, edit `lang_chain_llm.py` line 28:

```python
# Change from:
model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

# To one of these:
model_id = "meta-llama/Llama-3.2-1B-Instruct"  # Smaller, faster
# OR
model_id = "meta-llama/Meta-Llama-3-8B-Instruct"  # Larger, better quality
```

And uncomment the `token` parameter on line 29:

```python
tokenizer = AutoTokenizer.from_pretrained(model_id, token=os.getenv("HUGGINGFACEHUB_API_TOKEN"))
```

And on line 30-34:

```python
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    token=os.getenv("HUGGINGFACEHUB_API_TOKEN"),  # Add this line back
    device_map="auto",
    torch_dtype="auto"
)
```

### Step 4: Run the Application
```bash
source .venv/bin/activate
python3 lang_chain_llm.py
```

## System Requirements

- **GPU**: Recommended for faster inference (CUDA-capable GPU)
- **RAM**: 
  - TinyLlama (1.1B): ~4GB
  - Llama 3.2 (1B): ~4GB
  - Llama 3 (8B): ~16GB+
- **Disk Space**: ~2-20GB depending on model size

## Notes

- First run will download the model (can take several minutes)
- Subsequent runs will use the cached model
- Models are cached in `~/.cache/huggingface/`
- The current TinyLlama model works without any special access requirements
