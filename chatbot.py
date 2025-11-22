# rag_ollama_langchain.py

import os
import glob
from requests import post as rpost
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# LangChain core and components
from langchain_core.language_models.llms import LLM
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.runnables import RunnablePassthrough
from langchain_core.messages import HumanMessage
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


# -----------------------------
# 1) API function to call Ollama
# -----------------------------
def call_llama(prompt: str) -> str:
    """
    Calls a local Ollama model with a prompt and returns the generated response text.
    Ensure Ollama is running: `ollama serve` and that the configured model is available.
    """
    # Get configuration from environment variables
    model = os.getenv("OLLAMA_MODEL", "llama3.1")
    base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    
    headers = {"Content-Type": "application/json"}
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
    }
    response = rpost(f"{base_url}/api/generate", headers=headers, json=payload)
    response.raise_for_status()
    data = response.json()
    return data["response"]


# --------------------------------------
# 2) Custom LangChain LLM wrapper class
# --------------------------------------
class LLaMa(LLM):
    """
    Minimal LangChain-compatible LLM that uses the local Ollama endpoint via call_llama.
    """

    def _call(self, prompt: str, **kwargs) -> str:
        return call_llama(prompt)

    @property
    def _llm_type(self) -> str:
        # Arbitrary identifier for this LLM type
        return "llama-3.1-8b"


# ---------------------------------
# 3) Build retriever from PDF
# ---------------------------------
FAISS_INDEX_PATH = "faiss_index"

# HuggingFace sentence-transformers embeddings; good default for quick RAG demos
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

if os.path.exists(FAISS_INDEX_PATH):
    # Load the FAISS index from disk
    print("Loading FAISS index from disk.")
    vectorstore = FAISS.load_local(
        FAISS_INDEX_PATH, embeddings, allow_dangerous_deserialization=True
    )
else:
    # Create the FAISS index from all PDFs in the directory
    print("Creating FAISS index from all PDFs in the directory.")
    
    # Find all PDF files in the current directory
    pdf_files = glob.glob("*.pdf")
    
    if not pdf_files:
        raise FileNotFoundError("No PDF files found in the current directory.")
    
    print(f"Found {len(pdf_files)} PDF file(s): {', '.join(pdf_files)}")
    
    # Load all PDFs
    all_documents = []
    for pdf_file in pdf_files:
        print(f"Loading {pdf_file}...")
        loader = PyPDFLoader(pdf_file)
        all_documents.extend(loader.load())
    
    print(f"Loaded {len(all_documents)} pages total from all PDFs.")

    # Split the documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_documents(all_documents)
    
    print(f"Split into {len(texts)} text chunks.")

    # Build FAISS index
    vectorstore = FAISS.from_documents(texts, embeddings)

    # Save the FAISS index to disk
    print(f"Saving FAISS index to {FAISS_INDEX_PATH}...")
    vectorstore.save_local(FAISS_INDEX_PATH)
    print("FAISS index created and saved successfully!")

# Get a retriever
retriever = vectorstore.as_retriever(k=5)


# ----------------------------
# 4) Prompt template for the LLM
# ----------------------------
qa_template = """
You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question.
If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.

<context>
{context}
</context>
"""

qa_prompt = ChatPromptTemplate.from_messages([
    ("system", qa_template),
    MessagesPlaceholder("messages"),
])


# ------------------------------------------------
# 5) Document chain + retrieval orchestration
# ------------------------------------------------
# Takes retrieved docs and "stuffs" them into the prompt for the LLM
document_chain = create_stuff_documents_chain(LLaMa(), qa_prompt)

def parse_retriever_input(params):
    # Extract the latest human message to drive the retriever
    return params["messages"][-1].content

# Build a Runnable pipeline:
# - Resolve 'context' by passing the latest message into the retriever
# - Generate 'answer' by calling the document chain with the context
retrieval_chain = (
    RunnablePassthrough
    .assign(context=parse_retriever_input | retriever)
    .assign(answer=document_chain)
)


# ----------------------------
# 6) Simple test invocation
# ----------------------------
if __name__ == "__main__":
    user_query = "fuel tank capacity of the Triumph Tiger Sport?"
    response = retrieval_chain.invoke({
        "messages": [HumanMessage(user_query)]
    })
    # The response contains both 'context' and 'answer' keys; print the model's answer
    print(response["answer"])