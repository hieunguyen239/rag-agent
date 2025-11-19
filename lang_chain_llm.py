# rag_ollama_langchain.py

import os
from requests import post as rpost

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
    Ensure Ollama is running: `ollama serve` and that model `llama3.1` is available.
    """
    headers = {"Content-Type": "application/json"}
    payload = {
        "model": "llama3",
        "prompt": prompt,
        "stream": False,
    }
    response = rpost("http://192.168.2.5:11434/api/generate", headers=headers, json=payload)
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
    # Create the FAISS index from the PDF
    print("Creating FAISS index from PDF.")
    # Load the PDF
    loader = PyPDFLoader("TRIUMPH_2023_TIGER_SPORT_-TRIDENT_660_ENG.pdf")
    documents = loader.load()

    # Split the document into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_documents(documents)

    # Build FAISS index
    vectorstore = FAISS.from_documents(texts, embeddings)

    # Save the FAISS index to disk
    vectorstore.save_local(FAISS_INDEX_PATH)

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