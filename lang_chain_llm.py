# rag_ollama_langchain.py

from requests import post as rpost

# LangChain core and components
from langchain_core.language_models.llms import LLM
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.runnables import RunnablePassthrough
from langchain_core.messages import HumanMessage


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
        "model": "llama3.1",
        "prompt": prompt,
        "stream": False,
    }
    response = rpost("http://172.20.96.1:11434/api/generate", headers=headers, json=payload)
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
# 3) Build retriever (FAISS + HF)
# ---------------------------------
# Example documents (replace or expand for your domain)
documents = [
    {"content": "What is your return policy? We accept returns within 30 days if items are unused and in original packaging."},
    {"content": "How long does shipping take? Standard shipping takes 3-5 business days; express options are available at checkout."},
    {"content": "Do you offer refunds for damaged items? No, please contact support with photos within 7 days for a replacement or refund."},
    {"content": "Can I change my order after placing it? Orders can be updated within 24 hours before they ship."},
    {"content": "What payment methods do you accept? We accept major credit cards, PayPal, and local e-wallets."},
]

texts = [doc["content"] for doc in documents]

# HuggingFace sentence-transformers embeddings; good default for quick RAG demos
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Build FAISS index and get a retriever
retriever = FAISS.from_texts(texts, embeddings).as_retriever(k=5)


# ----------------------------
# 4) Prompt template for the LLM
# ----------------------------
faq_template = """
You are a chat agent for my E-Commerce Company. As a chat agent, it is your duty to help the human with their inquiry and make them a happy customer.
Help them, using the following context:
<context>
{context}
</context>
"""

faq_prompt = ChatPromptTemplate.from_messages([
    ("system", faq_template),
    MessagesPlaceholder("messages"),
])


# ------------------------------------------------
# 5) Document chain + retrieval orchestration
# ------------------------------------------------
# Takes retrieved docs and "stuffs" them into the prompt for the LLM
document_chain = create_stuff_documents_chain(LLaMa(), faq_prompt)

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
    user_query = "I received a damaged item. I want my money back."
    response = retrieval_chain.invoke({
        "messages": [HumanMessage(user_query)]
    })
    # The response contains both 'context' and 'answer' keys; print the model's answer
    print(response["answer"])
