# rag_ollama_langchain.py

import os
import glob

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# LangChain core and components
from langchain_core.language_models.llms import LLM
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.runnables import RunnablePassthrough
from langchain_core.messages import HumanMessage
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

# -----------------------------
# 1) Initialize Google Gemini
# -----------------------------
# Reads GOOGLE_API_KEY from environment automaticallly
llm = ChatGoogleGenerativeAI(model="gemini-flash-latest")


from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone

# ---------------------------------
# 3) Build retriever from PDF
# ---------------------------------
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")

# Google Generative AI Embeddings
embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")

def get_vectorstore():
    # Attempt to use Pinecone if credentials are provided
    if PINECONE_API_KEY and PINECONE_INDEX_NAME:
        print(f"Checking Pinecone index: {PINECONE_INDEX_NAME}...")
        try:
            pc = Pinecone(api_key=PINECONE_API_KEY)
            index = pc.Index(PINECONE_INDEX_NAME)
            stats = index.describe_index_stats()
            
            if stats['total_vector_count'] > 0:
                print(f"Using Pinecone Cloud Vector Store (Found {stats['total_vector_count']} vectors).")
                return PineconeVectorStore(index_name=PINECONE_INDEX_NAME, embedding=embeddings)
            else:
                print("Pinecone index is empty. Triggering initial upload...")
                return None
        except Exception as e:
            raise RuntimeError(f"Error connecting to Pinecone: {e}")

    raise ValueError("PINECONE_API_KEY and PINECONE_INDEX_NAME must be set in the environment.")

vectorstore = get_vectorstore()

if vectorstore is None:
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

    # Build vectorstore and save
    if PINECONE_API_KEY and PINECONE_INDEX_NAME:
        print(f"Upserting to Pinecone index: {PINECONE_INDEX_NAME}...")
        vectorstore = PineconeVectorStore.from_documents(
            texts, 
            embeddings, 
            index_name=PINECONE_INDEX_NAME
        )
        print("Documents successfully upserted to Pinecone!")
    else:
        raise ValueError("Cannot upload to Pinecone: PINECONE_API_KEY or PINECONE_INDEX_NAME is missing.")

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
document_chain = create_stuff_documents_chain(llm, qa_prompt)

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