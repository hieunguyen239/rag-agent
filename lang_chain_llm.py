# rag_huggingface_langchain.py

import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# LangChain core and components
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from langchain_core.prompts import PromptTemplate
from langchain_classic.chains import RetrievalQA
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# --------------------------------------
# 2) Initialize Hugging Face LLM
# --------------------------------------
# Using TinyLlama - a small, ungated instruction-tuned model
# This downloads the model locally but gives you full control
# For Llama 3, you need to request access at: https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct
print("Loading model... This may take a while on first run.")
model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    torch_dtype="auto"
)

pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=256,
    temperature=0.1,
    do_sample=True,
    top_p=0.95,
)

llm = HuggingFacePipeline(pipeline=pipe)


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
prompt_template = """You are a helpful assistant. Use ONLY the information from the context below to answer the question.

IMPORTANT: The context contains tables with two columns - "Trident" and "Tiger Sport". When asked about "Tiger Sport", use values from the "Tiger Sport" column (the RIGHT column), NOT the "Trident" column.

Context:
{context}

Question: {question}

Answer:"""

qa_prompt = PromptTemplate(
    template=prompt_template, 
    input_variables=["context", "question"]
)


# ------------------------------------------------
# 5) Create the RetrievalQA chain
# ------------------------------------------------
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True,
    chain_type_kwargs={"prompt": qa_prompt}
)


# ----------------------------
# 6) Simple test invocation
# ----------------------------
if __name__ == "__main__":
    user_query = "fuel tank capacity of the Triumph Tiger Sport?"
    response = qa_chain.invoke({"query": user_query})
    
    print("\n" + "="*50)
    print("QUESTION:", user_query)
    print("="*50)
    print("\nANSWER:")
    print(response["result"])
    print("="*50)