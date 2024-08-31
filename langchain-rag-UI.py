import os
from urllib.request import urlretrieve
import numpy as np
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.llms import HuggingFacePipeline
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import gradio as gr

# Load PDF files from the specified local directory
pdf_loader = PyPDFDirectoryLoader("./Dataset/")
pdf_documents = pdf_loader.load()

# Split the loaded documents into smaller chunks to facilitate better processing
document_splitter = RecursiveCharacterTextSplitter(
    chunk_size=700,
    chunk_overlap=50,
)
document_chunks = document_splitter.split_documents(pdf_documents)

# Function to calculate the average length of documents in characters
calculate_avg_doc_length = lambda docs: sum([len(doc.page_content) for doc in docs]) // len(docs)
avg_length_before_split = calculate_avg_doc_length(pdf_documents)
avg_length_after_split = calculate_avg_doc_length(document_chunks)

# Print the details before and after splitting the documents
print(f'Before splitting, {len(pdf_documents)} documents were loaded with an average length of {avg_length_before_split} characters.')
print(f'After splitting, {len(document_chunks)} document chunks were created with an average length of {avg_length_after_split} characters.')

# Initialize the embedding model from Hugging Face
embedding_model = HuggingFaceBgeEmbeddings(
    model_name="BAAI/bge-small-en-v1.5",  # Alternatively, "sentence-transformers/all-MiniLM-l6-v2" for a lighter model.
    model_kwargs={'device': 'cpu'}, 
    encode_kwargs={'normalize_embeddings': True}
)

# Generate an embedding for a sample document chunk
sample_embedding_vector = np.array(embedding_model.embed_query(document_chunks[0].page_content))
print("Sample embedding of a document chunk: ", sample_embedding_vector)
print("Size of the embedding: ", sample_embedding_vector.shape)

# Create a vector store (FAISS) from the document chunks and their embeddings
faiss_vector_store = FAISS.from_documents(document_chunks, embedding_model)

# Initialize a retriever to find the 3 most relevant documents using similarity search
similarity_retriever = faiss_vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 3})

# Initialize the LLM (Ollama model)
llm_model = Ollama(model="mistral")

# Define the prompt template for generating responses
answer_prompt_template = """Use the following pieces of context to answer the question at the end. Please follow the following rules:
1. If you don't know the answer, don't try to make up an answer. Just say "I can't find the final answer but you may want to check the following links".
2. If you find the answer, write the answer in a concise way with five sentences maximum with the product link.

{context}

Question: {question}

Helpful Answer:
"""

# Create a PromptTemplate instance
custom_prompt_template = PromptTemplate(
 template=answer_prompt_template, input_variables=["context", "question"]
)

# Create a RetrievalQA chain with the specified LLM, retriever, and prompt template
retrieval_qa_chain = RetrievalQA.from_chain_type(
    llm=llm_model,
    chain_type="stuff",
    retriever=similarity_retriever,
    return_source_documents=True,
    chain_type_kwargs={"prompt": custom_prompt_template}
)

# Define the function to generate responses based on the user's question
def generate_response(user_question, chat_history):
    response_result = retrieval_qa_chain.invoke({"query": user_question})  # Use the user's question
    return response_result["result"]

# Create the Gradio interface for the chatbot
gr.ChatInterface(
    generate_response,
    chatbot=gr.Chatbot(height=500),
    textbox=gr.Textbox(placeholder="Ask me questions related to outfits for any occasion", container=False, scale=7),
    title="ShopGenius",
    examples=["What kind of dresses can be worn on a beach vacation?", "What are the colors I can wear on a summer date?"],
    cache_examples=True,
    retry_btn=None,
).launch(share=True)
