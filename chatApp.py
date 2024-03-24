from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import LlamaCpp
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.document_loaders import PyPDFDirectoryLoader, csv_loader
import sys

# Initialize PDF loader (adjust according to your setup)
loader = PyPDFDirectoryLoader("content/")
data = loader.load()

# Split text into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=20)
text_chunks = text_splitter.split_documents(data)

# Create embeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vector_store = FAISS.from_documents(text_chunks, embedding=embeddings)

# Initialize the Llama model
llm = LlamaCpp(
    streaming=True,
    model_path="MODEL/mistral-7b-instruct-v0.1.Q4_0.gguf",
    temperature=0.75,
    top_p=1,
    verbose=True,
    n_ctx=4096
)

# Setup RetrievalQA
qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=vector_store.as_retriever(search_kwargs={"k": 2}))

# Interactive mode
while True:
    user_input = input("Input Prompt: ")
    if user_input == 'exit':
        print('Exiting')
        sys.exit()
    if user_input == '':
        continue
    result = qa.invoke({'query': user_input})
    print(f"Answer: {result}")
