from langchain .document_loaders import PyPDFLoader,TextLoader,UnstructuredWordDocumentLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from chromadb.config import Settings
from langchain.chains.summarize import load_summarize_chain
from langchain.llms import OpenAI
import tiktoken
import chromadb
import hashlib
import os

m = hashlib.md5()

persist_directory = 'chromadb'
embeddings = OpenAIEmbeddings(openai_api_key=os.environ['OPENAI_API_KEY'])
encoding = tiktoken.encoding_for_model('davinci')
tokenizer = tiktoken.get_encoding(encoding.name)
client = chromadb.Client(Settings(
        chroma_db_impl="duckdb+parquet",
        persist_directory="db"
))


def tk_len(text):
    token = tokenizer.encode (
        text,
        disallowed_special=()
    )
    return len(token)

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=20,
    length_function=tk_len,
    separators=['\n\n','\n',',','']
)

def Embeddings(chunks):
    vectordb = Chroma.from_documents(chunks,embeddings,persist_directory=persist_directory, collection_name='podcasts')
    vectordb.persist()


def saveFiles(filenames):
    for file in filenames:
        _,file_extension = os.path.splitext(file)
        if file_extension == ".pdf":
            loader = PyPDFLoader(file)
            docs = loader.load()
        elif file_extension == ".txt":
            loader = TextLoader(file)
            docs = loader.load()
        elif file_extension == ".docx" or file_extension == ".doc":
            loader = UnstructuredWordDocumentLoader(file)
            docs = loader.load()
        chunks = text_splitter.split_documents(docs)
        ids = []
        metadatas = []
        print('chunky',len(chunks))
        Embeddings(chunks)




def Delete_files(filename):
    metadata = {}
   
    collection_name = client.list_collections()[0].name
    collection = client.get_collection(name=collection_name)
    metadata['source'] = filename
    collection.delete(
        where=metadata
    )    


def response(query):
    vectordb = Chroma(embedding_function=embeddings, persist_directory=persist_directory, collection_name='podcasts')
    assist = RetrievalQA.from_llm(OpenAI(temperature=0, model_name="text-davinci-003"),
                                                retriever=vectordb.as_retriever(kwargs={'2'}))
    response = assist(query)
    return response['result']


def summarise():
    vectordb = Chroma(embedding_function=embeddings, persist_directory=persist_directory, collection_name='podcasts')
    llm=ChatOpenAI(temperature=0, openai_api_key=os.environ['OPENAI_API_KEY'], model_name="gpt-3.5-turbo", max_tokens=2300)
   
    assist = RetrievalQA.from_llm(llm , retriever=vectordb.as_retriever(kwargs={'50'}))
    response = assist("Summarize the following in medium app like article(in 500 words) format with headers and sub headers")
    return response['result']
   
   
    # chain = load_summarize_chain(llm, chain_type="stuff")

    # search = vectordb.as_retriever(kwargs={'25'})
    # # search = vectordb.similarity_search(" ", k=10)

    # print("search results",(search))
    # summary = chain.run(input_documents=search, question="Summarize the following in medium app like article format with headers and sub headers")
    # return summary




    