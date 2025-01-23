from git import Repo
from langchain.text_splitter import Language
from langchain.document_loaders.generic import GenericLoader
from langchain.document_loaders.parsers import LanguageParser
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings


#clone the repo
def repo_ingestion(repo_url):
    os.makedirs('repo', exist_ok=True)
    repo_path = 'repo'
    Repo.clone_from(repo_url, to_path=repo_path)
    
#loading repo as documents
def load_repo(repo_path):
    # Load the documents from the repository
    loader = GenericLoader.from_filesystem(repo_path,
                                           glob='**/*',
                                           suffixes=['.py'],
                                           parser=LanguageParser(Language.PYTHON,parser_threshold=0.5))
    documents = loader.load()
    return documents

#Create text chunks
def text_splitter(documents):
    # Create text chunks
    documents_splitter = RecursiveCharacterTextSplitter.from_language(language=Language.PYTHON,
                                                                      chunk_size=500,
                                                                      chunk_overlap=20)
    text_chunks = documents_splitter.split_documents(documents)
    return text_chunks

#Load embeddings model
def load_embeddings():
    embedding=GoogleGenerativeAIEmbeddings(model='models/embedding-001')
    return embedding