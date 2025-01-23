from src.helper import repo_ingestion, load_repo, text_splitter, load_embeddings
from dotenv import load_dotenv
from langchain.vectorstores import Chroma
import os

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
os.environ['GOOOGL_API_KEY'] = GOOGLE_API_KEY

documents = load_repo('repo/')
print('-----------------')
print('Length of docs',len(documents))
print('-----------------')
text_chunks = text_splitter(documents)
embeddings = load_embeddings()

vectordb = Chroma.from_documents(text_chunks,embedding=embeddings,persist_directory='db')
print('-----------------')
print(vectordb.search('explain app.py?'))
print('-----------------')
vectordb.persist()
                                 