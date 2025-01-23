from langchain.vectorstores import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.memory import ConversationSummaryMemory
from langchain.chains import ConversationalRetrievalChain
from src.helper import repo_ingestion, load_repo, text_splitter, load_embeddings
import os
from dotenv import load_dotenv
from flask import Flask, request, jsonify, render_template
import subprocess

app = Flask(__name__)

load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
os.environ['GOOOGL_API_KEY'] = GOOGLE_API_KEY

embeddings = load_embeddings()
persist_directory = 'db'

vectordb = Chroma(persist_directory=persist_directory,
                  embedding_function=embeddings)
llm = ChatGoogleGenerativeAI(model='gemini-1.5-pro', max_tokens=500,temperature=0.3)
memory = ConversationSummaryMemory(llm=llm,memory_key='chat_history',return_messages=True)
qa = ConversationalRetrievalChain.from_llm(llm,retriever=vectordb.as_retriever(search_type="mmr",search_kwargs={"k":5}),memory=memory)

@app.route('/',methods=['GET','POST'])
def index():
    return render_template('index.html')

@app.route('/chatbot',methods=['POST'])
def gitRepo():
    user_input = request.form['question']
    repo_ingestion(user_input)
    subprocess.run(['python', 'store_index.py'], check=True)
    return jsonify({'response':str(user_input)})


@app.route('/get',methods=['GET','POST'])
def chat():
    msg = request.form['msg']
    if msg == 'clear':
        os.system('rd /s /q "repo"')
        return str('the source code has been deleted')
    result = qa(msg)
    return str(result['answer'])

if __name__ == '__main__':
    app.run(debug=True, port=8080,host='0.0.0.0')