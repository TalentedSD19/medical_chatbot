from flask import Flask,render_template,jsonify,request
from langchain_chroma import Chroma
from langchain.llms import OpenAI
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from src.prompt import *
from dotenv import load_dotenv
import os

app = Flask(__name__)

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

embeddings=OpenAIEmbeddings()

vectorstore = Chroma(persist_directory=r"db", embedding_function=embeddings)

PROMPT=PromptTemplate(template=prompt_template, input_variables=["context", "question"])
chain_type_kwargs={"prompt": PROMPT}

llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY,model_name="gpt-3.5-turbo",temperature=0.3)

qa=RetrievalQA.from_chain_type(
    llm=llm, 
    chain_type="stuff", 
    retriever=vectorstore.as_retriever(search_kwargs={'k': 3}),
    return_source_documents=True, 
    chain_type_kwargs=chain_type_kwargs)

@app.route("/")
def index():
    return render_template("chat.html")


@app.route("/get",methods=["GET","POST"])
def chat():
    msg = request.form["msg"]
    print(msg)
    result = qa({"query":msg})
    print("Response: ", result)
    return str(result['result'])

if __name__ == '__main__':
    app.run(debug=True)



