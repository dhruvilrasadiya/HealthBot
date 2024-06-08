from flask import Flask, render_template, jsonify, request
from src.helper import download_hugging_face_embeddings, format_docs
from langchain_pinecone import PineconeVectorStore
import os
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain.llms import CTransformers
from src.prompt import *
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

app = Flask(__name__)

load_dotenv()

embeddings = download_hugging_face_embeddings()

index_name="medicalchatbot"

docsearch=PineconeVectorStore.from_existing_index(index_name, embeddings)

PROMPT=PromptTemplate(template=prompt_template, input_variables=["context", "question"])
llm=CTransformers(model="model/llama-2-7b-chat.ggmlv3.q4_0.bin",
                  model_type="llama",
                  config={'max_new_tokens':512,
                          'temperature':0.8})

retriever = docsearch.as_retriever(search_kwargs={'k': 2})

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | PROMPT
    | llm
    | StrOutputParser()
)

@app.route("/")
def index():
    return render_template('chat.html')



@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    input = msg
    print(input)
    result=rag_chain.invoke(input)
    print("Response : ", result)
    return str(result)



if __name__ == '__main__':
    app.run(host="0.0.0.0", port= 8080, debug= True)