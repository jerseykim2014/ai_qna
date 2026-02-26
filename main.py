from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_classic.retrievers.multi_query import MultiQueryRetriever
from langchain_openai import ChatOpenAI
# from langchain import hub
from langsmith import Client
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
import streamlit as st
import tempfile
import os
from dotenv import load_dotenv
load_dotenv()

#title
st.title("ChatPDF")
st.write("---")

#OpenAI Key"
openai_key = st.text_input('OPEN_AI_API_KEY', type='password')

#file upload
uploaded_file = st.file_uploader("PDF Upload", type=["pdf"])
st.write("---")

def pdf_to_document(uploaded_file):
  temp_dir = tempfile.TemporaryDirectory()
  temp_filepath = os.path.join(temp_dir.name, uploaded_file.name)
  with open(temp_filepath, "wb") as f:
    f.write(uploaded_file.getvalue())
  loader = PyPDFLoader(temp_filepath)
  pages = loader.load_and_split()
  return pages
# #loader
# loader = PyPDFLoader("luck.pdf")
# pages = loader.load_and_split()

#업로드된 파일 처리
if uploaded_file is not None:
  pages = pdf_to_document(uploaded_file)
  #splitter
  text_splitter = RecursiveCharacterTextSplitter(
    #set a really small chunk size, just to show.
    chunk_size=300,
    chunk_overlap=20,
    length_function=len,
    is_separator_regex=False,
  )
  texts = text_splitter.split_documents(pages)
  # print("pages[0] ",pages[0])
  # print("texts[0] ",texts[0])

  #Embedding
  embeddings_model = OpenAIEmbeddings(
    model="text-embedding-3-large",
    openai_api_key=openai_key
    # With the `text-embedding-3` class
    # of models, you can specify the size
    # of the embeddings you want returned.
    # dimensions=1024
  )

  #Chroma DB
  db = Chroma.from_documents(texts, embeddings_model)

  #User Input
  st.header("PDF에게 질문해 보세요!!")
  question = st.text_input("질문을 입력하세요")

  if st.button("질문하기"):
    with st.spinner("Wait for it..."):
      #Retriever
      # question = "아내가 먹고 싶어하는 음식은 무엇이야?"
      llm = ChatOpenAI(temperature=0)
      retriever_from_llm = MultiQueryRetriever.from_llm(
        retriever=db.as_retriever(), llm=llm
      )
      #Prompt Template
      # prompt = hub.pull("rlm/rag-prompt")
      client = Client()
      prompt = client.pull_prompt("rlm/rag-prompt")
      #Generate
      def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)
      rag_chain = (
        {
          "question": RunnablePassthrough(),
          "context": retriever_from_llm | format_docs}
          | prompt
          | llm
          | StrOutputParser()
      )

      #Question
      result = rag_chain.invoke(question)
      st.write(result)
