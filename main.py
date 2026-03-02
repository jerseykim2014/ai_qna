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
import re
from streamlit_extras.buy_me_a_coffee import button
from openai import OpenAI, AuthenticationError, APIConnectionError, APIStatusError
from dotenv import load_dotenv
load_dotenv()

#title
st.title("ChatPDF")
st.write("---")

def looks_like_openai_key(key: str) -> bool:
  key = key.strip()
  # Accepts current key styles (e.g. sk-..., sk-proj-...).
  return bool(re.fullmatch(r"sk-[A-Za-z0-9_-]{20,}", key))

def validate_openai_key_live(key: str):
  try:
    client = OpenAI(api_key=key, timeout=8.0)
    client.models.list()
    return True, ""
  except AuthenticationError:
    return False, "Invalid OpenAI API key (authentication failed)."
  except APIConnectionError:
    return False, "Could not reach OpenAI. Check your network and try again."
  except APIStatusError as e:
    if e.status_code in (401, 403):
      return False, "OpenAI API key is not authorized for this request."
    return False, f"OpenAI API error (status {e.status_code})."
  except Exception:
    return False, "Unexpected error while validating the API key."

is_key_verified = False
openai_key = st.session_state.get("validated_key", "")

# OpenAI key input is shown only before successful verification.
if openai_key:
  is_key_verified = True
  st.success("OpenAI API key verified.")
else:
  openai_key = st.text_input('OPEN_AI_API_KEY', type='password').strip()
  if not openai_key:
    st.info("Enter your OpenAI API key to proceed further.")
  elif not looks_like_openai_key(openai_key):
    st.error("Invalid key format. It should start with 'sk-'.")
  else:
    with st.spinner("Validating API key..."):
      is_valid, error_message = validate_openai_key_live(openai_key)
    if not is_valid:
      st.error(error_message)
    else:
      st.session_state["validated_key"] = openai_key
      is_key_verified = True
      st.success("OpenAI API key verified.")

#file upload
uploaded_file = st.file_uploader(
  "PDF Upload",
  type=["pdf"],
  disabled=not is_key_verified
)
st.write("---")

#Buy me a coffee
button(username="jerseykim", floating=True, width=221)

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

st.header("PDF에게 질문해 보세요!!")
question = st.text_input(
  "질문을 입력하세요",
  disabled=(not is_key_verified or uploaded_file is None)
)
ask_button_disabled = (
  not is_key_verified or uploaded_file is None or not question.strip()
)
ask_clicked = st.button("질문하기", disabled=ask_button_disabled)

if is_key_verified and uploaded_file is None:
  st.info("Upload a PDF to enable question input.")

#업로드된 파일 처리
if ask_clicked:
  with st.spinner("Wait for it..."):
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

    #Retriever
    # question = "아내가 먹고 싶어하는 음식은 무엇이야?"
    llm = ChatOpenAI(temperature=0, openai_api_key=openai_key)
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
