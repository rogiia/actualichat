import requests
from bs4 import BeautifulSoup
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
import datetime

BASE_URL = "https://www.naciodigital.cat/crono"

def get_page(url):
  try:
    response = requests.get(url)
    if response.ok:
      return BeautifulSoup(response.text, 'html.parser')
    else:
      print("ERROR:")
      print(response)
  except requests.exceptions.ConnectionError as exc:
    print(exc)

def create_documents(texts):
  text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
  docs = text_splitter.create_documents(texts)
  return docs

def save_embeddings(docs):
  date = datetime.datetime.now().strftime("%Y%m%d")
  embeddings = OpenAIEmbeddings()
  db = FAISS.from_documents(docs, embeddings)
  db.save_local(f"faiss_{date}")

page = get_page(BASE_URL)

titulars = page.find_all('h2', class_='titolnoticiallistat', limit=1)
docs = []

for titular in titulars:
  titol = titular.find('a').get_text()
  urlNoticia = titular.find('a')['href']
  pageNoticia = get_page(urlNoticia)
  subtitol = pageNoticia.find('h2').get_text()
  text = pageNoticia.find('div', class_='amp_textnoticia').get_text().split('Altres notícies que et poden interessar')[0]
  docs.append(titol)
  docs.append(subtitol)
  docs.append(text)

d = create_documents(docs)
save_embeddings(d)
