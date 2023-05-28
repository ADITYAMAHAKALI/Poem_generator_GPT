from dotenv import load_dotenv
from langchain.llms import OpenAI
import os
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma

# Load the .env file
load_dotenv()

# Get the API key
api_key = os.getenv("OPENAI_API_KEY")
persist_directory = 'db'

# llm = OpenAI(openai_api_key = api_key, temperature=0.9, max_tokens=100)
# print(llm.predict("Who should have been wife of prime minister of India?").strip())
loader = CSVLoader(file_path='Scraping/writers/rabindranath-tagore.csv')

data = loader.load()
#print(data)
text_splitter = CharacterTextSplitter(chunk_size=1800, chunk_overlap=100)
docs = text_splitter.split_documents(data)
embeddings = OpenAIEmbeddings()
db = Chroma.from_documents(docs, embeddings,persist_directory=persist_directory)
db.persist()
query = "the children meet with shouts and dances"
similar = db.similarity_search(query)
print(similar)