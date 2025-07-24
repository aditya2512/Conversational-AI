# %% [markdown]
# 
# # Agents for reading documentation files and websites related to a particular topic and answering question based on them.

# %% [markdown]
# ## Install necessary libraries
# Install libraries required for web scraping, document loading, and agent creation (e.g., `requests`, `BeautifulSoup`, `langchain`, `langchain-experimental`, etc.).

# %%
pip install requests beautifulsoup4 langchain langchain-experimental lxml unstructured

# %% [markdown]
# ## Load data
# Load data from provided documentation files (e.g., PDF, text files) and scrape content from specified websites related to the topic.

# %%
import requests
from bs4 import BeautifulSoup
from langchain_community.document_loaders import PyPDFLoader, TextLoader
import os

# Define website URLs 
website_urls = [
  "https://arxiv.org/pdf/2010.07487v3.pdf",
  "https://arxiv.org/pdf/2008.02275v3.pdf",
  "https://arxiv.org/pdf/2401.13481",
  "https://arxiv.org/pdf/2505.07468",
]

# Data structure to store content
all_content = []

# Scrape content from websites
for url in website_urls:
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise an exception for bad status codes
        soup = BeautifulSoup(response.content, 'lxml')
        # Extract text content (you might need to refine this based on website structure)
        text_content = soup.get_text(separator='\n', strip=True)
        all_content.append({"url": url, "content": text_content})
        print(f"Scraped content from {url}")
    except requests.exceptions.RequestException as e:
        print(f"Error scraping {url}: {e}")
    except Exception as e:
        print(f"Error processing content from {url}: {e}")


# uploading PDF files here.
# pdf_file_path = "path/to/your/uploaded/file.pdf" # Replace with the actual path
# loader = PyPDFLoader(pdf_file_path)
# pdf_documents = loader.load()
# all_content.extend(pdf_documents) # Add loaded PDF documents to all_content


# Display a sample of the stored content
print("\n--- Sample of Loaded/Scraped Content ---")
if all_content:
    # Print content based on type (Document object or dictionary)
    for item in all_content[:2]: # Display first 2 items as sample
        if isinstance(item, dict):
            print(f"Source: {item['url']}\nContent snippet: {item['content'][:500]}...")
        else:
            print(f"Source: {item.metadata.get('source', 'Local File')}\nContent snippet: {item.page_content[:500]}...")
else:
    print("No content was loaded or scraped.")

# %% [markdown]
# ## Process Data
# 
# ### Subtask:
# Split the loaded content into smaller chunks and generate embeddings for each chunk.

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings # Example of another embedding model

# Split the documents into smaller chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
# Assuming 'all_content' is a list of strings or objects with a 'content' attribute
texts = []
for item in all_content:
    if isinstance(item, dict):
        texts.append(item['content'])
    else:
        texts.append(item.page_content)

split_texts = text_splitter.create_documents(texts)


# Generate embeddings for each chunk (replace with your chosen embedding model)
# This is a placeholder. You'll need to choose and configure an actual embedding model.

embeddings = HuggingFaceEmbeddings()
doc_embeddings = embeddings.embed_documents([t.page_content for t in split_texts])

print(f"Split {len(texts)} documents into {len(split_texts)} chunks.")
print("\n--- Sample of Split Text Chunks ---")
for i, chunk in enumerate(split_texts[:2]):
    print(f"Chunk {i+1}:\n{chunk.page_content[:500]}...\n")

# %% [markdown]
# ## Embeddings to Store in Vector Store
# embeddings for the split text chunks and store them in a vector store.

# %%
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

vectorstore = FAISS.from_documents(split_texts, embeddings)

print(f"Created a vector store with {len(split_texts)} chunks.")
print("\n--- Vector Store Info ---")

# Perform similarity searches on the vectorstore
query = "What is the main topic of the documents?"
docs = vectorstore.similarity_search(query)
print(f"Most similar documents to the query: {docs}")


