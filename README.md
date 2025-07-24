# Conversational-AI
Agents for reading documentation files and websites related to a particular topic and answering question based on them.

## Overview

This project builds a question-answering AI assistant capable of reading research papers (PDFs) and website content related to a specific topic and answering questions based on that content. It uses web scraping, document loaders, text chunking, embedding generation, and a FAISS vector store to enable semantic search and question answering.

## Key Features

- Scrapes content from academic paper URLs (e.g., arXiv)
- Loads PDF and text files using LangChain loaders
- Splits text into manageable chunks for embedding
- Generates embeddings using HuggingFace models
- Stores embeddings in a FAISS vector store for efficient similarity search
- Answers user queries based on semantic similarity

## Tech Stack

- **Python**
- **LangChain**
- **BeautifulSoup4** for web scraping
- **HuggingFace Transformers** for embeddings
- **FAISS** for vector-based similarity search

## Setup

#### Install the required libraries:

pip install requests beautifulsoup4 langchain langchain-experimental lxml unstructured
(Optional) Upload PDF files and configure their path using PyPDFLoader.

#### Run the script to:

- Scrape website content

- Load and split documents

- Generate and store embeddings

- Search for relevant answers using a query

#### Usage
- Modify the website_urls list in the script to point to documentation or article pages you'd like to include.

#### Run
- python Conversational_ai.py
