# AI-Powered-HR-Assistant
Faster, Fairer, and More Efficient Hiring Using Graph RAG

Graph-RAG is a sophisticated question-answering system that leverages knowledge graphs and Retrieval-Augmented Generation (RAG) to provide accurate and context-aware responses. The system processes PDF documents, extracts structured information, and stores it in a Neo4j graph database for efficient querying and retrieval.

![image](https://github.com/user-attachments/assets/f8f09f0d-e9a4-487e-8742-11822912abbd)

Chatbot UI:

<img width="782" alt="chatbot_ss_2" src="https://github.com/user-attachments/assets/554e98e7-c5f2-4329-a928-7fc0cc906955" />

Link to connect with us - https://bit.ly/poster-informs-2025


## Features

- PDF document processing and text extraction
- Automatic knowledge graph construction from documents
- Structured information extraction
- Graph-based question answering
- Interview question generation
- Interactive web interface using Streamlit
- Support for multiple LLM providers (Groq, OpenAI, Ollama)
- Flexible embedding options (Cohere, OpenAI, Ollama)

## Prerequisites

- Python 3.8+
- Neo4j Database
- API keys for:
  - Groq (or other LLM provider)
  - Cohere (or other embedding provider)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/mahithabsl/Graph-RAG.git
cd Graph-RAG
```

2. Install required dependencies:
```bash
pip install -r requirements.txt
```

3. Configure Neo4j:
   - Install and start Neo4j database
   - Update the `config.ini` file with your Neo4j credentials

4. Set up API keys:
   - Update the `config.ini` file with your API keys for the LLM and embedding providers

## Configuration

The `config.ini` file contains all necessary configuration settings:

- Neo4j connection details
- LLM provider settings
- PDF processing parameters
- Embedding model configuration

## Project Structure

- `Pdf_2_Text.py`: Handles PDF text extraction and resume parsing
- `MetaData_Extraction.ipynb`: Extracts structured metadata from parsed text
- `Pdf_2_Graph.py`: Constructs knowledge graph from extracted information
- `Graph_QA.py`: Implements the question-answering system
- `Generate_Interview_Questions.ipynb`: Generates interview questions based on resume content
- `data/`: Directory for storing PDF documents
- `static/`: Static assets for the web interface

## Usage

### Step 1: Resume Parsing and PDF to Text
```python
from graph_rag.Pdf_2_Text import process_pdf

# Process a PDF resume
text_content = process_pdf("path/to/your/resume.pdf")
```

### Step 2: Metadata Extraction
```python
# Run the MetaData_Extraction.ipynb notebook
# This extracts structured information from the parsed text
```

### Step 3: PDF to Graph Construction
```python
from graph_rag.Pdf_2_Graph import process_document

# Process a document and build the knowledge graph
process_document("path/to/your/document.pdf", metadata_info={}, meta={})
```

### Step 4: Graph-based Question Answering
```bash
streamlit run graph_rag/Graph_QA.py
```
Access the web interface at `http://localhost:8501`

### Step 5: Interview Question Generation
```python
# Run the Generate_Interview_Questions.ipynb notebook
# This generates relevant interview questions based on the resume content
```

## How It Works

1. **Resume Parsing and Text Extraction**:
   - PDFs are processed and converted to text
   - Structured information is extracted from resumes

2. **Metadata Extraction**:
   - Key information is identified and structured
   - Education, experience, skills, and other relevant data are extracted

3. **Knowledge Graph Construction**:
   - Entities and relationships are identified
   - Information is stored in Neo4j graph database
   - Graph structure enables complex relationship queries

4. **Question Answering**:
   - User questions are processed
   - Relevant information is retrieved from the graph
   - LLM generates context-aware responses

5. **Interview Question Generation**:
   - Resume content is analyzed
   - Relevant interview questions are generated
   - Questions are tailored to the candidate's background

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
