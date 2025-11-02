# Multimodal RAG Pipeline for Educational Content

A complete Retrieval-Augmented Generation (RAG) pipeline built with Python, LangChain, Ollama, and Qdrant. This system processes academic PDF documents containing text, mathematical formulas, and images/diagrams, indexes the content, and provides intelligent querying capabilities.

## Features

- **Multimodal PDF Processing**: Extracts both text and images from PDF documents
- **Intelligent Chunking**: Preserves context, especially for mathematical content
- **Vector Embeddings**: Uses Ollama's `nomic-embed-text` model for embeddings
- **Qdrant Vector Store**: Efficient similarity search and retrieval
- **RAG Pipeline**: LangChain-orchestrated question-answering system
- **Ollama LLM Integration**: Uses `llama3` or `mistral` for answer generation
- **Context Summarization**: Summarizes retrieved context before final answer generation
- **Conversational Caching**: Maintains conversation history for follow-up questions
- **Prompt Caching**: Caches LLM responses for identical queries

## Prerequisites

1. **Docker** (for running Qdrant)  
   > ⚙️ **Note:** If Docker is not connected or Qdrant is not running,  
   > the system will automatically switch to **local mode** and create a  
   > `./qdrant_local/` directory for storing embeddings locally —  
   > so the project will still run without Docker.
2. **Ollama** installed and running
   - Download from: https://ollama.ai
   - Install required models (see Setup section)

## Setup

### 1. Install Python Dependencies

```bash
pip install -r requirements.txt
```

### 2. Start Qdrant Vector Database

```bash
docker run -p 6333:6333 -p 6334:6334 qdrant/qdrant
```

Qdrant will be available at `http://localhost:6333`

Note: If Docker is not connected or Qdrant is not running,
the system will automatically switch to local mode and create a
./qdrant_local/ directory for storing embeddings locally —
so the project will still run without Docker
### 3. Install Ollama Models

Ensure Ollama is running, then pull the required models:

```bash
# Embedding model
ollama pull nomic-embed-text

# LLM model (choose one)
ollama pull llama3
# OR
ollama pull mistral
```

### 4. Index the PDF Document

Place your PDF file (`Maths_Grade_10.pdf`) in the project root, then run:

```bash
python setup_pipeline.py
```

**Expected Output:**
```
============================================================
PDF RAG Pipeline Setup
============================================================
Loading PDF: Maths_Grade_10.pdf
Processed page 1/50: 1234 characters, 2 images
...
Extracted 50 pages with text content
Total images extracted: 15

Chunking documents with intelligent strategy...
Created 234 chunks from 50 documents

Setting up Qdrant connection...
Qdrant connected successfully at localhost:6333

Creating collection 'maths_grade_10' in Qdrant...
Indexing 234 chunks into Qdrant...

✓ Successfully indexed 234 documents in Qdrant collection 'maths_grade_10'

============================================================
Setup Complete!
============================================================
Collection: maths_grade_10
Documents indexed: 234
Images extracted: 15
Image metadata saved to: image_metadata.json
Images directory: extracted_images

You can now run rag_query.py to query the indexed content.
```

## Usage

### Basic Query (Non-Conversational)

```bash
python rag_query.py --question "Explain the steps involved in solving a quadratic equation as mentioned in Chapter 4."
```

**Expected Output:**
```
Query: Explain the steps involved in solving a quadratic equation as mentioned in Chapter 4.

Retrieving relevant documents...
Retrieved 4 relevant documents
Similarity scores: ['0.8234', '0.7891', '0.7654', '0.7432']

============================================================
FINAL ANSWER:
------------------------------------------------------------
To solve a quadratic equation, the following steps are typically followed:

1. Write the equation in standard form: ax² + bx + c = 0
2. Identify the coefficients a, b, and c
3. Apply the quadratic formula: x = (-b ± √(b² - 4ac)) / 2a
...

------------------------------------------------------------
SOURCES:
------------------------------------------------------------
1. Page 45
   Preview: Chapter 4: Quadratic Equations. A quadratic equation is a polynomial equation of degree 2...

2. Page 46
   (Contains images/diagrams)
   Preview: Solving Quadratic Equations. Step 1: Identify the coefficients...
...
============================================================
```

### Multimodal Retrieval (Images/Diagrams)

```bash
python rag_query.py --question "What does the diagram illustrating the properties of a trapezoid show?"
```

**Expected Output:**
```
Query: What does the diagram illustrating the properties of a trapezoid show?

Retrieving relevant documents...
Retrieved 4 relevant documents
Similarity scores: ['0.8123', '0.7890', '0.7543', '0.7321']

============================================================
FINAL ANSWER:
------------------------------------------------------------
Based on the diagram on page 67, the trapezoid diagram shows:
- Two parallel sides (bases) of different lengths
- Two non-parallel sides (legs)
- The height is the perpendicular distance between the bases
- Angles at the base are supplementary
...

------------------------------------------------------------
SOURCES:
------------------------------------------------------------
1. Page 67
   (Contains images/diagrams)
   Preview: Properties of Trapezoids. A trapezoid is a quadrilateral with exactly one pair of parallel sides...
...
============================================================
```

### With Summarization

```bash
python rag_query.py --summarize --question "What is the concept of 'Arithmetic Progression'?"
```

**Expected Output:**
```
Query: What is the concept of 'Arithmetic Progression'?

Retrieving relevant documents...
Retrieved 4 relevant documents
Similarity scores: ['0.8456', '0.8234', '0.8012', '0.7789']

Generating summary of retrieved context...
Retrieved Context Summary:
An arithmetic progression (AP) is a sequence of numbers where each term after the first is obtained by adding a constant difference (d) to the previous term. The general form is: a, a+d, a+2d, a+3d, ... where 'a' is the first term and 'd' is the common difference. Key formulas include: nth term = a + (n-1)d, and sum of n terms = n/2[2a + (n-1)d].

============================================================
RETRIEVED CONTEXT SUMMARY:
------------------------------------------------------------
An arithmetic progression (AP) is a sequence of numbers where each term after the first is obtained by adding a constant difference (d) to the previous term. The general form is: a, a+d, a+2d, a+3d, ... where 'a' is the first term and 'd' is the common difference. Key formulas include: nth term = a + (n-1)d, and sum of n terms = n/2[2a + (n-1)d].

------------------------------------------------------------
FINAL ANSWER:
------------------------------------------------------------
An Arithmetic Progression (AP) is a sequence of numbers where the difference between consecutive terms is constant. This constant difference is called the "common difference" (denoted as 'd'). The sequence follows a predictable pattern, making it useful for solving various mathematical problems...

------------------------------------------------------------
SOURCES:
------------------------------------------------------------
1. Page 89
   Preview: Arithmetic Progressions. An arithmetic progression is a sequence where each term after the first is obtained...
...
============================================================
```

### Conversational Mode (Follow-up Questions)

```bash
# First question establishes context
python rag_query.py --conversational --question "Who proposed the Pythagorean theorem?"

# Second question uses memory from previous conversation
python rag_query.py --conversational --question "What is the formula associated with his discovery?"
```

**Expected Output (Second Query):**
```
Query: What is the formula associated with his discovery?

Note: Conversational mode enabled. Previous context is remembered.

============================================================
FINAL ANSWER:
------------------------------------------------------------
Based on our previous conversation about Pythagoras, the formula associated with his discovery is the Pythagorean theorem, which states: a² + b² = c², where a and b are the lengths of the legs of a right triangle, and c is the length of the hypotenuse...

------------------------------------------------------------
SOURCES:
------------------------------------------------------------
1. Page 123
   Preview: The Pythagorean theorem, attributed to Pythagoras, states that in a right-angled triangle...
...
============================================================
```

### Prompt Caching Demonstration

```bash
# First query (cache miss)
python rag_query.py --question "Explain integration by parts."

# Second identical query (cache hit - significantly faster)
python rag_query.py --question "Explain integration by parts."
```

The second query will be faster as it retrieves the cached LLM response.

## Project Structure

```
optimized_multimodel/
│
├── Maths_Grade_10.pdf          # Input PDF document
├── setup_pipeline.py           # PDF processing and indexing script
├── rag_query.py                # RAG querying script
├── requirements.txt            # Python dependencies
├── README.md                   # This file
```

## Command-Line Options

### `setup_pipeline.py`

No command-line arguments required. It processes `Maths_Grade_10.pdf` automatically.

### `rag_query.py`

| Option | Short | Description |
|--------|-------|-------------|
| `--question` | `-q` | **Required.** The question to ask about the PDF content |
| `--summarize` | `-s` | Enable summarization of retrieved context |
| `--conversational` | `-c` | Enable conversational mode for follow-up questions |
| `--k` | | Number of documents to retrieve (default: 4) |
| `--no-cache` | | Disable prompt caching |

## Technical Details

### PDF Processing

- Uses **PyMuPDF (fitz)** for PDF parsing
- Extracts text content from all pages
- Identifies and extracts images/diagrams embedded in PDFs
- Preserves image metadata (position, format, page number)

### Chunking Strategy

- Uses `RecursiveCharacterTextSplitter` with intelligent separators
- Prioritizes paragraph breaks (`\n\n`) to maintain context
- Preserves metadata including page numbers and image references
- Chunk size: 1000 characters with 200 character overlap

### Embeddings

- Model: `nomic-embed-text` (via Ollama)
- Dimension: Typically 768 (model-dependent)
- Distance metric: Cosine similarity

### Vector Store

- **Qdrant** running in Docker
- Collection: `maths_grade_10`
- Stores document chunks with metadata
- Enables similarity search for retrieval

### RAG Pipeline

- Retrieval: Top-k similarity search (default k=4)
- Context augmentation: Retrieved chunks + optional summarization
- Generation: Ollama LLM (`llama3` or `mistral`)
- Custom prompt template for educational content

### Caching

- **Prompt Caching**: In-memory cache for identical queries (LangChain's `InMemoryCache`)
- **Conversational Memory**: `ConversationBufferMemory` maintains chat history

## Troubleshooting

### Qdrant Connection Error

```
Error connecting to Qdrant: Connection refused
```

**Solution**: Make sure Qdrant is running:
```bash
docker run -p 6333:6333 -p 6334:6334 qdrant/qdrant
```
If not using Docker, ensure the local directory ./qdrant_local/ exists.
The script automatically switches to local mode when Docker Qdrant is unavailable.
### Ollama Model Not Found

```
Error loading LLM model: Model 'llama3' not found
```

**Solution**: Pull the required model:
```bash
ollama pull llama3
```

### Collection Not Found

```
Error connecting to vector store: Collection 'maths_grade_10' not found
```

**Solution**: Run the setup script first:
```bash
python setup_pipeline.py
```

### PDF Processing Errors

If image extraction fails, the pipeline will continue with text-only processing. Check the console output for warnings.

## Demonstration Checklist

✅ **Indexing Run**: `python setup_pipeline.py`  
✅ **Basic RAG Query**: `python rag_query.py --question "Explain..."`  
✅ **Multimodal Retrieval**: Query about images/diagrams  
✅ **Summarization**: `python rag_query.py --summarize --question "..."`  
✅ **Caching**: Run same query twice to show cache performance  
✅ **Conversational**: Use `--conversational` flag for follow-up questions  

## License

This project is created for educational/internship purposes.


