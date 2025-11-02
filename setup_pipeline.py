"""
PDF RAG Pipeline Setup Script
This script handles PDF parsing, multimodal extraction, chunking, embedding generation, and Qdrant indexing.
"""

import os
import sys
import fitz  # PyMuPDF
from pathlib import Path
from typing import List, Dict, Any
import json
import base64
from io import BytesIO

try:
    from langchain_ollama import OllamaEmbeddings
except ImportError:
    from langchain_community.embeddings import OllamaEmbeddings
try:
    from langchain_qdrant import Qdrant
except ImportError:
    from langchain_community.vectorstores import Qdrant
try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:
    from langchain.text_splitter import RecursiveCharacterTextSplitter
try:
    from langchain_core.documents import Document
except ImportError:
    from langchain.schema import Document
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

# Configuration
PDF_PATH = "Maths_Grade_10.pdf"
COLLECTION_NAME = "maths_grade_10"
QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
EMBEDDING_MODEL = "nomic-embed-text"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
IMAGES_DIR = "extracted_images"
METADATA_FILE = "image_metadata.json"


def extract_pdf_content(pdf_path: str) -> tuple[List[Document], Dict[str, Any]]:
    """
    Extract text and images from PDF using PyMuPDF.
    Returns documents with text and metadata about images.
    """
    print(f"Loading PDF: {pdf_path}")
    
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")
    
    # Create directory for extracted images
    os.makedirs(IMAGES_DIR, exist_ok=True)
    
    doc = fitz.open(pdf_path)
    documents = []
    image_metadata = {}
    
    for page_num in range(len(doc)):
        page = doc[page_num]
        
        # Extract text
        text = page.get_text()
        
        # Extract images
        image_list = page.get_images()
        page_images = []
        
        for img_index, img in enumerate(image_list):
            try:
                xref = img[0]
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]
                image_ext = base_image["ext"]
                
                # Save image
                image_filename = f"page_{page_num + 1}_img_{img_index + 1}.{image_ext}"
                image_path = os.path.join(IMAGES_DIR, image_filename)
                
                with open(image_path, "wb") as img_file:
                    img_file.write(image_bytes)
                
                # Get image position on page
                image_rects = page.get_image_rects(xref)
                
                page_images.append({
                    "filename": image_filename,
                    "path": image_path,
                    "rects": [{"x0": r.x0, "y0": r.y0, "x1": r.x1, "y1": r.y1} for r in image_rects],
                    "format": image_ext
                })
            except Exception as e:
                print(f"Warning: Could not extract image {img_index} from page {page_num + 1}: {e}")
        
        # Create document for the page
        if text.strip():
            doc_metadata = {
                "page": page_num + 1,
                "source": pdf_path,
                "total_pages": len(doc),
                "image_count": len(page_images),
                "images": page_images
            }
            
            documents.append(Document(
                page_content=text,
                metadata=doc_metadata
            ))
        
        # Store image metadata
        image_metadata[f"page_{page_num + 1}"] = page_images
        
        print(f"Processed page {page_num + 1}/{len(doc)}: {len(text)} characters, {len(page_images)} images")
    
    doc.close()
    
    # Save image metadata
    with open(METADATA_FILE, "w") as f:
        json.dump(image_metadata, f, indent=2)
    
    print(f"\nExtracted {len(documents)} pages with text content")
    print(f"Total images extracted: {sum(len(imgs) for imgs in image_metadata.values())}")
    
    return documents, image_metadata


def intelligent_chunking(documents: List[Document]) -> List[Document]:
    """
    Intelligent chunking that preserves context, especially for mathematical content.
    Uses recursive character text splitter with context-aware settings.
    """
    print("\nChunking documents with intelligent strategy...")
    
    # Use RecursiveCharacterTextSplitter with custom separators
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],  # Prioritize paragraph breaks
        length_function=len,
    )
    
    chunks = []
    
    for doc in documents:
        # Split the document
        doc_chunks = text_splitter.split_documents([doc])
        
        # Preserve metadata in each chunk
        for chunk in doc_chunks:
            chunk.metadata.update({
                "original_page": doc.metadata.get("page"),
                "chunk_index": len(chunks)
            })
            
            # Add image references if present on the same page
            chunk.metadata["images"] = doc.metadata.get("images", [])
        
        chunks.extend(doc_chunks)
    
    print(f"Created {len(chunks)} chunks from {len(documents)} documents")
    return chunks


def setup_qdrant(collection_name: str, use_memory: bool = False) -> QdrantClient:
    """
    Setup Qdrant client and create collection if it doesn't exist.
    Falls back to in-memory mode if Docker Qdrant is not available.
    """
    print(f"\nSetting up Qdrant connection...")
    
    if use_memory:
        print("Using Qdrant local file mode (no Docker required)...")
        try:
            # Use a local directory instead of :memory: for compatibility with LangChain
            local_path = "./qdrant_local"
            client = QdrantClient(path=local_path)
            print(f"[OK] Qdrant local client created at {local_path}")
            return client
        except Exception as e:
            print(f"Error creating local Qdrant client: {e}")
            sys.exit(1)
    
    try:
        client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
        
        # Check if collection exists
        collections = client.get_collections()
        collection_names = [col.name for col in collections.collections]
        
        if collection_name in collection_names:
            print(f"Collection '{collection_name}' already exists. Deleting it to recreate...")
            client.delete_collection(collection_name)
        
        print(f"Qdrant connected successfully at {QDRANT_HOST}:{QDRANT_PORT}")
        return client
        
    except Exception as e:
        print(f"Error connecting to Qdrant server: {e}")
        print("Attempting to use local file Qdrant (no Docker required)...")
        try:
            local_path = "./qdrant_local"
            client = QdrantClient(path=local_path)
            print(f"[OK] Qdrant local client created at {local_path}")
            return client
        except Exception as e2:
            print(f"Error creating in-memory Qdrant: {e2}")
            print("\nTo use Docker Qdrant, start it with:")
            print("docker run -p 6333:6333 -p 6334:6334 qdrant/qdrant")
            sys.exit(1)


def generate_embeddings_and_index(chunks: List[Document], collection_name: str):
    """
    Generate embeddings using Ollama and index them in Qdrant.
    """
    print(f"\nInitializing Ollama embeddings model: {EMBEDDING_MODEL}")
    
    try:
        embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
        
        # Test embedding generation
        test_embedding = embeddings.embed_query("test")
        embedding_dim = len(test_embedding)
        print(f"Embedding dimension: {embedding_dim}")
        
    except Exception as e:
        print(f"Error initializing Ollama embeddings: {e}")
        print("Make sure Ollama is running and the model is available:")
        print(f"ollama pull {EMBEDDING_MODEL}")
        sys.exit(1)
    
    # Setup Qdrant client (will fallback to local file mode if Docker not available)
    use_memory_mode = False
    try:
        client = setup_qdrant(collection_name, use_memory=False)
    except SystemExit:
        raise
    except Exception as e:
        print(f"Server mode failed: {e}")
        use_memory_mode = True
        client = setup_qdrant(collection_name, use_memory=True)
    
    print(f"\nCreating collection '{collection_name}' in Qdrant...")
    
    # Create collection with proper vector size
    try:
        # Check if collection exists first (for local mode, this might fail silently)
        if not use_memory_mode:
            collections = client.get_collections()
            collection_names = [col.name for col in collections.collections]
            if collection_name in collection_names:
                client.delete_collection(collection_name)
    except Exception as e:
        # For local mode, get_collections might behave differently
        print(f"Note: {e}")
        pass
    
    # Create collection
    try:
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(
                size=embedding_dim,
                distance=Distance.COSINE
            )
        )
    except Exception as e:
        # Collection might already exist, try to delete and recreate
        try:
            client.delete_collection(collection_name)
            client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(
                    size=embedding_dim,
                    distance=Distance.COSINE
                )
            )
        except Exception as e2:
            print(f"Error creating collection: {e2}")
            raise
    
    print(f"Indexing {len(chunks)} chunks into Qdrant...")
    
    # Use LangChain's Qdrant wrapper for easy indexing
    # For both modes, use the client directly
    vector_store = Qdrant(
        client=client,
        collection_name=collection_name,
        embeddings=embeddings
    )
    
    # Manually add documents (works for both server and local modes)
    texts = [doc.page_content for doc in chunks]
    metadatas = [doc.metadata for doc in chunks]
    vector_store.add_texts(texts=texts, metadatas=metadatas)
    
    print(f"\n[SUCCESS] Successfully indexed {len(chunks)} documents in Qdrant collection '{collection_name}'")
    if use_memory_mode:
        print("Note: Using local file mode. Data is stored in ./qdrant_local/ directory.")
    
    return vector_store


def main():
    """
    Main pipeline setup function.
    """
    print("=" * 60)
    print("PDF RAG Pipeline Setup")
    print("=" * 60)
    
    # Step 1: Extract PDF content
    documents, image_metadata = extract_pdf_content(PDF_PATH)
    
    if not documents:
        print("Error: No content extracted from PDF")
        sys.exit(1)
    
    # Step 2: Intelligent chunking
    chunks = intelligent_chunking(documents)
    
    # Step 3: Generate embeddings and index in Qdrant
    vector_store = generate_embeddings_and_index(chunks, COLLECTION_NAME)
    
    print("\n" + "=" * 60)
    print("Setup Complete!")
    print("=" * 60)
    print(f"Collection: {COLLECTION_NAME}")
    print(f"Documents indexed: {len(chunks)}")
    print(f"Images extracted: {sum(len(imgs) for imgs in image_metadata.values())}")
    print(f"Image metadata saved to: {METADATA_FILE}")
    print(f"Images directory: {IMAGES_DIR}")
    print("\nYou can now run rag_query.py to query the indexed content.")


if __name__ == "__main__":
    main()

