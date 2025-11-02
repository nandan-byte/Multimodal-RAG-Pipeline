"""
RAG Query Script
This script handles querying the indexed PDF content using RAG pipeline with summarization and caching.
"""

import os
import sys
import argparse
import json
from typing import List, Dict, Any

try:
    from langchain_core.documents import Document
except ImportError:
    from langchain.schema import Document
from langchain_ollama import OllamaEmbeddings
from langchain_qdrant import Qdrant
try:
    from langchain_ollama import OllamaLLM
except ImportError:
    from langchain_community.llms import Ollama as OllamaLLM
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda
try:
    from langchain.memory import ConversationBufferMemory
except ImportError:
    try:
        from langchain_community.memory import ConversationBufferMemory
    except ImportError:
        # Fallback: create a simple memory class
        class ConversationBufferMemory:
            def __init__(self, **kwargs):
                self.chat_memory = type('obj', (object,), {'messages': []})()
try:
    from langchain.chains.conversational_retrieval import ConversationalRetrievalChain
except ImportError:
    try:
        from langchain.chains import ConversationalRetrievalChain
    except ImportError:
        ConversationalRetrievalChain = None
try:
    from langchain.cache import InMemoryCache
    from langchain.globals import set_llm_cache
except ImportError:
    InMemoryCache = None
    set_llm_cache = lambda x: None

# Configuration
COLLECTION_NAME = "maths_grade_10"
QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
EMBEDDING_MODEL = "nomic-embed-text"
LLM_MODEL = "llama3"  # or "mistral"
METADATA_FILE = "image_metadata.json"
MEMORY_FILE = "conversation_memory.json"


class RAGPipeline:
    """
    RAG Pipeline with summarization and conversational caching.
    """
    
    def __init__(self, enable_cache: bool = True, session_id: str = "default"):
        self.embeddings = None
        self.vector_store = None
        self.llm = None
        self.session_id = session_id
        self.memory_file = f"{MEMORY_FILE}_{session_id}"
        
        # Initialize memory
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=False,
            output_key="answer"
        )
        
        # Load previous conversation history if exists
        self._load_memory()
        
        self.enable_cache = enable_cache
        
        if enable_cache and InMemoryCache and set_llm_cache:
            try:
                # Enable prompt caching
                set_llm_cache(InMemoryCache())
                print("Prompt caching enabled (InMemoryCache)")
            except:
                print("Prompt caching not available (skipping)")
        
        self._initialize()
    
    def _initialize(self):
        """Initialize embeddings, vector store, and LLM."""
        print("Initializing RAG pipeline...")
        
        # Initialize embeddings
        try:
            self.embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
            print(f"[OK] Embeddings model loaded: {EMBEDDING_MODEL}")
        except Exception as e:
            print(f"Error loading embeddings model: {e}")
            print(f"Make sure Ollama is running and model is available: ollama pull {EMBEDDING_MODEL}")
            sys.exit(1)
        
        # Initialize vector store
        try:
            from qdrant_client import QdrantClient
            import os
            
            # Try local file mode first (since setup creates it)
            local_path = "./qdrant_local"
            if os.path.exists(local_path):
                try:
                    print("Connecting to local Qdrant file...")
                    qdrant_client = QdrantClient(path=local_path)
                    self.vector_store = Qdrant(
                        client=qdrant_client,
                        collection_name=COLLECTION_NAME,
                        embeddings=self.embeddings
                    )
                    # Test connection by checking collection exists
                    collections = qdrant_client.get_collections()
                    collection_names = [col.name for col in collections.collections]
                    if COLLECTION_NAME in collection_names:
                        print(f"[OK] Vector store connected: {COLLECTION_NAME} (local file mode)")
                    else:
                        raise Exception(f"Collection {COLLECTION_NAME} not found in local file")
                except Exception as e:
                    # Try server mode as fallback
                    try:
                        print("Local file failed, trying server mode...")
                        qdrant_client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
                        self.vector_store = Qdrant(
                            client=qdrant_client,
                            collection_name=COLLECTION_NAME,
                            embeddings=self.embeddings
                        )
                        print(f"[OK] Vector store connected: {COLLECTION_NAME} (server mode)")
                    except:
                        raise Exception(f"Failed to connect: {e}")
            else:
                # Try server mode only
                try:
                    qdrant_client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
                    self.vector_store = Qdrant(
                        client=qdrant_client,
                        collection_name=COLLECTION_NAME,
                        embeddings=self.embeddings
                    )
                    print(f"[OK] Vector store connected: {COLLECTION_NAME} (server mode)")
                except Exception as e:
                    raise Exception(f"Qdrant server not available and no local file found. Error: {e}")
        except Exception as e:
            print(f"Error connecting to vector store: {e}")
            print("Make sure Qdrant is running OR run setup_pipeline.py first to create the collection.")
            sys.exit(1)
        
        # Initialize LLM
        try:
            self.llm = OllamaLLM(model=LLM_MODEL, temperature=0.7)
            # Test LLM connection
            _ = self.llm.invoke("test")
            print(f"[OK] LLM model loaded: {LLM_MODEL}")
        except Exception as e:
            print(f"Error loading LLM model: {e}")
            print(f"Make sure Ollama is running and model is available: ollama pull {LLM_MODEL}")
            sys.exit(1)
        
        print("RAG pipeline initialized successfully!\n")
    
    def _load_memory(self):
        """Load previous conversation history from file."""
        if os.path.exists(self.memory_file):
            try:
                with open(self.memory_file, 'r', encoding='utf-8') as f:
                    memory_data = json.load(f)
                    
                # Restore conversation history to memory
                if 'chat_history' in memory_data and memory_data['chat_history']:
                    # Add previous messages to memory
                    for entry in memory_data['chat_history']:
                        if isinstance(entry, dict):
                            human_msg = entry.get('human', '')
                            ai_msg = entry.get('ai', '')
                            if human_msg and ai_msg:
                                try:
                                    self.memory.chat_memory.add_user_message(human_msg)
                                    self.memory.chat_memory.add_ai_message(ai_msg)
                                except:
                                    # Try alternative method if available
                                    if hasattr(self.memory.chat_memory, 'add_message'):
                                        self.memory.chat_memory.add_message("human", human_msg)
                                        self.memory.chat_memory.add_message("ai", ai_msg)
                    
                    print(f"[OK] Loaded previous conversation history (session: {self.session_id})")
            except Exception as e:
                # Silently fail - don't interrupt the main flow
                pass
    
    def _save_memory(self):
        """Save current conversation history to file."""
        try:
            chat_history = []
            
            # Try different memory structures
            if hasattr(self.memory, 'chat_memory'):
                if hasattr(self.memory.chat_memory, 'messages'):
                    messages = self.memory.chat_memory.messages
                    if messages:
                        # Extract conversation history
                        i = 0
                        while i < len(messages):
                            if i + 1 < len(messages):
                                # Get human and AI message pairs
                                human_msg = str(messages[i].content) if hasattr(messages[i], 'content') else str(messages[i])
                                ai_msg = str(messages[i+1].content) if hasattr(messages[i+1], 'content') else str(messages[i+1])
                                chat_history.append({
                                    'human': human_msg,
                                    'ai': ai_msg
                                })
                                i += 2
                            else:
                                i += 1
                elif hasattr(self.memory.chat_memory, 'human'):
                    # Try getting from buffer
                    if hasattr(self.memory, 'buffer'):
                        buffer = self.memory.buffer
                        if buffer:
                            # Parse buffer for conversation pairs
                            lines = buffer.split('\n')
                            i = 0
                            while i < len(lines):
                                if i + 1 < len(lines) and ('Human:' in lines[i] or 'AI:' in lines[i]):
                                    human_line = lines[i] if 'Human:' in lines[i] else None
                                    ai_line = lines[i+1] if 'AI:' in lines[i+1] else None
                                    if human_line and ai_line:
                                        chat_history.append({
                                            'human': human_line.replace('Human:', '').strip(),
                                            'ai': ai_line.replace('AI:', '').strip()
                                        })
                                    i += 2
                                else:
                                    i += 1
            
            # Also try buffer directly
            if not chat_history and hasattr(self.memory, 'buffer'):
                buffer = self.memory.buffer
                if buffer:
                    # Simple parsing of buffer
                    parts = buffer.split('\n\n')
                    for part in parts:
                        if 'Human:' in part and 'AI:' in part:
                            human_part = part.split('AI:')[0].replace('Human:', '').strip()
                            ai_part = part.split('AI:')[1].strip()
                            if human_part and ai_part:
                                chat_history.append({
                                    'human': human_part,
                                    'ai': ai_part
                                })
            
            # Save to file if we have conversation history
            if chat_history:
                with open(self.memory_file, 'w', encoding='utf-8') as f:
                    json.dump({'chat_history': chat_history}, f, indent=2, ensure_ascii=False)
        except Exception as e:
            # Silently fail - don't interrupt the main flow
            pass
    
    def _create_prompt_template(self) -> PromptTemplate:
        """Create a custom prompt template for RAG."""
        template = """Use the following pieces of context from a mathematics textbook to answer the question.
If you don't know the answer based on the context, say that you don't know, don't try to make up an answer.

Context: {context}

Question: {question}

Provide a detailed and accurate answer based on the context provided. If the context mentions images, diagrams, or formulas, reference them in your answer.
Answer:"""
        
        return PromptTemplate(
            template=template,
            input_variables=["context", "question"]
        )
    
    def _create_summarizer_prompt_template(self) -> PromptTemplate:
        """Create a prompt template for summarization."""
        template = """Summarize the following retrieved context from a mathematics textbook in a concise manner.
Focus on the key concepts, formulas, and important information.

Context: {context}

Concise Summary:"""
        
        return PromptTemplate(
            template=template,
            input_variables=["context"]
        )
    
    def summarize_context(self, documents: List[Document]) -> str:
        """
        Summarize retrieved context using Ollama.
        """
        if not documents:
            return "No context retrieved."
        
        # Combine all document contents
        context_text = "\n\n".join([doc.page_content for doc in documents])
        
        # Truncate if too long (to avoid token limits)
        if len(context_text) > 4000:
            context_text = context_text[:4000] + "..."
        
        # Create summarization chain
        summarizer_prompt = self._create_summarizer_prompt_template()
        
        summary = self.llm.invoke(
            summarizer_prompt.format(context=context_text)
        )
        
        return summary.strip()
    
    def query_with_summarization(
        self, 
        question: str, 
        k: int = 4,
        summarize: bool = False
    ) -> Dict[str, Any]:
        """
        Query with optional summarization of retrieved context.
        """
        print(f"Query: {question}\n")
        print("Retrieving relevant documents...")
        
        # Retrieve relevant documents
        docs = self.vector_store.similarity_search_with_score(question, k=k)
        
        retrieved_docs = [doc for doc, score in docs]
        scores = [score for doc, score in docs]
        
        print(f"Retrieved {len(retrieved_docs)} relevant documents")
        print(f"Similarity scores: {[f'{s:.4f}' for s in scores]}\n")
        
        result = {
            "question": question,
            "retrieved_documents": retrieved_docs,
            "scores": scores,
            "summary": None,
            "answer": None,
            "sources": []
        }
        
        # Generate summary if requested
        if summarize and retrieved_docs:
            print("Generating summary of retrieved context...")
            summary = self.summarize_context(retrieved_docs)
            result["summary"] = summary
            try:
                print(f"Retrieved Context Summary:\n{summary}\n")
            except UnicodeEncodeError:
                # Handle unicode characters that can't be encoded in Windows console
                summary_safe = summary.encode('ascii', 'ignore').decode('ascii')
                print(f"Retrieved Context Summary:\n{summary_safe}\n")
        
        # Create RAG chain using simpler approach
        retriever = self.vector_store.as_retriever(search_kwargs={"k": k})
        
        # If summarization is enabled, modify the question
        if summarize and result["summary"]:
            modified_question = f"Based on this summary: {result['summary']}\n\nOriginal question: {question}"
        else:
            modified_question = question
        
        # Get context documents
        source_docs = retriever.invoke(modified_question)
        context = "\n\n".join([doc.page_content for doc in source_docs])
        
        # Create the prompt template
        prompt_template = self._create_prompt_template()
        formatted_prompt = prompt_template.format(context=context, question=modified_question)
        
        # Invoke LLM
        answer = self.llm.invoke(formatted_prompt)
        result["answer"] = answer
        
        # Add source information
        for doc in source_docs:
            source_info = {
                "page": doc.metadata.get("page", "unknown"),
                "chunk_preview": doc.page_content[:200] + "...",
                "has_images": len(doc.metadata.get("images", [])) > 0
            }
            result["sources"].append(source_info)
        
        return result
    
    def query_conversational(
        self,
        question: str,
        k: int = 4
    ) -> Dict[str, Any]:
        """
        Query with conversational memory for follow-up questions.
        """
        print(f"Query: {question}\n")
        
        try:
            # Try using ConversationalRetrievalChain if available
            if ConversationalRetrievalChain is not None:
                prompt = self._create_prompt_template()
                
                conversational_chain = ConversationalRetrievalChain.from_llm(
                    llm=self.llm,
                    retriever=self.vector_store.as_retriever(search_kwargs={"k": k}),
                    memory=self.memory,
                    combine_docs_chain_kwargs={"prompt": prompt},
                    return_source_documents=True,
                    verbose=False
                )
                
                response = conversational_chain.invoke({"question": question})
                
                result = {
                    "question": question,
                    "answer": response.get("answer", "No answer generated"),
                    "chat_history": [],
                    "sources": []
                }
                
                # Extract source information
                if "source_documents" in response:
                    for doc in response["source_documents"]:
                        source_info = {
                            "page": doc.metadata.get("page", "unknown"),
                            "chunk_preview": doc.page_content[:200] + "...",
                            "has_images": len(doc.metadata.get("images", [])) > 0
                        }
                        result["sources"].append(source_info)
            else:
                # Manual conversational implementation
                # Get previous conversation context
                previous_context = ""
                if hasattr(self.memory, 'chat_memory') and hasattr(self.memory.chat_memory, 'messages'):
                    messages = self.memory.chat_memory.messages
                    if messages:
                        # Build context from previous messages
                        context_parts = []
                        for msg in messages[-4:]:  # Last 2 exchanges (4 messages)
                            content = str(msg.content) if hasattr(msg, 'content') else str(msg)
                            context_parts.append(content)
                        if context_parts:
                            previous_context = "\n\nPrevious conversation:\n" + "\n".join(context_parts)
                
                # Retrieve relevant documents
                retriever = self.vector_store.as_retriever(search_kwargs={"k": k})
                source_docs = retriever.invoke(question)
                context = "\n\n".join([doc.page_content for doc in source_docs])
                
                # Create enhanced prompt with conversation history
                prompt_template = self._create_prompt_template()
                enhanced_question = f"{question}{previous_context}"
                formatted_prompt = prompt_template.format(context=context, question=enhanced_question)
                
                # Invoke LLM
                answer = self.llm.invoke(formatted_prompt)
                
                # Save to memory
                try:
                    self.memory.chat_memory.add_user_message(question)
                    self.memory.chat_memory.add_ai_message(answer)
                except:
                    pass
                
                result = {
                    "question": question,
                    "answer": answer,
                    "chat_history": [],
                    "sources": []
                }
                
                # Add source information
                for doc in source_docs:
                    source_info = {
                        "page": doc.metadata.get("page", "unknown"),
                        "chunk_preview": doc.page_content[:200] + "...",
                        "has_images": len(doc.metadata.get("images", [])) > 0
                    }
                    result["sources"].append(source_info)
            
            # Extract chat history if available
            if hasattr(self.memory, 'chat_memory'):
                try:
                    result["chat_history"] = [str(msg) for msg in self.memory.chat_memory.messages]
                except:
                    pass
            
            # Save conversation history for persistence
            self._save_memory()
            
            return result
            
        except Exception as e:
            print(f"Error in conversational query: {e}")
            # Fallback to non-conversational query
            print("Falling back to standard RAG query...")
            return self.query_with_summarization(question, k=k, summarize=False)


def print_result(result: Dict[str, Any], show_summary: bool = False):
    """Pretty print the query result."""
    print("=" * 60)
    if show_summary and result.get("summary"):
        print("RETRIEVED CONTEXT SUMMARY:")
        print("-" * 60)
        try:
            print(result["summary"])
        except UnicodeEncodeError:
            # Handle unicode characters that can't be encoded in Windows console
            summary_safe = result["summary"].encode('ascii', 'ignore').decode('ascii')
            print(summary_safe)
        print("\n" + "-" * 60)
    
    print("FINAL ANSWER:")
    print("-" * 60)
    print(result["answer"])
    print("\n" + "-" * 60)
    
    if result.get("sources"):
        print("SOURCES:")
        print("-" * 60)
        for i, source in enumerate(result["sources"], 1):
            print(f"{i}. Page {source['page']}")
            if source["has_images"]:
                print(f"   (Contains images/diagrams)")
            try:
                print(f"   Preview: {source['chunk_preview']}")
            except UnicodeEncodeError:
                # Handle unicode characters that can't be encoded in Windows console
                preview = source['chunk_preview'].encode('ascii', 'ignore').decode('ascii')
                print(f"   Preview: {preview}")
            print()
    
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Query the RAG pipeline")
    parser.add_argument(
        "--question", "-q",
        type=str,
        required=True,
        help="Question to ask about the PDF content"
    )
    parser.add_argument(
        "--summarize", "-s",
        action="store_true",
        help="Enable summarization of retrieved context"
    )
    parser.add_argument(
        "--conversational", "-c",
        action="store_true",
        help="Enable conversational mode (for follow-up questions)"
    )
    parser.add_argument(
        "--k",
        type=int,
        default=4,
        help="Number of documents to retrieve (default: 4)"
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable prompt caching"
    )
    parser.add_argument(
        "--session-id",
        type=str,
        default="default",
        help="Session ID for persistent conversational memory (default: 'default')"
    )
    
    args = parser.parse_args()
    
    # Initialize RAG pipeline
    pipeline = RAGPipeline(enable_cache=not args.no_cache, session_id=args.session_id)
    
    # Execute query
    if args.conversational:
        result = pipeline.query_conversational(args.question, k=args.k)
        print("\nNote: Conversational mode enabled. Previous context is remembered.")
    else:
        result = pipeline.query_with_summarization(
            args.question,
            k=args.k,
            summarize=args.summarize
        )
    
    # Print result
    print_result(result, show_summary=args.summarize)


if __name__ == "__main__":
    main()

