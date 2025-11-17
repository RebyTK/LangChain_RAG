import os
import logging
from pathlib import Path
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, field

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.schema import Document

# Try to import Ollama - handle different langchain versions
try:
    from langchain_community.llms import Ollama
except ImportError:
    try:
        from langchain.llms import Ollama
    except ImportError:
        raise ImportError(
            "Could not import Ollama. Please ensure langchain-community is installed "
            "and compatible with your langchain version. Try: pip install langchain-community"
        )

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def _resolve_path(path: str) -> str:
    """Resolve relative paths to absolute paths relative to project root"""
    if os.path.isabs(path):
        return path
    # Try to find project root (directory containing 'rag_qa_system' or 'data')
    current = Path(__file__).parent.resolve()
    # If we're in rag_qa_system, go up one level
    if current.name == "rag_qa_system":
        project_root = current.parent
    else:
        project_root = current
    # Resolve the path relative to project root
    resolved = (project_root / path).resolve()
    return str(resolved)


@dataclass
class RAGConfig:
    """Configuration for RAG pipeline"""
    docs_path: str = "./data/documents"
    vector_store_path: str = "./data/vector_store"
    chunk_size: int = 1000
    chunk_overlap: int = 200
    top_k_results: int = 4
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    llm_temperature: float = 0.0
    llm_model: str = field(default_factory=lambda: os.getenv("OLLAMA_MODEL", "llama2"))
    ollama_base_url: str = field(default_factory=lambda: os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"))
    
    def __post_init__(self):
        """Resolve paths after initialization"""
        self.docs_path = _resolve_path(self.docs_path)
        self.vector_store_path = _resolve_path(self.vector_store_path)


class DocumentProcessor:
    """Handles document loading and chunking"""
    
    def __init__(self, config: RAGConfig):
        self.config = config
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
    
    def load_documents(self) -> List[Document]:
        """Load documents from the specified directory"""
        try:
            docs_path = Path(self.config.docs_path)
            logger.info(f"Loading documents from {docs_path} (absolute: {docs_path.resolve()})")
            
            # Check if directory exists
            if not docs_path.exists():
                raise FileNotFoundError(
                    f"Documents directory not found: {docs_path.resolve()}\n"
                    f"Please ensure the directory exists and contains .txt or .md files."
                )
            
            if not docs_path.is_dir():
                raise ValueError(f"Path is not a directory: {docs_path.resolve()}")
            
            # Check for files
            txt_files = list(docs_path.glob("**/*.txt"))
            md_files = list(docs_path.glob("**/*.md"))
            logger.info(f"Found {len(txt_files)} .txt files and {len(md_files)} .md files")
            
            if not txt_files and not md_files:
                raise ValueError(
                    f"No .txt or .md files found in {docs_path.resolve()}\n"
                    f"Please add documents to this directory."
                )
            
            # Load files individually for better error handling and encoding support
            documents = []
            all_files = list(docs_path.glob("**/*.txt")) + list(docs_path.glob("**/*.md"))
            
            for file_path in all_files:
                try:
                    # Try UTF-8 first, fallback to other encodings if needed
                    try:
                        loader = TextLoader(str(file_path), encoding="utf-8")
                    except:
                        # Fallback to default encoding
                        loader = TextLoader(str(file_path))
                    
                    docs = loader.load()
                    documents.extend(docs)
                    logger.debug(f"Successfully loaded: {file_path.name}")
                except Exception as e:
                    logger.warning(f"Failed to load {file_path.name}: {e}")
                    # Try with different encoding as last resort
                    try:
                        loader = TextLoader(str(file_path), encoding="latin-1")
                        docs = loader.load()
                        documents.extend(docs)
                        logger.info(f"Loaded {file_path.name} with latin-1 encoding")
                    except:
                        logger.error(f"Could not load {file_path.name} with any encoding")
            
            logger.info(f"Total loaded: {len(documents)} documents")
            return documents
            
        except Exception as e:
            logger.error(f"Error loading documents: {e}")
            raise
    
    def chunk_documents(self, documents: List[Document]) -> List[Document]:
        """Split documents into chunks"""
        try:
            logger.info("Chunking documents...")
            chunks = self.text_splitter.split_documents(documents)
            logger.info(f"Created {len(chunks)} chunks")
            return chunks
        except Exception as e:
            logger.error(f"Error chunking documents: {e}")
            raise


class VectorStoreManager:
    """Manages vector store operations"""
    
    def __init__(self, config: RAGConfig):
        self.config = config
        self.embeddings = HuggingFaceEmbeddings(
            model_name=config.model_name
        )
        self.vector_store = None
    
    def create_vector_store(self, chunks: List[Document]) -> Chroma:
        """Create and persist vector store from document chunks"""
        try:
            logger.info("Creating vector store...")
            
            # Create vector store directory if it doesn't exist
            Path(self.config.vector_store_path).mkdir(parents=True, exist_ok=True)
            
            self.vector_store = Chroma.from_documents(
                documents=chunks,
                embedding=self.embeddings,
                persist_directory=self.config.vector_store_path
            )
            
            logger.info(f"Vector store created with {len(chunks)} chunks")
            return self.vector_store
            
        except Exception as e:
            logger.error(f"Error creating vector store: {e}")
            raise
    
    def load_vector_store(self) -> Chroma:
        """Load existing vector store"""
        try:
            logger.info("Loading existing vector store...")
            
            self.vector_store = Chroma(
                persist_directory=self.config.vector_store_path,
                embedding_function=self.embeddings
            )
            
            logger.info("Vector store loaded successfully")
            return self.vector_store
            
        except Exception as e:
            logger.error(f"Error loading vector store: {e}")
            raise


class RAGPipeline:
    """Main RAG pipeline orchestrator"""
    
    def __init__(self, config: Optional[RAGConfig] = None):
        self.config = config or RAGConfig()
        self.doc_processor = DocumentProcessor(self.config)
        self.vector_manager = VectorStoreManager(self.config)
        self.qa_chain = None
        
        # Custom prompt template for better responses
        self.prompt_template = """Use the following pieces of context to answer the question at the end. 
If you don't know the answer based on the context, just say that you don't know, don't try to make up an answer.

Context:
{context}

Question: {question}

Answer: Let me provide a detailed answer based on the context above."""

    def setup(self, force_reindex: bool = False):
        """Setup the RAG pipeline"""
        try:
            vector_store_exists = Path(self.config.vector_store_path).exists()
            
            if force_reindex or not vector_store_exists:
                logger.info("Indexing documents...")
                
                # Load and process documents
                documents = self.doc_processor.load_documents()
                if not documents:
                    raise ValueError("No documents found to index")
                
                chunks = self.doc_processor.chunk_documents(documents)
                
                # Create vector store
                vector_store = self.vector_manager.create_vector_store(chunks)
            else:
                logger.info("Using existing vector store")
                vector_store = self.vector_manager.load_vector_store()
            
            # Create retrieval chain
            self._create_qa_chain(vector_store)
            
            logger.info("RAG pipeline setup complete")
            
        except Exception as e:
            logger.error(f"Error setting up RAG pipeline: {e}")
            raise
    
    def _create_qa_chain(self, vector_store: Chroma):
        """Create the question-answering chain"""
        try:
            # Create custom prompt
            PROMPT = PromptTemplate(
                template=self.prompt_template,
                input_variables=["context", "question"]
            )
            
            # Initialize local LLM via Ollama
            llm = Ollama(
                model=self.config.llm_model,
                base_url=self.config.ollama_base_url,
                temperature=self.config.llm_temperature
            )
            
            # Create retrieval QA chain
            self.qa_chain = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=vector_store.as_retriever(
                    search_kwargs={"k": self.config.top_k_results}
                ),
                return_source_documents=True,
                chain_type_kwargs={"prompt": PROMPT}
            )
            
        except Exception as e:
            logger.error(f"Error creating QA chain: {e}")
            raise
    
    def query(self, question: str) -> Dict[str, Any]:
        """
        Query the RAG system
        
        Args:
            question: User question
            
        Returns:
            Dictionary with answer and source documents
        """
        try:
            if not self.qa_chain:
                raise ValueError("RAG pipeline not setup. Call setup() first.")
            
            logger.info(f"Processing query: {question}")
            
            # Get answer using invoke (replaces deprecated __call__)
            result = self.qa_chain.invoke({"query": question})
            
            # Format response
            response = {
                "question": question,
                "answer": result["result"],
                "sources": []
            }
            
            # Add source citations
            if "source_documents" in result:
                for i, doc in enumerate(result["source_documents"], 1):
                    source_info = {
                        "id": i,
                        "content": doc.page_content[:200] + "...",
                        "metadata": doc.metadata
                    }
                    response["sources"].append(source_info)
            
            logger.info("Query processed successfully")
            return response
            
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            raise
    
    def get_relevant_documents(self, question: str) -> List[Document]:
        """Retrieve relevant documents without generating an answer"""
        try:
            if not self.vector_manager.vector_store:
                raise ValueError("Vector store not initialized")
            
            retriever = self.vector_manager.vector_store.as_retriever(
                search_kwargs={"k": self.config.top_k_results}
            )
            
            docs = retriever.get_relevant_documents(question)
            return docs
            
        except Exception as e:
            logger.error(f"Error retrieving documents: {e}")
            raise


def main():
    """Example usage of RAG pipeline"""
    
    # Initialize configuration
    config = RAGConfig(
        docs_path="./data/documents",
        vector_store_path="./data/vector_store",
        chunk_size=1000,
        chunk_overlap=200,
        top_k_results=4
    )
    
    # Create RAG pipeline
    rag = RAGPipeline(config)
    
    # Setup (will index documents or load existing index)
    rag.setup(force_reindex=False)
    
    # Example queries
    questions = [
        "What is Python?",
        "How do I use list comprehensions?",
        "What are Python decorators?"
    ]
    
    for question in questions:
        print(f"\n{'='*80}")
        print(f"Question: {question}")
        print(f"{'='*80}")
        
        result = rag.query(question)
        
        print(f"\nAnswer:\n{result['answer']}")
        
        if result['sources']:
            print(f"\n\nSources:")
            for source in result['sources']:
                print(f"\n[{source['id']}] {source['metadata'].get('source', 'Unknown')}")
                print(f"    {source['content']}")


if __name__ == "__main__":
    main()