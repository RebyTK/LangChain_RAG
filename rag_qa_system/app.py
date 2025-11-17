import streamlit as st
from pathlib import Path
from datetime import datetime
import time
import sys

# Ensure current directory is in path for imports
sys.path.insert(0, str(Path(__file__).parent))

try:
    from main import RAGPipeline, RAGConfig
except ImportError as e:
    st.error(f"Error: Could not import RAG modules: {e}")
    st.error("Make sure main.py is in the same directory and all dependencies are installed.")
    st.error(f"Python path: {sys.path[:3]}")
    st.stop()


# Page configuration
st.set_page_config(
    page_title="RAG Q&A System",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)


# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1E88E5;
        text-align: center;
        padding: 1rem 0;
    }
    .sub-header {
        text-align: center;
        color: #666;
        padding-bottom: 2rem;
    }
    .answer-box {
        background-color: #000000;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #1E88E5;
        margin: 1rem 0;
    }
    .source-box {
        background-color: #000000;
        padding: 1rem;
        border-radius: 8px;
        border-left: 3px solid #FFA726;
        margin: 0.5rem 0;
    }
    .stat-card {
        background-color: #f5f5f5;
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
    }
    /* Fix text input visibility */
    .stTextInput > div > div > input {
        background-color: white;
        color: #262730;
        border: 1px solid #d1d5db;
    }
    .stTextInput > div > div > input:focus {
        border-color: #1E88E5;
        box-shadow: 0 0 0 2px rgba(30, 136, 229, 0.2);
    }
    .stTextInput > label {
        color: #262730;
        font-weight: 500;
    }
    /* Ensure text is visible in input placeholder */
    .stTextInput > div > div > input::placeholder {
        color: #9ca3af;
    }
    </style>
""", unsafe_allow_html=True)


# Initialize session state
if 'rag_pipeline' not in st.session_state:
    st.session_state.rag_pipeline = None
if 'initialized' not in st.session_state:
    st.session_state.initialized = False
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'config' not in st.session_state:
    st.session_state.config = None


def get_config_value(attr: str, fallback):
    """Helper to read from session config with fallback."""
    if st.session_state.config and hasattr(st.session_state.config, attr):
        return getattr(st.session_state.config, attr)
    return fallback


def initialize_rag(config: RAGConfig, force_reindex: bool = False):
    """Initialize the RAG pipeline"""
    try:
        with st.spinner("🔧 Initializing RAG system..."):
            rag = RAGPipeline(config)
            rag.setup(force_reindex=force_reindex)
            st.session_state.rag_pipeline = rag
            st.session_state.initialized = True
            st.session_state.config = config
            st.success("✅ System initialized successfully!")
            return True
    except Exception as e:
        st.error(f"❌ Initialization failed: {str(e)}")
        return False


def display_answer(result: dict):
    """Display answer with sources in a beautiful format"""
    # Answer
    st.markdown("### 💡 Answer")
    st.markdown(f"""
        <div class="answer-box">
            {result['answer']}
        </div>
    """, unsafe_allow_html=True)
    
    # Sources
    if result.get('sources'):
        st.markdown("### 📚 Sources")
        
        for source in result['sources']:
            source_name = Path(source['metadata'].get('source', 'Unknown')).name
            
            with st.expander(f"📄 Source {source['id']}: {source_name}"):
                st.markdown(f"""
                    <div class="source-box">
                        <strong>Content Preview:</strong><br>
                        {source['content']}
                    </div>
                """, unsafe_allow_html=True)
                
                # Show full metadata
                if source['metadata']:
                    st.json(source['metadata'])


def main():
    """Main Streamlit application"""
    
    # Header
    st.markdown('<div class="main-header">🤖 RAG Question-Answering System</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Powered by LangChain | Retrieval-Augmented Generation</div>', unsafe_allow_html=True)
    
    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        docs_path = st.text_input(
            "Documents Path",
            value="./data/documents",
            help="Path to the directory containing your documents"
        )
        
        vector_store_path = st.text_input(
            "Vector Store Path",
            value="./data/vector_store",
            help="Path where vector embeddings will be stored"
        )
        
        chunk_size = st.slider(
            "Chunk Size",
            min_value=200,
            max_value=2000,
            value=1000,
            step=100,
            help="Size of document chunks for processing"
        )
        
        chunk_overlap = st.slider(
            "Chunk Overlap",
            min_value=0,
            max_value=500,
            value=200,
            step=50,
            help="Overlap between consecutive chunks"
        )
        
        top_k = st.slider(
            "Top-K Results",
            min_value=1,
            max_value=10,
            value=4,
            help="Number of relevant documents to retrieve"
        )
        
        temperature = st.slider(
            "LLM Temperature",
            min_value=0.0,
            max_value=1.0,
            value=0.0,
            step=0.1,
            help="Creativity level (0=deterministic, 1=creative)"
        )
        
        llm_model = st.text_input(
            "LLM Model (Ollama)",
            value=get_config_value('llm_model', RAGConfig().llm_model),
            help="Name of the Ollama model to use (e.g., llama2, llama3:instruct)"
        )
        
        ollama_base_url = st.text_input(
            "Ollama Base URL",
            value=get_config_value('ollama_base_url', RAGConfig().ollama_base_url),
            help="Address where the Ollama server is running"
        )
        
        st.divider()
        
        # Initialize button
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🚀 Initialize", use_container_width=True):
                config = RAGConfig(
                    docs_path=docs_path,
                    vector_store_path=vector_store_path,
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                    top_k_results=top_k,
                    llm_temperature=temperature,
                    llm_model=llm_model.strip() or RAGConfig().llm_model,
                    ollama_base_url=ollama_base_url.strip() or RAGConfig().ollama_base_url
                )
                initialize_rag(config)
        
        with col2:
            if st.button("🔄 Reindex", use_container_width=True):
                if st.session_state.config:
                    initialize_rag(st.session_state.config, force_reindex=True)
                else:
                    st.warning("Please initialize first")
        
        # Statistics
        if st.session_state.initialized:
            st.divider()
            st.header("📊 Statistics")
            
            st.metric("Queries Processed", len(st.session_state.chat_history))
            st.metric("Documents Path", Path(st.session_state.config.docs_path).name)
            st.metric("Top-K Results", st.session_state.config.top_k_results)
        
        # Clear history
        if st.button("🗑️ Clear History", use_container_width=True):
            st.session_state.chat_history = []
            st.rerun()
    
    # Main content area
    if not st.session_state.initialized:
        st.info("👈 Please configure settings in the sidebar and click 'Initialize' to start")
        
        # Quick start guide
        with st.expander("📖 Quick Start Guide"):
            st.markdown("""
                ### Getting Started
                
                1. **Configure Settings**: Adjust the configuration in the sidebar
                2. **Prepare Documents**: Place your documents in the specified directory
                3. **Initialize**: Click the 'Initialize' button to set up the system
                4. **Ask Questions**: Type your questions in the chat interface
                
                ### Supported File Types
                - `.txt` - Plain text files
                - `.md` - Markdown files
                
                ### Tips
                - Use specific questions for better answers
                - Check the sources to verify information
                - Adjust Top-K to retrieve more/fewer documents
                """)
        
        return
    
    # Chat interface
    st.header("💬 Chat Interface")
    
    # Display chat history
    for i, entry in enumerate(st.session_state.chat_history):
        with st.container():
            # Question
            st.markdown(f"**🙋 You:** {entry['question']}")
            
            # Answer
            display_answer(entry)
            
            st.divider()
    
    # Check if a suggestion was selected
    selected_question = st.session_state.get("selected_suggestion", "")
    
    # Question input
    question = st.text_input(
        "Ask a question",
        value=selected_question if selected_question else "",
        placeholder="Type your question here...",
        key="question_input"
    )
    
    # Clear the selected suggestion after using it
    if selected_question:
        st.session_state.selected_suggestion = ""
    
    col1, col2 = st.columns([1, 5])
    
    with col1:
        ask_button = st.button("Ask", use_container_width=True, type="primary")
    
    if ask_button and question:
        try:
            # Show spinner while processing
            with st.spinner("🤔 Thinking..."):
                start_time = time.time()
                result = st.session_state.rag_pipeline.query(question)
                end_time = time.time()
                
                result['timestamp'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                result['response_time'] = f"{end_time - start_time:.2f}s"
                
                # Add to history
                st.session_state.chat_history.append(result)
            
            # Rerun to display new result
            st.rerun()
            
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
    
    # Suggested questions
    if not st.session_state.chat_history:
        st.markdown("### 💡 Suggested Questions")
        
        suggestions = [
            "What is Python?",
            "How do I use list comprehensions?",
            "What are decorators?",
            "Explain Python classes"
        ]
        
        cols = st.columns(2)
        for i, suggestion in enumerate(suggestions):
            with cols[i % 2]:
                if st.button(suggestion, use_container_width=True, key=f"suggestion_{i}"):
                    # Store suggestion in session state and process it directly
                    st.session_state.selected_suggestion = suggestion
                    # Process the question immediately
                    try:
                        with st.spinner("🤔 Thinking..."):
                            start_time = time.time()
                            result = st.session_state.rag_pipeline.query(suggestion)
                            end_time = time.time()
                            
                            result['timestamp'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                            result['response_time'] = f"{end_time - start_time:.2f}s"
                            
                            st.session_state.chat_history.append(result)
                        st.rerun()
                    except Exception as e:
                        st.error(f"❌ Error: {str(e)}")
    
    # Footer
    st.divider()
    st.markdown("""
        <div style='text-align: center; color: #666; padding: 1rem;'>
            Built with ❤️ using LangChain | Streamlit | HuggingFace
        </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()