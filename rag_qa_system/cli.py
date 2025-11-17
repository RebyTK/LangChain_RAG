import sys
import argparse
from typing import Optional
from pathlib import Path

# Ensure current directory is in path for imports
sys.path.insert(0, str(Path(__file__).parent))

try:
    from main import RAGPipeline, RAGConfig
except ImportError as e:
    print(f"Error: Could not import RAG modules: {e}")
    print("Make sure main.py is in the same directory and all dependencies are installed.")
    sys.exit(1)


class Colors:
    """ANSI color codes for terminal output"""
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


class RAGCLI:
    """Command-line interface for RAG system"""
    
    def __init__(self, config: Optional[RAGConfig] = None):
        self.config = config or RAGConfig()
        self.rag = RAGPipeline(self.config)
        self.history = []
    
    def print_banner(self):
        """Print welcome banner"""
        banner = f"""
{Colors.CYAN}{'='*80}
{Colors.BOLD}RAG Question-Answering System{Colors.ENDC}
{Colors.CYAN}Powered by LangChain | Retrieval-Augmented Generation
{'='*80}{Colors.ENDC}
        """
        print(banner)
    
    def print_help(self):
        """Print available commands"""
        help_text = f"""
{Colors.BOLD}Available Commands:{Colors.ENDC}
  {Colors.GREEN}ask <question>{Colors.ENDC}     - Ask a question
  {Colors.GREEN}history{Colors.ENDC}            - View query history
  {Colors.GREEN}sources <question>{Colors.ENDC} - View relevant sources without generating answer
  {Colors.GREEN}stats{Colors.ENDC}              - Show system statistics
  {Colors.GREEN}reindex{Colors.ENDC}            - Rebuild vector store from documents
  {Colors.GREEN}help{Colors.ENDC}               - Show this help message
  {Colors.GREEN}exit{Colors.ENDC}               - Exit the application
        """
        print(help_text)
    
    def initialize(self, force_reindex: bool = False):
        """Initialize the RAG system"""
        try:
            print(f"\n{Colors.BLUE}Initializing RAG system...{Colors.ENDC}")
            self.rag.setup(force_reindex=force_reindex)
            print(f"{Colors.GREEN}✓ System ready!{Colors.ENDC}\n")
            return True
        except Exception as e:
            print(f"{Colors.FAIL}✗ Initialization failed: {e}{Colors.ENDC}")
            return False
    
    def ask_question(self, question: str):
        """Process a question and display results"""
        try:
            print(f"\n{Colors.BLUE}Thinking...{Colors.ENDC}")
            
            result = self.rag.query(question)
            self.history.append(result)
            
            # Display answer
            print(f"\n{Colors.BOLD}{Colors.GREEN}Answer:{Colors.ENDC}")
            print(f"{result['answer']}")
            
            # Display sources
            if result['sources']:
                print(f"\n{Colors.BOLD}{Colors.CYAN}Sources:{Colors.ENDC}")
                for source in result['sources']:
                    source_name = source['metadata'].get('source', 'Unknown')
                    source_path = Path(source_name).name if source_name != 'Unknown' else 'Unknown'
                    print(f"\n  {Colors.WARNING}[{source['id']}]{Colors.ENDC} {source_path}")
                    print(f"      {source['content'][:150]}...")
            
            print()  # Extra newline for readability
            
        except Exception as e:
            print(f"{Colors.FAIL}Error processing question: {e}{Colors.ENDC}")
    
    def show_sources(self, question: str):
        """Show relevant sources without generating answer"""
        try:
            print(f"\n{Colors.BLUE}Retrieving relevant sources...{Colors.ENDC}")
            
            docs = self.rag.get_relevant_documents(question)
            
            print(f"\n{Colors.BOLD}Found {len(docs)} relevant documents:{Colors.ENDC}\n")
            
            for i, doc in enumerate(docs, 1):
                source_name = doc.metadata.get('source', 'Unknown')
                source_path = Path(source_name).name if source_name != 'Unknown' else 'Unknown'
                
                print(f"{Colors.WARNING}[{i}]{Colors.ENDC} {source_path}")
                print(f"    {doc.page_content[:200]}...")
                print()
            
        except Exception as e:
            print(f"{Colors.FAIL}Error retrieving sources: {e}{Colors.ENDC}")
    
    def show_history(self):
        """Display query history"""
        if not self.history:
            print(f"\n{Colors.WARNING}No queries in history yet.{Colors.ENDC}\n")
            return
        
        print(f"\n{Colors.BOLD}Query History:{Colors.ENDC}\n")
        for i, entry in enumerate(self.history, 1):
            print(f"{Colors.CYAN}[{i}]{Colors.ENDC} {entry['question']}")
            print(f"    {entry['answer'][:100]}...")
            print()
    
    def show_stats(self):
        """Show system statistics"""
        stats = f"""
{Colors.BOLD}System Statistics:{Colors.ENDC}
  Documents Path:    {self.config.docs_path}
  Vector Store Path: {self.config.vector_store_path}
  Chunk Size:        {self.config.chunk_size}
  Chunk Overlap:     {self.config.chunk_overlap}
  Top-K Results:     {self.config.top_k_results}
  Queries Processed: {len(self.history)}
        """
        print(stats)
    
    def reindex(self):
        """Rebuild vector store"""
        confirm = input(f"\n{Colors.WARNING}This will rebuild the entire vector store. Continue? (yes/no): {Colors.ENDC}")
        if confirm.lower() in ['yes', 'y']:
            print(f"\n{Colors.BLUE}Reindexing documents...{Colors.ENDC}")
            self.initialize(force_reindex=True)
        else:
            print(f"{Colors.WARNING}Reindexing cancelled.{Colors.ENDC}")
    
    def run_interactive(self):
        """Run interactive CLI session"""
        self.print_banner()
        
        if not self.initialize():
            return
        
        self.print_help()
        
        while True:
            try:
                # Get user input
                user_input = input(f"{Colors.BOLD}> {Colors.ENDC}").strip()
                
                if not user_input:
                    continue
                
                # Parse command
                parts = user_input.split(maxsplit=1)
                command = parts[0].lower()
                args = parts[1] if len(parts) > 1 else ""
                
                # Handle commands
                if command in ['exit', 'quit', 'q']:
                    print(f"\n{Colors.GREEN}Goodbye!{Colors.ENDC}\n")
                    break
                
                elif command == 'help':
                    self.print_help()
                
                elif command == 'ask':
                    if args:
                        self.ask_question(args)
                    else:
                        print(f"{Colors.WARNING}Please provide a question. Usage: ask <question>{Colors.ENDC}")
                
                elif command == 'sources':
                    if args:
                        self.show_sources(args)
                    else:
                        print(f"{Colors.WARNING}Please provide a question. Usage: sources <question>{Colors.ENDC}")
                
                elif command == 'history':
                    self.show_history()
                
                elif command == 'stats':
                    self.show_stats()
                
                elif command == 'reindex':
                    self.reindex()
                
                else:
                    # Assume it's a question if no command matches
                    self.ask_question(user_input)
            
            except KeyboardInterrupt:
                print(f"\n\n{Colors.GREEN}Goodbye!{Colors.ENDC}\n")
                break
            
            except Exception as e:
                print(f"{Colors.FAIL}Error: {e}{Colors.ENDC}")
    
    def run_single_query(self, question: str):
        """Run a single query (non-interactive mode)"""
        if not self.initialize():
            sys.exit(1)
        
        self.ask_question(question)


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="RAG Question-Answering System CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        'question',
        nargs='*',
        help='Question to ask (if provided, runs in single-query mode)'
    )
    
    parser.add_argument(
        '--docs-path',
        default='./data/documents',
        help='Path to documents directory'
    )
    
    parser.add_argument(
        '--vector-store-path',
        default='./data/vector_store',
        help='Path to vector store directory'
    )
    
    parser.add_argument(
        '--reindex',
        action='store_true',
        help='Force reindexing of documents'
    )
    
    parser.add_argument(
        '--chunk-size',
        type=int,
        default=1000,
        help='Document chunk size'
    )
    
    parser.add_argument(
        '--top-k',
        type=int,
        default=4,
        help='Number of results to retrieve'
    )
    
    parser.add_argument(
        '--llm-model',
        default=None,
        help='Ollama model to use for generation (default: llama2)'
    )
    
    parser.add_argument(
        '--ollama-base-url',
        default=None,
        help='Ollama server URL (default: http://localhost:11434)'
    )
    
    parser.add_argument(
        '--llm-temperature',
        type=float,
        default=None,
        help='Override LLM temperature (default: config value)'
    )
    
    args = parser.parse_args()
    
    # Create configuration
    config_kwargs = {
        "docs_path": args.docs_path,
        "vector_store_path": args.vector_store_path,
        "chunk_size": args.chunk_size,
        "top_k_results": args.top_k
    }
    
    if args.llm_model:
        config_kwargs["llm_model"] = args.llm_model
    if args.ollama_base_url:
        config_kwargs["ollama_base_url"] = args.ollama_base_url
    if args.llm_temperature is not None:
        config_kwargs["llm_temperature"] = args.llm_temperature
    
    config = RAGConfig(**config_kwargs)
    
    # Create CLI
    cli = RAGCLI(config)
    
    # Run in appropriate mode
    if args.question:
        # Single query mode
        question = ' '.join(args.question)
        cli.run_single_query(question)
    else:
        # Interactive mode
        cli.run_interactive()


if __name__ == "__main__":
    main()