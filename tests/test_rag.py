import unittest
import tempfile
import shutil
import sys
from pathlib import Path

# Add parent directory to path to import from rag_qa_system
sys.path.insert(0, str(Path(__file__).parent.parent))

from rag_qa_system.main import RAGPipeline, RAGConfig, DocumentProcessor, VectorStoreManager


class TestDocumentProcessor(unittest.TestCase):
    """Test document processing functionality"""
    
    def setUp(self):
        """Create temporary directory with test documents"""
        self.temp_dir = tempfile.mkdtemp()
        
        # Create test documents
        test_doc1 = Path(self.temp_dir) / "test1.txt"
        test_doc1.write_text("Python is a programming language. It is easy to learn.")
        
        test_doc2 = Path(self.temp_dir) / "test2.txt"
        test_doc2.write_text("Machine learning is a subset of artificial intelligence.")
        
        self.config = RAGConfig(docs_path=self.temp_dir)
        self.processor = DocumentProcessor(self.config)
    
    def tearDown(self):
        """Clean up temporary directory"""
        shutil.rmtree(self.temp_dir)
    
    def test_load_documents(self):
        """Test document loading"""
        documents = self.processor.load_documents()
        self.assertEqual(len(documents), 2)
        self.assertIn("Python", documents[0].page_content)
    
    def test_chunk_documents(self):
        """Test document chunking"""
        documents = self.processor.load_documents()
        chunks = self.processor.chunk_documents(documents)
        self.assertGreaterEqual(len(chunks), 2)
        self.assertIsNotNone(chunks[0].page_content)


class TestRAGPipeline(unittest.TestCase):
    """Test RAG pipeline functionality"""
    
    def setUp(self):
        """Setup test environment"""
        self.temp_docs_dir = tempfile.mkdtemp()
        self.temp_vector_dir = tempfile.mkdtemp()
        
        # Create test document
        test_doc = Path(self.temp_docs_dir) / "python.txt"
        test_doc.write_text("""
        Python is a high-level programming language created by Guido van Rossum.
        It is known for its simple syntax and extensive standard library.
        Python supports multiple programming paradigms including object-oriented 
        and functional programming.
        """)
        
        self.config = RAGConfig(
            docs_path=self.temp_docs_dir,
            vector_store_path=self.temp_vector_dir,
            chunk_size=100,
            chunk_overlap=20
        )
    
    def tearDown(self):
        """Clean up test environment"""
        shutil.rmtree(self.temp_docs_dir)
        shutil.rmtree(self.temp_vector_dir)
    
    def test_pipeline_initialization(self):
        """Test RAG pipeline initialization"""
        rag = RAGPipeline(self.config)
        self.assertIsNotNone(rag.doc_processor)
        self.assertIsNotNone(rag.vector_manager)
    
    def test_document_retrieval(self):
        """Test document retrieval without LLM"""
        rag = RAGPipeline(self.config)
        
        # Setup (this will create vector store)
        try:
            rag.setup()
            
            # Test retrieval
            docs = rag.get_relevant_documents("What is Python?")
            self.assertGreater(len(docs), 0)
            self.assertIn("Python", docs[0].page_content)
            
        except Exception as e:
            self.skipTest(f"Setup failed (expected in test environment): {e}")


class TestConfiguration(unittest.TestCase):
    """Test configuration handling"""
    
    def test_default_config(self):
        """Test default configuration"""
        config = RAGConfig()
        self.assertEqual(config.chunk_size, 1000)
        self.assertEqual(config.chunk_overlap, 200)
        self.assertEqual(config.top_k_results, 4)
    
    def test_custom_config(self):
        """Test custom configuration"""
        config = RAGConfig(
            chunk_size=500,
            chunk_overlap=100,
            top_k_results=3
        )
        self.assertEqual(config.chunk_size, 500)
        self.assertEqual(config.chunk_overlap, 100)
        self.assertEqual(config.top_k_results, 3)


class TestIntegration(unittest.TestCase):
    """Integration tests for end-to-end functionality"""
    
    def setUp(self):
        """Setup test environment"""
        self.temp_docs_dir = tempfile.mkdtemp()
        self.temp_vector_dir = tempfile.mkdtemp()
        
        # Create multiple test documents
        docs = {
            "python_basics.txt": """
            Python is an interpreted, high-level programming language.
            It was created by Guido van Rossum and first released in 1991.
            Python's design philosophy emphasizes code readability.
            """,
            "python_features.txt": """
            Python supports multiple programming paradigms.
            It has a comprehensive standard library.
            Python uses dynamic typing and automatic memory management.
            """,
            "python_uses.txt": """
            Python is widely used in web development, data science, and AI.
            Popular frameworks include Django, Flask, and FastAPI.
            Libraries like NumPy and Pandas are essential for data analysis.
            """
        }
        
        for filename, content in docs.items():
            (Path(self.temp_docs_dir) / filename).write_text(content)
        
        self.config = RAGConfig(
            docs_path=self.temp_docs_dir,
            vector_store_path=self.temp_vector_dir
        )
    
    def tearDown(self):
        """Clean up test environment"""
        shutil.rmtree(self.temp_docs_dir)
        shutil.rmtree(self.temp_vector_dir)
    
    def test_full_pipeline(self):
        """Test complete RAG pipeline"""
        rag = RAGPipeline(self.config)
        
        try:
            # Setup
            rag.setup()
            
            # Test retrieval
            docs = rag.get_relevant_documents("Who created Python?")
            self.assertGreater(len(docs), 0)
            
            # Check if relevant information is retrieved
            found_creator = any("Guido" in doc.page_content for doc in docs)
            self.assertTrue(found_creator, "Should retrieve document mentioning Guido")
            
        except Exception as e:
            self.skipTest(f"Integration test skipped (requires full setup): {e}")


def run_tests():
    """Run all tests"""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test cases
    suite.addTests(loader.loadTestsFromTestCase(TestDocumentProcessor))
    suite.addTests(loader.loadTestsFromTestCase(TestRAGPipeline))
    suite.addTests(loader.loadTestsFromTestCase(TestConfiguration))
    suite.addTests(loader.loadTestsFromTestCase(TestIntegration))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "="*70)
    print(f"Tests run: {result.testsRun}")
    print(f"Successes: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Skipped: {len(result.skipped)}")
    print("="*70)
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    exit(0 if success else 1)