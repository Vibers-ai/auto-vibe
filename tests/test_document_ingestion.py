"""Tests for document ingestion agent."""

import pytest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch

from src.agents.document_ingestion import DocumentIngestionAgent
from src.utils.config import Config


class TestDocumentIngestionAgent:
    """Test document ingestion functionality."""
    
    @pytest.fixture
    def mock_config(self):
        """Create a mock config for testing."""
        config = Mock(spec=Config)
        config.gemini_api_key = "test-key"
        config.gemini_model = "test-model"
        return config
    
    @pytest.fixture
    def temp_docs_dir(self):
        """Create a temporary directory with sample documents."""
        temp_dir = tempfile.mkdtemp()
        docs_path = Path(temp_dir) / "docs"
        docs_path.mkdir()
        
        # Create sample text file
        (docs_path / "requirements.md").write_text("""
# Project Requirements

This is a sample project requirements document.

## Features
- User authentication
- Task management
- API endpoints

## Technical Stack
- Backend: Python FastAPI
- Frontend: React
- Database: PostgreSQL
""")
        
        # Create sample plain text file
        (docs_path / "notes.txt").write_text("""
Additional notes:
- Use JWT for authentication
- Implement REST API
- Add unit tests
""")
        
        yield str(docs_path)
        
        # Cleanup
        shutil.rmtree(temp_dir)
    
    @patch('src.agents.document_ingestion.genai')
    def test_document_collection(self, mock_genai, mock_config, temp_docs_dir):
        """Test document collection functionality."""
        agent = DocumentIngestionAgent(mock_config)
        
        # Test document collection
        docs_dir = Path(temp_docs_dir)
        documents = agent._collect_documents(docs_dir)
        
        assert 'text' in documents
        assert len(documents['text']) == 2  # requirements.md and notes.txt
        assert len(documents['pdf']) == 0
        assert len(documents['docx']) == 0
        assert len(documents['images']) == 0
        
        # Check specific files
        text_files = [f.name for f in documents['text']]
        assert 'requirements.md' in text_files
        assert 'notes.txt' in text_files
    
    @patch('src.agents.document_ingestion.genai')
    def test_text_file_processing(self, mock_genai, mock_config, temp_docs_dir):
        """Test text file processing."""
        agent = DocumentIngestionAgent(mock_config)
        
        # Test processing a markdown file
        md_file = Path(temp_docs_dir) / "requirements.md"
        content = agent._process_text_file(md_file)
        
        assert content is not None
        assert "Project Requirements" in content
        assert "User authentication" in content
        assert "Python FastAPI" in content
    
    @patch('src.agents.document_ingestion.genai')
    def test_table_to_markdown(self, mock_genai, mock_config):
        """Test table to markdown conversion."""
        agent = DocumentIngestionAgent(mock_config)
        
        table_data = [
            ["Name", "Type", "Description"],
            ["user_id", "INTEGER", "Primary key"],
            ["username", "VARCHAR", "Unique username"],
            ["email", "VARCHAR", "User email"]
        ]
        
        markdown = agent._table_to_markdown(table_data)
        
        assert "| Name | Type | Description |" in markdown
        assert "| --- | --- | --- |" in markdown
        assert "| user_id | INTEGER | Primary key |" in markdown
        assert "| username | VARCHAR | Unique username |" in markdown
    
    @patch('src.agents.document_ingestion.genai')
    def test_section_formatting(self, mock_genai, mock_config):
        """Test content section formatting."""
        agent = DocumentIngestionAgent(mock_config)
        
        title = "test_document.md"
        content = "This is test content"
        
        formatted = agent._format_section(title, content)
        
        assert "## Source: test_document.md" in formatted
        assert "This is test content" in formatted
        assert "---" in formatted
    
    @patch('src.agents.document_ingestion.genai')
    @patch('src.agents.document_ingestion.Path.write_text')
    def test_project_brief_generation(self, mock_write_text, mock_genai, mock_config):
        """Test ProjectBrief.md generation."""
        agent = DocumentIngestionAgent(mock_config)
        
        all_content = [
            "## Source: doc1.md\nContent 1\n---",
            "## Source: doc2.txt\nContent 2\n---"
        ]
        
        # Mock the write_text method to capture the content
        mock_write_text.return_value = None
        
        result_path = agent._generate_project_brief(all_content)
        
        # Check that write_text was called
        mock_write_text.assert_called_once()
        
        # Get the content that was written
        written_content = mock_write_text.call_args[0][0]
        
        assert "# Project Brief" in written_content
        assert "Generated on:" in written_content
        assert "Content 1" in written_content
        assert "Content 2" in written_content
        assert "End of Project Brief" in written_content
        
        assert result_path == "ProjectBrief.md"
    
    @patch('src.agents.document_ingestion.genai')
    def test_empty_docs_directory(self, mock_genai, mock_config):
        """Test behavior with empty docs directory."""
        agent = DocumentIngestionAgent(mock_config)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            empty_docs = Path(temp_dir) / "empty_docs"
            empty_docs.mkdir()
            
            documents = agent._collect_documents(empty_docs)
            
            assert documents['text'] == []
            assert documents['pdf'] == []
            assert documents['docx'] == []
            assert documents['images'] == []
    
    @patch('src.agents.document_ingestion.genai')
    def test_nonexistent_docs_directory(self, mock_genai, mock_config):
        """Test behavior with non-existent docs directory."""
        agent = DocumentIngestionAgent(mock_config)
        
        with pytest.raises(FileNotFoundError):
            agent.process_documents("non_existent_directory")
    
    @patch('src.agents.document_ingestion.genai')
    @patch('src.agents.document_ingestion.console')
    def test_process_documents_integration(self, mock_console, mock_genai, mock_config, temp_docs_dir):
        """Test the full document processing integration."""
        # Setup mock for Gemini
        mock_genai.configure.return_value = None
        mock_genai.GenerativeModel.return_value = Mock()
        
        agent = DocumentIngestionAgent(mock_config)
        
        with patch.object(agent, '_generate_project_brief') as mock_generate:
            mock_generate.return_value = "ProjectBrief.md"
            
            result = agent.process_documents(temp_docs_dir)
            
            assert result == "ProjectBrief.md"
            mock_generate.assert_called_once()
            
            # Check that content was processed
            call_args = mock_generate.call_args[0][0]
            assert len(call_args) >= 2  # Should have content from both files


class TestDocumentFormats:
    """Test handling of different document formats."""
    
    @pytest.fixture
    def mock_config(self):
        """Create a mock config for testing."""
        config = Mock(spec=Config)
        config.gemini_api_key = "test-key"
        config.gemini_model = "test-model"
        return config
    
    @patch('src.agents.document_ingestion.genai')
    def test_file_extension_recognition(self, mock_genai, mock_config):
        """Test that different file extensions are recognized correctly."""
        agent = DocumentIngestionAgent(mock_config)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            docs_path = Path(temp_dir)
            
            # Create files with different extensions
            (docs_path / "doc.md").touch()
            (docs_path / "doc.txt").touch()
            (docs_path / "doc.rst").touch()
            (docs_path / "doc.pdf").touch()
            (docs_path / "doc.docx").touch()
            (docs_path / "doc.png").touch()
            (docs_path / "doc.jpg").touch()
            (docs_path / "doc.unknown").touch()  # Unknown extension
            
            documents = agent._collect_documents(docs_path)
            
            assert len(documents['text']) == 3  # md, txt, rst
            assert len(documents['pdf']) == 1
            assert len(documents['docx']) == 1
            assert len(documents['images']) == 2  # png, jpg
            
            # Unknown extension should not be included
            all_files = (documents['text'] + documents['pdf'] + 
                        documents['docx'] + documents['images'])
            file_names = [f.name for f in all_files]
            assert 'doc.unknown' not in file_names
    
    @patch('src.agents.document_ingestion.genai')
    @patch('src.agents.document_ingestion.pdfplumber')
    def test_pdf_processing_error_handling(self, mock_pdfplumber, mock_genai, mock_config):
        """Test PDF processing error handling."""
        agent = DocumentIngestionAgent(mock_config)
        
        # Mock PDF processing to raise an exception
        mock_pdfplumber.open.side_effect = Exception("PDF processing error")
        
        with tempfile.NamedTemporaryFile(suffix='.pdf') as temp_file:
            result = agent._process_pdf_file(Path(temp_file.name))
            
            # Should return None on error
            assert result is None
    
    @patch('src.agents.document_ingestion.genai')
    @patch('src.agents.document_ingestion.Document')
    def test_docx_processing_error_handling(self, mock_document, mock_genai, mock_config):
        """Test DOCX processing error handling."""
        agent = DocumentIngestionAgent(mock_config)
        
        # Mock DOCX processing to raise an exception
        mock_document.side_effect = Exception("DOCX processing error")
        
        with tempfile.NamedTemporaryFile(suffix='.docx') as temp_file:
            result = agent._process_docx_file(Path(temp_file.name))
            
            # Should return None on error
            assert result is None