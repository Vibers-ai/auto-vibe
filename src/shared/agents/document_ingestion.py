"""Document Ingestion & Synthesis Agent

This module handles the parsing and synthesis of various document formats
into a unified ProjectBrief.md file.
"""


import sys
from pathlib import Path
# src 디렉토리를 Python path에 추가
src_dir = Path(__file__).parent.parent if 'src' in str(Path(__file__).parent) else Path(__file__).parent
while src_dir.name != 'src' and src_dir.parent != src_dir:
    src_dir = src_dir.parent
if src_dir.name == 'src':
    sys.path.insert(0, str(src_dir))

import os
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime

import pdfplumber
from docx import Document
from PIL import Image
import google.generativeai as genai
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

from shared.utils.config import Config
from shared.utils.file_utils import ensure_directory, read_text_file

logger = logging.getLogger(__name__)
console = Console()


class DocumentIngestionAgent:
    """Agent responsible for parsing and synthesizing project documentation."""
    
    def __init__(self, config: Config):
        self.config = config
        self.gemini_model = None
        self._setup_gemini()
        
    def _setup_gemini(self):
        """Initialize Gemini model for image analysis."""
        genai.configure(api_key=self.config.gemini_api_key)
        self.gemini_model = genai.GenerativeModel(self.config.gemini_model)
    
    def process_documents(self, docs_path: str = "docs") -> str:
        """Process all documents in the docs folder and create ProjectBrief.md
        
        Args:
            docs_path: Path to the documents folder
            
        Returns:
            Path to the generated ProjectBrief.md file
        """
        docs_dir = Path(docs_path)
        if not docs_dir.exists():
            raise FileNotFoundError(f"Documents directory not found: {docs_path}")
        
        console.print(f"[blue]Processing documents in {docs_path}...[/blue]")
        
        # Collect all documents
        documents = self._collect_documents(docs_dir)
        
        # Process each document type
        all_content = []
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            
            # Process text files
            task = progress.add_task("Processing text files...", total=len(documents['text']))
            for file_path in documents['text']:
                content = self._process_text_file(file_path)
                if content:
                    all_content.append(self._format_section(file_path.name, content))
                progress.advance(task)
            
            # Process PDFs
            task = progress.add_task("Processing PDF files...", total=len(documents['pdf']))
            for file_path in documents['pdf']:
                content = self._process_pdf_file(file_path)
                if content:
                    all_content.append(self._format_section(file_path.name, content))
                progress.advance(task)
            
            # Process Word documents
            task = progress.add_task("Processing Word documents...", total=len(documents['docx']))
            for file_path in documents['docx']:
                content = self._process_docx_file(file_path)
                if content:
                    all_content.append(self._format_section(file_path.name, content))
                progress.advance(task)
            
            # Process images
            task = progress.add_task("Processing images...", total=len(documents['images']))
            for file_path in documents['images']:
                content = self._process_image_file(file_path)
                if content:
                    all_content.append(self._format_section(f"{file_path.name} (Image Analysis)", content))
                progress.advance(task)
        
        # Generate ProjectBrief.md
        project_brief_path = self._generate_project_brief(all_content)
        
        console.print(f"[green]✓ ProjectBrief.md generated at: {project_brief_path}[/green]")
        return project_brief_path
    
    def _collect_documents(self, docs_dir: Path) -> Dict[str, List[Path]]:
        """Collect and categorize all documents in the directory."""
        documents = {
            'text': [],
            'pdf': [],
            'docx': [],
            'images': []
        }
        
        for file_path in docs_dir.rglob('*'):
            if file_path.is_file():
                suffix = file_path.suffix.lower()
                
                if suffix in ['.md', '.txt', '.rst']:
                    documents['text'].append(file_path)
                elif suffix == '.pdf':
                    documents['pdf'].append(file_path)
                elif suffix in ['.docx', '.doc']:
                    documents['docx'].append(file_path)
                elif suffix in ['.png', '.jpg', '.jpeg', '.gif', '.bmp']:
                    documents['images'].append(file_path)
        
        return documents
    
    def _process_text_file(self, file_path: Path) -> Optional[str]:
        """Process text-based files (Markdown, plain text, etc.)."""
        try:
            return read_text_file(str(file_path))
        except Exception as e:
            logger.error(f"Error processing text file {file_path}: {e}")
            return None
    
    def _process_pdf_file(self, file_path: Path) -> Optional[str]:
        """Extract text and tables from PDF files."""
        try:
            content = []
            
            with pdfplumber.open(file_path) as pdf:
                for page_num, page in enumerate(pdf.pages, 1):
                    # Extract text
                    text = page.extract_text()
                    if text:
                        content.append(f"Page {page_num}:\n{text}")
                    
                    # Extract tables
                    tables = page.extract_tables()
                    for table_idx, table in enumerate(tables):
                        if table:
                            table_md = self._table_to_markdown(table)
                            content.append(f"\nTable {table_idx + 1} on Page {page_num}:\n{table_md}")
            
            return "\n\n".join(content)
        
        except Exception as e:
            logger.error(f"Error processing PDF file {file_path}: {e}")
            return None
    
    def _process_docx_file(self, file_path: Path) -> Optional[str]:
        """Extract text and tables from Word documents."""
        try:
            doc = Document(file_path)
            content = []
            
            # Extract paragraphs
            for para in doc.paragraphs:
                if para.text.strip():
                    content.append(para.text)
            
            # Extract tables
            for table_idx, table in enumerate(doc.tables):
                table_data = []
                for row in table.rows:
                    row_data = [cell.text.strip() for cell in row.cells]
                    table_data.append(row_data)
                
                if table_data:
                    table_md = self._table_to_markdown(table_data)
                    content.append(f"\nTable {table_idx + 1}:\n{table_md}")
            
            return "\n\n".join(content)
        
        except Exception as e:
            logger.error(f"Error processing Word document {file_path}: {e}")
            return None
    
    def _process_image_file(self, file_path: Path) -> Optional[str]:
        """Analyze images using Gemini's multimodal capabilities."""
        try:
            image = Image.open(file_path)
            
            prompt = """Analyze this image and provide a detailed description. 
            If this is a UI mockup, wireframe, or design document, describe:
            - The layout and structure
            - UI components present (buttons, forms, navigation, etc.)
            - Any text or labels visible
            - The apparent functionality or purpose
            - Color scheme and visual style if relevant
            
            If this is a diagram or flowchart, describe:
            - The type of diagram
            - Components and their relationships
            - Any process flow or data flow shown
            - Key information conveyed
            
            Be as specific and detailed as possible."""
            
            response = self.gemini_model.generate_content([prompt, image])
            return response.text
        
        except Exception as e:
            logger.error(f"Error processing image file {file_path}: {e}")
            return None
    
    def _table_to_markdown(self, table_data: List[List[str]]) -> str:
        """Convert table data to Markdown format."""
        if not table_data or not table_data[0]:
            return ""
        
        # Create header
        header = "| " + " | ".join(str(cell) for cell in table_data[0]) + " |"
        separator = "| " + " | ".join("---" for _ in table_data[0]) + " |"
        
        # Create rows
        rows = []
        for row in table_data[1:]:
            row_str = "| " + " | ".join(str(cell) for cell in row) + " |"
            rows.append(row_str)
        
        return "\n".join([header, separator] + rows)
    
    def _format_section(self, title: str, content: str) -> str:
        """Format a content section for the ProjectBrief."""
        return f"""
## Source: {title}

{content}

---
"""
    
    def _generate_project_brief(self, all_content: List[str]) -> str:
        """Generate the final ProjectBrief.md file."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        brief_content = f"""# Project Brief
Generated on: {timestamp}

This document is a synthesized compilation of all project documentation found in the docs folder.
It serves as the primary input for the AI planning and code generation process.

---

{"".join(all_content)}

## End of Project Brief

This concludes the synthesized project documentation. The Master Planner Agent will use this 
information to design the complete software architecture and create detailed implementation tasks.
"""
        
        # Save to file
        output_path = Path("ProjectBrief.md")
        output_path.write_text(brief_content, encoding='utf-8')
        
        return str(output_path)