"""
Table Agent - Production-Grade Enterprise Document Table Extraction
Strategic AI Advisory Implementation - Production Hardened

Purpose: Production-ready autonomous table detection and extraction agent for Claude Code.
Handles complex table structures with parallel processing, layout-aware extraction,
comprehensive quality validation, and enterprise observability.
"""

import io
import logging
import asyncio
import json
import re
import time
import uuid
from typing import Dict, Any, Optional, List, Union, Tuple, Set
from pathlib import Path
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor, TimeoutError
import traceback

# Core table extraction dependencies
import pandas as pd
import numpy as np
from PIL import Image
import cv2

# PDF processing with fallbacks
try:
    import pdfplumber
    import camelot
    import tabula
    HAS_PDF_TOOLS = True
except ImportError as e:
    logging.warning(f"PDF tools not available: {e}")
    HAS_PDF_TOOLS = False

# Document processing
try:
    from docx import Document
    from docx.table import Table as DocxTable
    HAS_DOCX = True
except ImportError:
    HAS_DOCX = False

# OCR integration for scanned PDFs
try:
    import pytesseract
    from pdf2image import convert_from_bytes
    HAS_OCR = True
except ImportError:
    HAS_OCR = False

# HTML processing
try:
    from bs4 import BeautifulSoup
    HAS_HTML_PARSER = True
except ImportError:
    HAS_HTML_PARSER = False

# Text similarity for deduplication
try:
    from difflib import SequenceMatcher
    import jellyfish  # for Levenshtein distance
    HAS_TEXT_SIMILARITY = True
except ImportError:
    HAS_TEXT_SIMILARITY = False

@dataclass
class TableExtractionResult:
    """Production-grade result container with comprehensive metadata."""
    tables: List[Dict[str, Any]]
    metadata: Dict[str, Any]
    extraction_methods_used: List[str]
    processing_time_ms: int
    document_id: str
    quality_metrics: Dict[str, Any]
    business_insights: Dict[str, Any]
    error_details: Optional[Dict[str, Any]] = None

@dataclass
class TableStructure:
    """Production table structure with provenance and quality metrics."""
    data: List[List[str]]
    headers: List[str]
    page_number: Optional[int]
    bbox: Optional[Tuple[float, float, float, float]]
    table_id: str
    confidence: float
    extraction_method: str
    structure_type: str
    provenance: Dict[str, Any]  # method, page, bbox, processing_time
    quality_score: float
    column_types: Dict[str, str]
    data_quality_metrics: Dict[str, Any]

class ProductionTableAgent:
    """
    Production-grade table extraction agent with enterprise reliability.
    
    Strategic Value:
    - Parallel processing with timeout controls and partial failure handling
    - Layout-aware PDF processing with OCR fallback for scanned documents
    - Comprehensive data quality validation and normalization
    - Enterprise observability with structured logging and metrics
    """
    
    def __init__(self, 
                 extraction_strategy: str = "comprehensive",
                 max_runtime_seconds: int = 300,
                 max_pages: Optional[int] = None,
                 sample_rate: float = 1.0,
                 parallel_processing: bool = True,
                 enable_ocr_fallback: bool = True,
                 quality_threshold: float = 0.7):
        """
        Initialize production table agent with enterprise controls.
        
        Args:
            extraction_strategy: "speed", "quality", "comprehensive"
            max_runtime_seconds: Maximum processing time per document
            max_pages: Limit pages processed (None = all pages)
            sample_rate: Fraction of pages to process (0.1 = every 10th page)
            parallel_processing: Enable parallel method execution
            enable_ocr_fallback: Use OCR for scanned PDFs
            quality_threshold: Minimum quality score for table acceptance
        """
        self.extraction_strategy = extraction_strategy
        self.max_runtime_seconds = max_runtime_seconds
        self.max_pages = max_pages
        self.sample_rate = sample_rate
        self.parallel_processing = parallel_processing
        self.enable_ocr_fallback = enable_ocr_fallback
        self.quality_threshold = quality_threshold
        
        # Production logging
        self.logger = self._setup_production_logging()
        
        # Method timeout configuration
        self.method_timeouts = {
            "pdfplumber": 60,
            "camelot_lattice": 90,
            "camelot_stream": 60,
            "tabula": 45,
            "docx_native": 30,
            "html_extraction": 30,
            "pandas_structured": 20,
            "ocr_fallback": 120
        }
        
        # Verify system dependencies
        self._verify_system_dependencies()
        
        # Performance and quality tracking
        self.processing_stats = {
            "documents_processed": 0,
            "method_performance": {},
            "quality_distributions": {},
            "error_patterns": {},
            "processing_times": [],
            "parallel_efficiency": []
        }

    def _setup_production_logging(self) -> logging.Logger:
        """Setup structured logging for production observability."""
        
        logger = logging.getLogger(f"table_agent_{id(self)}")
        logger.setLevel(logging.INFO)
        
        # Structured formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - '
            'doc_id:%(document_id)s - method:%(method)s - %(message)s',
            defaults={'document_id': 'unknown', 'method': 'unknown'}
        )
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger

    def _verify_system_dependencies(self):
        """Verify system dependencies with health checks."""
        
        dependency_status = {
            "pdf_tools": HAS_PDF_TOOLS,
            "docx_support": HAS_DOCX,
            "ocr_support": HAS_OCR,
            "html_parser": HAS_HTML_PARSER,
            "text_similarity": HAS_TEXT_SIMILARITY
        }
        
        # Log dependency status
        self.logger.info(f"System dependencies: {dependency_status}")
        
        # Test critical dependencies
        if HAS_PDF_TOOLS:
            try:
                # Test Camelot (requires Java/Ghostscript)
                import subprocess
                subprocess.run(['java', '-version'], capture_output=True, check=True, timeout=5)
                subprocess.run(['gs', '--version'], capture_output=True, check=True, timeout=5)
                self.logger.info("Camelot dependencies (Java/Ghostscript) verified")
            except (subprocess.CalledProcessError, FileNotFoundError, TimeoutError):
                self.logger.warning("Camelot dependencies not available - will skip Camelot methods")
        
        if not any([HAS_PDF_TOOLS, HAS_DOCX, HAS_HTML_PARSER]):
            raise RuntimeError("No extraction engines available - install required dependencies")

    async def extract_tables(self, 
                           file_bytes: bytes, 
                           filename: Optional[str] = None,
                           mime_type: Optional[str] = None,
                           table_filters: Optional[Dict[str, Any]] = None,
                           document_id: Optional[str] = None) -> TableExtractionResult:
        """
        Production-grade table extraction with comprehensive error handling.
        
        Args:
            file_bytes: Document bytes
            filename: Original filename for type detection
            mime_type: MIME type if known
            table_filters: Filters like {"required_headers": ["Name", "Amount"]}
            document_id: Unique document identifier for tracking
        """
        
        if document_id is None:
            document_id = str(uuid.uuid4())
        
        start_time = time.time()
        processing_context = {
            'document_id': document_id,
            'method': 'extract_tables',
            'filename': filename
        }
        
        self.logger.info(f"Starting table extraction", extra=processing_context)
        
        try:
            # Document analysis with timeout
            doc_analysis = await asyncio.wait_for(
                self._analyze_document_comprehensive(file_bytes, filename, mime_type),
                timeout=30
            )
            
            # Create processing plan with page sampling
            processing_plan = self._create_production_processing_plan(doc_analysis, table_filters)
            
            # Execute extraction with parallel processing and timeouts
            if self.parallel_processing:
                extraction_results = await self._execute_parallel_extraction(
                    file_bytes, processing_plan, document_id
                )
            else:
                extraction_results = await self._execute_sequential_extraction(
                    file_bytes, processing_plan, document_id
                )
            
            # Process and validate results
            validated_tables = await self._process_and_validate_tables(
                extraction_results, table_filters, document_id
            )
            
            # Generate quality metrics and business insights
            quality_metrics = await self._calculate_quality_metrics(validated_tables)
            business_insights = await self._generate_production_insights(validated_tables, doc_analysis)
            
            processing_time = int((time.time() - start_time) * 1000)
            
            # Update stats
            self._update_processing_stats(
                methods_used=processing_plan["methods"],
                tables_found=len(validated_tables),
                processing_time=processing_time,
                quality_metrics=quality_metrics,
                success=True
            )
            
            self.logger.info(
                f"Table extraction completed - {len(validated_tables)} tables found",
                extra={**processing_context, 'tables_found': len(validated_tables)}
            )
            
            return TableExtractionResult(
                tables=[asdict(table) for table in validated_tables],
                metadata={
                    "document_analysis": doc_analysis,
                    "processing_plan": processing_plan,
                    "extraction_results_summary": self._summarize_extraction_results(extraction_results),
                    "page_sampling": {
                        "sample_rate": self.sample_rate,
                        "pages_processed": len(processing_plan.get("page_range", [])),
                        "total_pages": doc_analysis.get("page_count", 1)
                    }
                },
                extraction_methods_used=list(processing_plan["methods"]),
                processing_time_ms=processing_time,
                document_id=document_id,
                quality_metrics=quality_metrics,
                business_insights=business_insights
            )
            
        except asyncio.TimeoutError:
            processing_time = int((time.time() - start_time) * 1000)
            error_msg = f"Processing timeout after {self.max_runtime_seconds}s"
            self.logger.error(error_msg, extra=processing_context)
            
            return self._create_error_result(
                document_id, error_msg, processing_time, "timeout"
            )
            
        except Exception as e:
            processing_time = int((time.time() - start_time) * 1000)
            error_msg = f"Table extraction failed: {str(e)}"
            self.logger.error(error_msg, extra={**processing_context, 'error': str(e), 'traceback': traceback.format_exc()})
            
            self._update_processing_stats(
                methods_used=[],
                tables_found=0,
                processing_time=processing_time,
                quality_metrics={},
                success=False,
                error_type=type(e).__name__
            )
            
            return self._create_error_result(
                document_id, error_msg, processing_time, type(e).__name__
            )

    async def _analyze_document_comprehensive(self, 
                                            file_bytes: bytes, 
                                            filename: Optional[str] = None,
                                            mime_type: Optional[str] = None) -> Dict[str, Any]:
        """Comprehensive document analysis for production planning."""
        
        analysis = {
            "document_type": self._detect_document_type_robust(file_bytes, filename, mime_type),
            "file_size_bytes": len(file_bytes),
            "page_count": 1,
            "is_scanned": False,
            "has_tables": False,
            "complexity_score": 0.5,
            "processing_recommendations": []
        }
        
        try:
            if analysis["document_type"] == "pdf" and HAS_PDF_TOOLS:
                pdf_analysis = await self._analyze_pdf_layout_aware(file_bytes)
                analysis.update(pdf_analysis)
            elif analysis["document_type"] == "docx" and HAS_DOCX:
                docx_analysis = await self._analyze_docx_structure(file_bytes)
                analysis.update(docx_analysis)
            elif analysis["document_type"] in ["html", "xml"] and HAS_HTML_PARSER:
                html_analysis = await self._analyze_html_tables(file_bytes)
                analysis.update(html_analysis)
                
        except Exception as e:
            self.logger.warning(f"Document analysis partial failure: {str(e)}")
            analysis["analysis_warnings"] = [str(e)]
        
        return analysis

    def _detect_document_type_robust(self, 
                                   file_bytes: bytes, 
                                   filename: Optional[str] = None,
                                   mime_type: Optional[str] = None) -> str:
        """Robust document type detection with multiple fallbacks."""
        
        # MIME type mapping
        if mime_type:
            mime_map = {
                "application/pdf": "pdf",
                "application/vnd.openxmlformats-officedocument.wordprocessingml.document": "docx",
                "text/html": "html",
                "application/xhtml+xml": "html",
                "text/csv": "csv",
                "application/vnd.ms-excel": "excel",
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": "xlsx"
            }
            if mime_type in mime_map:
                return mime_map[mime_type]
        
        # File signature detection (magic numbers)
        signatures = [
            (b'%PDF', 'pdf'),
            (b'PK\x03\x04', 'zip_based'),  # Could be DOCX, XLSX
            (b'<html', 'html'),
            (b'<!DOCTYPE html', 'html'),
            (b'\xd0\xcf\x11\xe0', 'ole'),  # Old Office formats
        ]
        
        file_start = file_bytes[:1024].lower()
        for signature, doc_type in signatures:
            if file_start.startswith(signature.lower()):
                if doc_type == 'zip_based' and filename:
                    if filename.endswith('.docx'):
                        return 'docx'
                    elif filename.endswith(('.xlsx', '.xls')):
                        return 'xlsx'
                return doc_type
        
        # Filename extension fallback
        if filename:
            ext = Path(filename).suffix.lower()
            ext_map = {
                '.pdf': 'pdf',
                '.docx': 'docx', 
                '.xlsx': 'xlsx',
                '.xls': 'excel',
                '.csv': 'csv',
                '.html': 'html',
                '.htm': 'html',
                '.xml': 'xml'
            }
            if ext in ext_map:
                return ext_map[ext]
        
        # Content-based heuristics
        try:
            text_sample = file_bytes[:4096].decode('utf-8', errors='ignore').lower()
            if '<table' in text_sample and ('<html' in text_sample or '<body' in text_sample):
                return 'html'
        except:
            pass
        
        return 'unknown'

    async def _analyze_pdf_layout_aware(self, file_bytes: bytes) -> Dict[str, Any]:
        """Layout-aware PDF analysis for optimal extraction strategy."""
        
        analysis = {
            "page_count": 0,
            "has_embedded_text": False,
            "is_scanned": False,
            "has_tables": False,
            "table_detection_confidence": 0.0,
            "recommended_methods": [],
            "complexity_score": 0.5
        }
        
        if not HAS_PDF_TOOLS:
            analysis["error"] = "PDF tools not available"
            return analysis
        
        try:
            # Quick analysis with pdfplumber
            with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
                analysis["page_count"] = len(pdf.pages)
                
                # Sample pages for text/table detection
                sample_pages = min(3, len(pdf.pages))
                text_found = 0
                tables_found = 0
                
                for i in range(sample_pages):
                    page = pdf.pages[i]
                    
                    # Check for embedded text
                    text = page.extract_text()
                    if text and len(text.strip()) > 50:
                        text_found += 1
                    
                    # Quick table detection
                    try:
                        page_tables = page.extract_tables()
                        if page_tables:
                            tables_found += len(page_tables)
                    except:
                        pass
                
                analysis["has_embedded_text"] = text_found > 0
                analysis["is_scanned"] = text_found < sample_pages * 0.5
                analysis["has_tables"] = tables_found > 0
                analysis["table_detection_confidence"] = min(1.0, tables_found / max(sample_pages, 1))
                
                # Complexity scoring
                complexity = 0.0
                complexity += min(1.0, analysis["page_count"] / 20)  # Page count factor
                complexity += 0.5 if analysis["is_scanned"] else 0.1  # Scanned penalty
                complexity += 0.3 if tables_found > 5 else 0.0  # High table count
                analysis["complexity_score"] = min(1.0, complexity)
                
                # Method recommendations
                if analysis["is_scanned"] and self.enable_ocr_fallback:
                    analysis["recommended_methods"] = ["ocr_fallback", "camelot_lattice"]
                elif analysis["has_tables"]:
                    analysis["recommended_methods"] = ["camelot_lattice", "camelot_stream", "pdfplumber", "tabula"]
                else:
                    analysis["recommended_methods"] = ["pdfplumber", "camelot_stream"]
                
        except Exception as e:
            self.logger.warning(f"PDF layout analysis failed: {str(e)}")
            analysis["analysis_error"] = str(e)
            analysis["recommended_methods"] = ["pdfplumber"]  # Safe fallback
        
        return analysis

    async def _analyze_docx_structure(self, file_bytes: bytes) -> Dict[str, Any]:
        """Analyze DOCX document structure."""
        
        analysis = {
            "has_tables": False,
            "table_count": 0,
            "complexity_score": 0.1
        }
        
        if not HAS_DOCX:
            analysis["error"] = "DOCX support not available"
            return analysis
        
        try:
            doc = Document(io.BytesIO(file_bytes))
            table_count = len(doc.tables)
            
            analysis.update({
                "has_tables": table_count > 0,
                "table_count": table_count,
                "complexity_score": min(1.0, table_count / 10),
                "recommended_methods": ["docx_native"]
            })
            
        except Exception as e:
            analysis["analysis_error"] = str(e)
        
        return analysis

    async def _analyze_html_tables(self, file_bytes: bytes) -> Dict[str, Any]:
        """Analyze HTML document for table content."""
        
        analysis = {
            "has_tables": False,
            "table_count": 0,
            "complexity_score": 0.1
        }
        
        if not HAS_HTML_PARSER:
            analysis["error"] = "HTML parser not available"
            return analysis
        
        try:
            html_content = file_bytes.decode('utf-8', errors='ignore')
            soup = BeautifulSoup(html_content, 'html.parser')
            tables = soup.find_all('table')
            
            analysis.update({
                "has_tables": len(tables) > 0,
                "table_count": len(tables),
                "complexity_score": min(1.0, len(tables) / 5),
                "recommended_methods": ["html_extraction", "pandas_structured"]
            })
            
        except Exception as e:
            analysis["analysis_error"] = str(e)
        
        return analysis

    def _create_production_processing_plan(self, 
                                         doc_analysis: Dict[str, Any],
                                         table_filters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Create production processing plan with controls and sampling."""
        
        plan = {
            "methods": set(),
            "page_range": None,
            "timeouts": {},
            "parallel_execution": self.parallel_processing,
            "quality_controls": {
                "min_confidence": self.quality_threshold,
                "table_filters": table_filters or {}
            }
        }
        
        # Method selection based on document type and strategy
        doc_type = doc_analysis.get("document_type", "unknown")
        recommended = doc_analysis.get("recommended_methods", [])
        
        if doc_type == "pdf":
            if self.extraction_strategy == "speed":
                plan["methods"] = {"pdfplumber"}
            elif self.extraction_strategy == "comprehensive":
                plan["methods"] = {"pdfplumber", "camelot_lattice", "camelot_stream", "tabula"}
                if doc_analysis.get("is_scanned") and self.enable_ocr_fallback:
                    plan["methods"].add("ocr_fallback")
            else:  # balanced
                plan["methods"] = {"pdfplumber", "camelot_lattice"}
                
        elif doc_type == "docx":
            plan["methods"] = {"docx_native"}
            
        elif doc_type in ["html", "xml"]:
            plan["methods"] = {"html_extraction"}
            
        elif doc_type in ["csv", "xlsx", "excel"]:
            plan["methods"] = {"pandas_structured"}
            
        else:
            # Try multiple approaches for unknown types
            plan["methods"] = {"pdfplumber", "html_extraction", "pandas_structured"}
        
        # Apply page sampling
        total_pages = doc_analysis.get("page_count", 1)
        if self.max_pages and total_pages > self.max_pages:
            # Take evenly distributed sample
            step = total_pages // self.max_pages
            plan["page_range"] = list(range(0, total_pages, max(1, step)))[:self.max_pages]
        elif self.sample_rate < 1.0:
            # Random sampling
            import random
            sample_size = max(1, int(total_pages * self.sample_rate))
            plan["page_range"] = sorted(random.sample(range(total_pages), sample_size))
        else:
            plan["page_range"] = list(range(total_pages))
        
        # Set method timeouts
        for method in plan["methods"]:
            plan["timeouts"][method] = self.method_timeouts.get(method, 60)
        
        return plan

    async def _execute_parallel_extraction(self, 
                                         file_bytes: bytes,
                                         processing_plan: Dict[str, Any],
                                         document_id: str) -> Dict[str, List[TableStructure]]:
        """Execute extraction methods in parallel with timeout handling."""
        
        async def run_method_with_timeout(method: str) -> Tuple[str, List[TableStructure]]:
            timeout = processing_plan["timeouts"].get(method, 60)
            
            try:
                with ThreadPoolExecutor(max_workers=1) as executor:
                    future = executor.submit(self._execute_method_sync, method, file_bytes, processing_plan, document_id)
                    result = await asyncio.wait_for(
                        asyncio.wrap_future(future),
                        timeout=timeout
                    )
                    return method, result
                    
            except asyncio.TimeoutError:
                self.logger.warning(f"Method {method} timed out after {timeout}s", 
                                  extra={'document_id': document_id, 'method': method})
                return method, []
            except Exception as e:
                self.logger.error(f"Method {method} failed: {str(e)}", 
                                extra={'document_id': document_id, 'method': method, 'error': str(e)})
                return method, []
        
        # Execute all methods in parallel
        tasks = [run_method_with_timeout(method) for method in processing_plan["methods"]]
        
        try:
            results = await asyncio.wait_for(
                asyncio.gather(*tasks, return_exceptions=True),
                timeout=self.max_runtime_seconds
            )
            
            # Process results and exceptions
            extraction_results = {}
            for result in results:
                if isinstance(result, Exception):
                    self.logger.error(f"Parallel execution exception: {result}", 
                                    extra={'document_id': document_id})
                    continue
                
                method, tables = result
                extraction_results[method] = tables
            
            return extraction_results
            
        except asyncio.TimeoutError:
            self.logger.error(f"Overall extraction timeout after {self.max_runtime_seconds}s",
                            extra={'document_id': document_id})
            return {}

    async def _execute_sequential_extraction(self, 
                                           file_bytes: bytes,
                                           processing_plan: Dict[str, Any],
                                           document_id: str) -> Dict[str, List[TableStructure]]:
        """Execute extraction methods sequentially with individual timeouts."""
        
        extraction_results = {}
        
        for method in processing_plan["methods"]:
            timeout = processing_plan["timeouts"].get(method, 60)
            
            try:
                with ThreadPoolExecutor(max_workers=1) as executor:
                    future = executor.submit(self._execute_method_sync, method, file_bytes, processing_plan, document_id)
                    tables = await asyncio.wait_for(
                        asyncio.wrap_future(future),
                        timeout=timeout
                    )
                    extraction_results[method] = tables
                    
            except asyncio.TimeoutError:
                self.logger.warning(f"Method {method} timed out after {timeout}s", 
                                  extra={'document_id': document_id, 'method': method})
                extraction_results[method] = []
            except Exception as e:
                self.logger.error(f"Method {method} failed: {str(e)}", 
                                extra={'document_id': document_id, 'method': method, 'error': str(e)})
                extraction_results[method] = []
        
        return extraction_results

    def _execute_method_sync(self, 
                           method: str, 
                           file_bytes: bytes,
                           processing_plan: Dict[str, Any],
                           document_id: str) -> List[TableStructure]:
        """Synchronous method execution for thread pool."""
        
        start_time = time.time()
        page_range = processing_plan.get("page_range", [0])
        
        try:
            if method == "pdfplumber":
                tables = self._extract_pdf_pdfplumber_corrected(file_bytes, page_range)
            elif method == "camelot_lattice":
                tables = self._extract_camelot_corrected(file_bytes, "lattice", page_range)
            elif method == "camelot_stream":
                tables = self._extract_camelot_corrected(file_bytes, "stream", page_range)
            elif method == "tabula":
                tables = self._extract_tabula_corrected(file_bytes, page_range)
            elif method == "docx_native":
                tables = self._extract_docx_corrected(file_bytes)
            elif method == "html_extraction":
                tables = self._extract_html_corrected(file_bytes)
            elif method == "pandas_structured":
                tables = self._extract_pandas_corrected(file_bytes)
            elif method == "ocr_fallback":
                tables = self._extract_ocr_fallback(file_bytes, page_range)
            else:
                self.logger.warning(f"Unknown method: {method}")
                return []
            
            processing_time = time.time() - start_time
            
            # Add provenance to all tables
            for table in tables:
                table.provenance.update({
                    "processing_time_seconds": processing_time,
                    "document_id": document_id,
                    "pages_sampled": page_range
                })
            
            self.logger.info(f"Method {method} completed - {len(tables)} tables found",
                           extra={'document_id': document_id, 'method': method, 
                                 'tables_found': len(tables), 'processing_time': processing_time})
            
            return tables
            
        except Exception as e:
            self.logger.error(f"Method {method} execution failed: {str(e)}",
                            extra={'document_id': document_id, 'method': method, 'error': str(e)})
            return []

    def _extract_camelot_corrected(self, 
                                 file_bytes: bytes, 
                                 flavor: str,
                                 page_range: List[int]) -> List[TableStructure]:
        """Corrected Camelot extraction with proper confidence handling."""
        
        if not HAS_PDF_TOOLS:
            return []
        
        tables = []
        
        # Create temporary file for Camelot
        import tempfile
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp_file:
            tmp_file.write(file_bytes)
            tmp_file.flush()
            
            try:
                # Convert page range to Camelot format
                pages_str = ','.join(str(p + 1) for p in page_range) if page_range else 'all'
                
                # Extract with specified flavor
                camelot_tables = camelot.read_pdf(tmp_file.name, pages=pages_str, flavor=flavor)
                
                for idx, table in enumerate(camelot_tables):
                    if len(table.df) < 2:  # Skip empty or single-row tables
                        continue
                    
                    # CORRECTED: Extract confidence from parsing report
                    parsing_report = table.parsing_report
                    confidence = parsing_report['accuracy'] / 100.0 if 'accuracy' in parsing_report else 0.8
                    
                    # CORRECTED: Proper header extraction
                    df = table.df.copy()
                    headers, data_rows = self._extract_headers_corrected(df)
                    
                    # Clean and normalize data
                    cleaned_data = self._normalize_table_data(data_rows)
                    
                    # Infer column types
                    column_types = self._infer_column_types(headers, cleaned_data)
                    
                    table_structure = TableStructure(
                        data=cleaned_data,
                        headers=headers,
                        page_number=table.page,
                        bbox=(table._bbox[0], table._bbox[1], table._bbox[2], table._bbox[3]) 
                              if hasattr(table, '_bbox') else None,
                        table_id=f"camelot_{flavor}_p{table.page}_t{idx+1}",
                        confidence=confidence,
                        extraction_method=f"camelot_{flavor}",
                        structure_type=self._classify_table_structure_enhanced(cleaned_data, headers),
                        provenance={
                            "method": f"camelot_{flavor}",
                            "page": table.page,
                            "table_index": idx,
                            "flavor": flavor,
                            "parsing_report": parsing_report
                        },
                        quality_score=self._calculate_table_quality_score(cleaned_data, headers, confidence),
                        column_types=column_types,
                        data_quality_metrics=self._calculate_data_quality_metrics(cleaned_data, headers, column_types)
                    )
                    
                    tables.append(table_structure)
                    
            except Exception as e:
                self.logger.error(f"Camelot {flavor} extraction failed: {str(e)}")
            finally:
                # Clean up temporary file
                Path(tmp_file.name).unlink(missing_ok=True)
        
        return tables

    def _extract_tabula_corrected(self, file_bytes: bytes, page_range: List[int]) -> List[TableStructure]:
        """Corrected Tabula extraction with proper header handling."""
        
        if not HAS_PDF_TOOLS:
            return []
        
        tables = []
        
        import tempfile
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp_file:
            tmp_file.write(file_bytes)
            tmp_file.flush()
            
            try:
                # Convert page range to Tabula format
                pages_str = ','.join(str(p + 1) for p in page_range) if page_range else 'all'
                
                # Extract tables
                tabula_tables = tabula.read_pdf(
                    tmp_file.name, 
                    pages=pages_str, 
                    multiple_tables=True,
                    pandas_options={'header': None}  # Let us handle headers
                )
                
                for idx, df in enumerate(tabula_tables):
                    if len(df) < 2:
                        continue
                    
                    # CORRECTED: Proper header extraction
                    headers, data_rows = self._extract_headers_corrected(df)
                    
                    # Clean and normalize data
                    cleaned_data = self._normalize_table_data(data_rows)
                    column_types = self._infer_column_types(headers, cleaned_data)
                    
                    table_structure = TableStructure(
                        data=cleaned_data,
                        headers=headers,
                        page_number=None,  # Tabula doesn't reliably provide page numbers in multi-page
                        bbox=None,
                        table_id=f"tabula_t{idx+1}",
                        confidence=0.8,  # Default confidence for Tabula
                        extraction_method="tabula",
                        structure_type=self._classify_table_structure_enhanced(cleaned_data, headers),
                        provenance={
                            "method": "tabula",
                            "table_index": idx,
                            "pages_requested": pages_str
                        },
                        quality_score=self._calculate_table_quality_score(cleaned_data, headers, 0.8),
                        column_types=column_types,
                        data_quality_metrics=self._calculate_data_quality_metrics(cleaned_data, headers, column_types)
                    )
                    
                    tables.append(table_structure)
                    
            except Exception as e:
                self.logger.error(f"Tabula extraction failed: {str(e)}")
            finally:
                Path(tmp_file.name).unlink(missing_ok=True)
        
        return tables

    def _extract_headers_corrected(self, df: pd.DataFrame) -> Tuple[List[str], List[List[str]]]:
        """
        CORRECTED: Properly extract headers from DataFrame.
        Detects header row (first non-empty row) and separates from data.
        """
        
        # Find first non-empty row as header
        header_row_idx = 0
        for i, row in df.iterrows():
            row_values = [str(val).strip() if pd.notna(val) else "" for val in row.values]
            if any(val for val in row_values):  # Found non-empty row
                header_row_idx = i
                break
        
        # Extract headers
        header_row = df.iloc[header_row_idx]
        headers = [self._normalize_header(str(val)) if pd.notna(val) else f"Column_{i}" 
                  for i, val in enumerate(header_row.values)]
        
        # Handle duplicate headers
        headers = self._deduplicate_headers(headers)
        
        # Extract data rows (everything after header)
        data_df = df.iloc[header_row_idx + 1:]
        data_rows = []
        
        for _, row in data_df.iterrows():
            row_data = [str(val).strip() if pd.notna(val) else "" for val in row.values]
            # Only include non-empty rows
            if any(cell.strip() for cell in row_data):
                data_rows.append(row_data)
        
        return headers, data_rows

    def _normalize_header(self, header: str) -> str:
        """Normalize header names for consistency."""
        
        if not header or header.strip() == "":
            return ""
        
        # Clean up common issues
        normalized = header.strip()
        
        # Handle "Unnamed: n" from pandas
        if normalized.startswith("Unnamed:"):
            return ""
        
        # Title case for readability
        normalized = normalized.title()
        
        # Remove excessive whitespace
        normalized = re.sub(r'\s+', ' ', normalized)
        
        return normalized

    def _deduplicate_headers(self, headers: List[str]) -> List[str]:
        """Deduplicate headers while preserving order."""
        
        seen = set()
        deduplicated = []
        
        for i, header in enumerate(headers):
            if not header:  # Empty header
                header = f"Column_{i+1}"
            
            original_header = header
            counter = 1
            
            while header in seen:
                header = f"{original_header}_{counter}"
                counter += 1
            
            seen.add(header)
            deduplicated.append(header)
        
        return deduplicated

    def _normalize_table_data(self, data_rows: List[List[str]]) -> List[List[str]]:
        """Normalize table data with financial and numeric handling."""
        
        normalized_rows = []
        
        for row in data_rows:
            normalized_row = []
            
            for cell in row:
                if not cell:
                    normalized_row.append("")
                    continue
                
                # Clean up cell content
                cleaned = str(cell).strip()
                
                # Handle common numeric formats
                cleaned = self._normalize_numeric_value(cleaned)
                
                # Remove non-breaking spaces and other whitespace
                cleaned = re.sub(r'\s+', ' ', cleaned)
                
                normalized_row.append(cleaned)
            
            if any(cell.strip() for cell in normalized_row):  # Only include non-empty rows
                normalized_rows.append(normalized_row)
        
        return normalized_rows

    def _normalize_numeric_value(self, value: str) -> str:
        """Normalize numeric values with financial formatting."""
        
        if not value or not isinstance(value, str):
            return str(value) if value else ""
        
        # Remove common formatting
        cleaned = value.strip()
        
        # Remove non-breaking spaces (common in PDFs)
        cleaned = cleaned.replace('\u00a0', ' ').replace('\xa0', ' ')
        
        # Handle parentheses as negatives for financial data
        if cleaned.startswith('(') and cleaned.endswith(')'):
            cleaned = '-' + cleaned[1:-1].strip()
        
        # Remove currency symbols but preserve the number
        currency_pattern = r'^[\$€£¥₹]\s*'
        if re.match(currency_pattern, cleaned):
            cleaned = re.sub(currency_pattern, '', cleaned)
        
        # Handle thousands separators (locale-aware)
        # European format: 1.234.567,89 vs US format: 1,234,567.89
        if ',' in cleaned and '.' in cleaned:
            # Determine format by position of last comma vs last dot
            last_comma = cleaned.rfind(',')
            last_dot = cleaned.rfind('.')
            
            if last_comma > last_dot:
                # European format: 1.234,89
                cleaned = cleaned.replace('.', '').replace(',', '.')
            else:
                # US format: 1,234.89
                cleaned = cleaned.replace(',', '')
        elif ',' in cleaned:
            # Could be thousands separator or decimal
            comma_pos = cleaned.rfind(',')
            if len(cleaned) - comma_pos == 3:  # Likely thousands separator
                cleaned = cleaned.replace(',', '')
            else:
                cleaned = cleaned.replace(',', '.')  # Likely decimal separator
        
        return cleaned

    def _infer_column_types(self, headers: List[str], data_rows: List[List[str]]) -> Dict[str, str]:
        """Infer column data types for quality validation."""
        
        column_types = {}
        
        for col_idx, header in enumerate(headers):
            values = [row[col_idx] if col_idx < len(row) else "" 
                     for row in data_rows]
            
            # Remove empty values for type inference
            non_empty_values = [v for v in values if v.strip()]
            
            if not non_empty_values:
                column_types[header] = "empty"
                continue
            
            # Count different value patterns
            numeric_count = 0
            date_count = 0
            text_count = 0
            
            for value in non_empty_values[:20]:  # Sample first 20 values
                if self._is_numeric(value):
                    numeric_count += 1
                elif self._is_date_like(value):
                    date_count += 1
                else:
                    text_count += 1
            
            total = len(non_empty_values[:20])
            
            # Assign type based on majority
            if numeric_count / total > 0.7:
                column_types[header] = "numeric"
            elif date_count / total > 0.5:
                column_types[header] = "date"
            else:
                column_types[header] = "text"
        
        return column_types

    def _is_numeric(self, value: str) -> bool:
        """Check if value appears to be numeric."""
        
        if not value:
            return False
        
        # Clean value for numeric testing
        cleaned = re.sub(r'[^\d\.\-\+]', '', value)
        
        try:
            float(cleaned)
            return True
        except ValueError:
            return False

    def _is_date_like(self, value: str) -> bool:
        """Check if value appears to be a date."""
        
        if not value or len(value) < 6:
            return False
        
        # Simple date pattern matching
        date_patterns = [
            r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}',  # MM/DD/YYYY or MM-DD-YYYY
            r'\d{4}[/-]\d{1,2}[/-]\d{1,2}',    # YYYY-MM-DD
            r'\d{1,2}\s+\w{3,}\s+\d{4}',       # DD Month YYYY
            r'\w{3,}\s+\d{1,2},?\s+\d{4}'      # Month DD, YYYY
        ]
        
        return any(re.search(pattern, value) for pattern in date_patterns)

    async def _process_and_validate_tables(self, 
                                         extraction_results: Dict[str, List[TableStructure]],
                                         table_filters: Optional[Dict[str, Any]],
                                         document_id: str) -> List[TableStructure]:
        """
        Process extraction results with deduplication, validation, and filtering.
        """
        
        # Collect all tables
        all_tables = []
        for method, tables in extraction_results.items():
            all_tables.extend(tables)
        
        if not all_tables:
            return []
        
        # CORRECTED: Soften deduplication with Jaccard similarity
        deduplicated_tables = self._deduplicate_tables_improved(all_tables)
        
        # Apply quality filtering
        quality_filtered = [table for table in deduplicated_tables 
                          if table.quality_score >= self.quality_threshold]
        
        # Apply table filters if specified
        if table_filters:
            filtered_tables = self._apply_table_filters(quality_filtered, table_filters)
        else:
            filtered_tables = quality_filtered
        
        # CORRECTED: Improved table continuation detection and merging
        if len(filtered_tables) > 1:
            merged_tables = self._merge_continued_tables_improved(filtered_tables)
        else:
            merged_tables = filtered_tables
        
        self.logger.info(f"Table processing: {len(all_tables)} -> {len(deduplicated_tables)} -> "
                        f"{len(quality_filtered)} -> {len(merged_tables)}",
                        extra={'document_id': document_id})
        
        return merged_tables

    def _deduplicate_tables_improved(self, tables: List[TableStructure]) -> List[TableStructure]:
        """
        CORRECTED: Improved deduplication using Jaccard similarity and fuzzy matching.
        """
        
        if not HAS_TEXT_SIMILARITY:
            # Fallback to simple deduplication
            return self._deduplicate_tables_simple(tables)
        
        unique_tables = []
        
        for table in tables:
            is_duplicate = False
            
            for existing in unique_tables:
                if self._tables_similar_improved(table, existing):
                    # Keep the higher quality table
                    if table.quality_score > existing.quality_score:
                        unique_tables.remove(existing)
                        unique_tables.append(table)
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                unique_tables.append(table)
        
        return unique_tables

    def _tables_similar_improved(self, table1: TableStructure, table2: TableStructure) -> bool:
        """
        CORRECTED: Improved similarity detection using Jaccard similarity.
        """
        
        # Quick dimension check (but not strict equality)
        if abs(len(table1.headers) - len(table2.headers)) > 2:
            return False
        
        if abs(len(table1.data) - len(table2.data)) > len(table1.data) * 0.5:
            return False
        
        # CORRECTED: Jaccard similarity for headers
        headers1_set = set(h.lower().strip() for h in table1.headers)
        headers2_set = set(h.lower().strip() for h in table2.headers)
        
        if not headers1_set or not headers2_set:
            return False
        
        header_jaccard = len(headers1_set & headers2_set) / len(headers1_set | headers2_set)
        
        if header_jaccard < 0.6:  # Less strict than before
            return False
        
        # CORRECTED: Fuzzy cell sampling for data similarity
        sample_size = min(10, len(table1.data), len(table2.data))
        if sample_size == 0:
            return header_jaccard > 0.8
        
        data_matches = 0
        total_cells = 0
        
        for i in range(sample_size):
            row1 = table1.data[i] if i < len(table1.data) else []
            row2 = table2.data[i] if i < len(table2.data) else []
            
            for j in range(min(len(row1), len(row2))):
                total_cells += 1
                cell1 = row1[j].lower().strip()
                cell2 = row2[j].lower().strip()
                
                # Exact match
                if cell1 == cell2:
                    data_matches += 1
                # Fuzzy match for non-empty cells
                elif cell1 and cell2 and len(cell1) > 3 and len(cell2) > 3:
                    similarity = SequenceMatcher(None, cell1, cell2).ratio()
                    if similarity > 0.8:
                        data_matches += 0.8  # Partial credit
        
        data_similarity = data_matches / total_cells if total_cells > 0 else 0
        
        # Combined similarity threshold
        combined_similarity = (header_jaccard * 0.7) + (data_similarity * 0.3)
        return combined_similarity > 0.7

    def _merge_continued_tables_improved(self, tables: List[TableStructure]) -> List[TableStructure]:
        """
        CORRECTED: Improved table continuation detection with fuzzy header matching.
        """
        
        if len(tables) <= 1:
            return tables
        
        merged_tables = []
        processed_indices = set()
        
        # Sort tables by page number and position for better continuation detection
        sorted_tables = sorted(tables, key=lambda t: (t.page_number or 0, t.table_id))
        
        for i, table in enumerate(sorted_tables):
            if i in processed_indices:
                continue
            
            continuation_candidates = []
            
            # Look for continuations in subsequent tables
            for j in range(i + 1, len(sorted_tables)):
                if j in processed_indices:
                    continue
                
                other_table = sorted_tables[j]
                
                if self._is_table_continuation_improved(table, other_table):
                    continuation_candidates.append((j, other_table))
            
            if continuation_candidates:
                # Merge table with its continuations
                all_parts = [table] + [candidate[1] for candidate in continuation_candidates]
                merged_table = self._merge_table_sequence_improved(all_parts)
                merged_tables.append(merged_table)
                
                # Mark as processed
                processed_indices.add(i)
                for idx, _ in continuation_candidates:
                    processed_indices.add(idx)
            else:
                merged_tables.append(table)
                processed_indices.add(i)
        
        return merged_tables

    def _is_table_continuation_improved(self, table1: TableStructure, table2: TableStructure) -> bool:
        """
        CORRECTED: Improved continuation detection with fuzzy header matching and column reordering.
        """
        
        if not HAS_TEXT_SIMILARITY:
            return False
        
        # Quick size check
        if abs(len(table1.headers) - len(table2.headers)) > 1:
            return False
        
        # CORRECTED: Fuzzy header matching with Levenshtein distance
        headers1 = [h.lower().strip() for h in table1.headers]
        headers2 = [h.lower().strip() for h in table2.headers]
        
        # Try direct order first
        direct_match_score = 0
        for h1, h2 in zip(headers1, headers2):
            if h1 == h2:
                direct_match_score += 1
            elif h1 and h2:
                # Use Levenshtein distance for fuzzy matching
                distance = jellyfish.levenshtein_distance(h1, h2)
                max_len = max(len(h1), len(h2))
                similarity = 1 - (distance / max_len) if max_len > 0 else 0
                if similarity >= 0.9:  # High similarity threshold
                    direct_match_score += similarity
        
        direct_match_ratio = direct_match_score / max(len(headers1), len(headers2), 1)
        
        if direct_match_ratio >= 0.8:
            return True
        
        # CORRECTED: Try to handle column reordering
        # Create best mapping between headers
        if len(headers1) == len(headers2):
            best_mapping_score = 0
            
            # For small tables, try all permutations; for larger ones, use greedy matching
            if len(headers1) <= 6:
                from itertools import permutations
                for perm in permutations(headers2):
                    score = sum(1 for h1, h2 in zip(headers1, perm) if h1 == h2)
                    best_mapping_score = max(best_mapping_score, score)
            else:
                # Greedy matching for larger tables
                used_indices = set()
                for h1 in headers1:
                    best_match = 0
                    best_idx = -1
                    for i, h2 in enumerate(headers2):
                        if i in used_indices:
                            continue
                        if h1 == h2:
                            best_match = 1
                            best_idx = i
                            break
                        elif h1 and h2:
                            distance = jellyfish.levenshtein_distance(h1, h2)
                            similarity = 1 - (distance / max(len(h1), len(h2)))
                            if similarity > best_match and similarity >= 0.9:
                                best_match = similarity
                                best_idx = i
                    
                    if best_idx >= 0:
                        best_mapping_score += best_match
                        used_indices.add(best_idx)
            
            mapping_ratio = best_mapping_score / len(headers1)
            if mapping_ratio >= 0.8:
                return True
        
        # Check for explicit continuation indicators
        if table2.data and len(table2.data) > 0:
            first_row_text = ' '.join(table2.data[0]).lower()
            continuation_indicators = ['continued', 'cont.', '(cont)', 'page']
            if any(indicator in first_row_text for indicator in continuation_indicators):
                return direct_match_ratio >= 0.6  # Lower threshold with explicit indicators
        
        # Page sequence check
        if (table1.page_number and table2.page_number and 
            table2.page_number == table1.page_number + 1 and
            direct_match_ratio >= 0.7):
            return True
        
        return False

    def _merge_table_sequence_improved(self, tables: List[TableStructure]) -> TableStructure:
        """Improved table sequence merging with header alignment."""
        
        if not tables:
            return None
        
        base_table = tables[0]
        merged_data = list(base_table.data)
        
        # Track provenance from all merged tables
        all_provenance = [base_table.provenance]
        min_confidence = base_table.confidence
        
        for table in tables[1:]:
            # Align data with base table headers if needed
            if len(table.headers) == len(base_table.headers):
                # Check if headers need reordering
                reordered_data = self._reorder_data_to_match_headers(
                    table.data, table.headers, base_table.headers
                )
                merged_data.extend(reordered_data)
            else:
                # Simple concatenation if headers don't align
                merged_data.extend(table.data)
            
            all_provenance.append(table.provenance)
            min_confidence = min(min_confidence, table.confidence)
        
        # Create merged table structure
        merged_table = TableStructure(
            data=merged_data,
            headers=base_table.headers,
            page_number=base_table.page_number,
            bbox=base_table.bbox,
            table_id=f"merged_{base_table.table_id}",
            confidence=min_confidence,
            extraction_method=f"merged_{base_table.extraction_method}",
            structure_type="merged_continuation",
            provenance={
                "merged_from": all_provenance,
                "merge_method": "continuation",
                "table_count": len(tables)
            },
            quality_score=min(table.quality_score for table in tables),
            column_types=base_table.column_types,
            data_quality_metrics=self._recalculate_quality_metrics_for_merged(merged_data, base_table.headers)
        )
        
        return merged_table

    def _reorder_data_to_match_headers(self, 
                                     data: List[List[str]], 
                                     source_headers: List[str], 
                                     target_headers: List[str]) -> List[List[str]]:
        """Reorder data columns to match target header order."""
        
        if len(source_headers) != len(target_headers):
            return data  # Can't reorder if different sizes
        
        # Create mapping from source to target positions
        header_mapping = {}
        for i, source_header in enumerate(source_headers):
            source_lower = source_header.lower().strip()
            for j, target_header in enumerate(target_headers):
                target_lower = target_header.lower().strip()
                if source_lower == target_lower:
                    header_mapping[i] = j
                    break
        
        # If we couldn't map most headers, return original data
        if len(header_mapping) < len(source_headers) * 0.8:
            return data
        
        # Reorder data
        reordered_data = []
        for row in data:
            reordered_row = [''] * len(target_headers)
            for source_idx, target_idx in header_mapping.items():
                if source_idx < len(row):
                    reordered_row[target_idx] = row[source_idx]
            reordered_data.append(reordered_row)
        
        return reordered_data

# ... [Continue with remaining methods - the implementation would continue with the other extraction methods, quality calculations, insights generation, etc. This shows the key production corrections requested] ...
