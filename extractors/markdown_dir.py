#!/usr/bin/env python3
"""
MarkdownDirExtractor - Production markdown directory processor.

Features:
- Recursive .md file discovery
- .cashewignore support
- Checkpointing by file path + mtime
- LLM extraction with paragraph fallback
- Domain detection from folder structure
"""

import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.extractors import BaseExtractor
from extractors.utils import (
    load_ignore_patterns, should_ignore, split_into_paragraphs,
    detect_domain_from_path, parse_extraction_lines
)

logger = logging.getLogger("cashew.extractors.markdown_dir")


class MarkdownDirExtractor(BaseExtractor):
    """Extract knowledge from markdown files in a directory."""

    @property
    def name(self) -> str:
        return "markdown"

    def __init__(self):
        # Track processed files with their mtimes
        self._processed: Dict[str, float] = {}

    def extract(self, source_path: str, model_fn: Optional[Callable], 
                db_path: str) -> List[Dict[str, Any]]:
        """Extract knowledge from markdown files."""
        source_dir = Path(source_path)
        
        if not source_dir.exists():
            logger.error(f"Source path does not exist: {source_path}")
            return []

        # Handle single file vs directory
        if source_dir.is_file() and source_dir.suffix == '.md':
            md_files = [source_dir]
            base_path = source_dir.parent
        elif source_dir.is_dir():
            # Load ignore patterns
            ignore_file = source_dir / ".cashewignore"
            ignore_patterns = load_ignore_patterns(ignore_file)
            
            # Find all markdown files
            md_files = []
            for md_file in source_dir.rglob("*.md"):
                if should_ignore(md_file, source_dir, ignore_patterns):
                    continue
                md_files.append(md_file)
            
            base_path = source_dir
        else:
            logger.error(f"Invalid source path: {source_path}")
            return []

        nodes = []

        # Process each file
        for md_file in md_files:
            rel_path = str(md_file.relative_to(base_path))
            current_mtime = md_file.stat().st_mtime
            
            # Check if already processed and unchanged
            if (rel_path in self._processed and 
                self._processed[rel_path] >= current_mtime):
                continue

            try:
                with open(md_file, 'r', encoding='utf-8') as f:
                    content = f.read()
            except IOError as e:
                logger.warning(f"Could not read {md_file}: {e}")
                continue

            # Skip empty files
            if not content.strip():
                continue

            # Detect domain from folder structure
            domain = detect_domain_from_path(md_file, base_path)

            # Prepare source file tag
            source_tag = f"extractor:markdown:{rel_path}"

            # Extract knowledge using model or fallback to paragraphs
            if model_fn:
                extracted_nodes = self._extract_with_llm(
                    content, model_fn, domain, source_tag)
            else:
                extracted_nodes = self._extract_paragraphs(
                    content, domain, source_tag)

            nodes.extend(extracted_nodes)
            self._processed[rel_path] = current_mtime

        return nodes

    def _extract_with_llm(self, content: str, model_fn: Callable, 
                          domain: str, source_tag: str) -> List[Dict[str, Any]]:
        """Extract knowledge using LLM."""
        prompt = f"""Extract key insights, facts, decisions, and important information from this markdown content.

Content:
{content}

Return distinct knowledge statements that would be valuable to remember. Each should be:
- A specific insight, decision, fact, or important observation
- Standalone and actionable
- Free of markdown formatting
- Focused on substantive content

Extract only meaningful information, skip formatting, navigation, or boilerplate text."""

        try:
            response = model_fn(prompt)
            logger.debug(f"LLM raw ({source_tag}):\n{response}\n---")
            statements = parse_extraction_lines(response)
            
            return [{
                "content": stmt,
                "type": self._classify_content(stmt),
                "confidence": 0.8,
                "domain": domain,
                "source_file": source_tag
            } for stmt in statements if len(stmt) > 15]
            
        except Exception as e:
            logger.warning(f"LLM extraction failed: {e}")
            # Fallback to paragraph extraction
            return self._extract_paragraphs(content, domain, source_tag)

    def _extract_paragraphs(self, content: str, domain: str, 
                            source_tag: str) -> List[Dict[str, Any]]:
        """Fallback extraction using paragraph splitting."""
        paragraphs = split_into_paragraphs(content)
        
        return [{
            "content": para,
            "type": "observation",
            "confidence": 0.6,
            "domain": domain,
            "source_file": source_tag
        } for para in paragraphs]

    def _classify_content(self, content: str) -> str:
        """Classify extracted content type."""
        content_lower = content.lower()
        
        # Look for decision indicators
        decision_words = ['decided', 'choose', 'selected', 'will implement']
        if any(word in content_lower for word in decision_words):
            return "decision"
        
        # Look for belief/opinion indicators
        belief_words = ['believe', 'think', 'opinion', 'should', 'recommend']
        if any(word in content_lower for word in belief_words):
            return "belief"
        
        # Look for factual indicators
        fact_words = ['is', 'are', 'was', 'were', 'has', 'have', 'definition']
        if any(word in content_lower for word in fact_words):
            return "fact"
        
        # Look for insight indicators
        insight_words = ['learned', 'discovered', 'insight', 'realization']
        if any(word in content_lower for word in insight_words):
            return "insight"
        
        # Default to observation
        return "observation"

    def get_state(self) -> Dict[str, Any]:
        return {"processed": self._processed}

    def set_state(self, state: Dict[str, Any]):
        self._processed = state.get("processed", {})