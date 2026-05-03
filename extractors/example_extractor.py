#!/usr/bin/env python3
"""
Example Extractor — reference implementation showing how to write a cashew extractor plugin.

Usage:
    from cashew.core.extractors import ExtractorRegistry
    from cashew.extractors.example_extractor import MarkdownExtractor
    
    registry = ExtractorRegistry(data_dir="./data")
    registry.register(MarkdownExtractor())
    result = registry.run("markdown", "/path/to/notes/", model_fn=None, db_path="./data/graph.db")
"""

import os
import glob
from typing import Any, Callable, Dict, List, Optional

from ..core.extractors import BaseExtractor


class MarkdownExtractor(BaseExtractor):
    """Extracts knowledge from markdown files.
    
    Tracks which files have been processed to avoid re-extraction.
    A simple example — real extractors would use model_fn for
    LLM-based extraction.
    """

    @property
    def name(self) -> str:
        return "markdown"

    def __init__(self):
        self._processed: List[str] = []

    def extract(self, source_path: str, model_fn: Optional[Callable],
                db_path: str) -> List[Dict[str, Any]]:
        """Extract knowledge from markdown files in source_path.
        
        For each unprocessed .md file, extracts paragraphs as observation nodes.
        In a real extractor, you'd use model_fn to do LLM-based extraction.
        """
        nodes = []
        
        if os.path.isfile(source_path):
            files = [source_path]
        elif os.path.isdir(source_path):
            files = glob.glob(os.path.join(source_path, "**/*.md"), recursive=True)
        else:
            return nodes

        for filepath in files:
            if filepath in self._processed:
                continue

            try:
                with open(filepath, 'r') as f:
                    content = f.read()
            except IOError:
                continue

            # Simple paragraph extraction (real extractors use model_fn)
            paragraphs = [p.strip() for p in content.split('\n\n') if p.strip()]
            for para in paragraphs:
                if len(para) < 20:  # skip short fragments
                    continue
                nodes.append({
                    "content": para,
                    "type": "observation",
                    "source_file": f"extractor:markdown:{os.path.basename(filepath)}",
                })

            self._processed.append(filepath)

        return nodes

    def get_state(self) -> Dict[str, Any]:
        return {"processed": self._processed}

    def set_state(self, state: Dict[str, Any]):
        self._processed = state.get("processed", [])
