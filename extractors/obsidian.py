#!/usr/bin/env python3
"""
ObsidianExtractor - Extract knowledge from Obsidian vault markdown files.

Features:
- Recursive .md file discovery
- YAML frontmatter parsing (tags, aliases, dates)
- Wikilink relationship detection
- .obsidianignore support
- Checkpointing by file mtime
- Auto-domain detection from folder structure
"""

import os
import logging
import sqlite3
from pathlib import Path
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.extractors import BaseExtractor
from extractors.utils import (
    TYPE_TAGGING_INSTRUCTION,
    parse_frontmatter, extract_wikilinks, load_ignore_patterns,
    should_ignore, split_into_paragraphs, detect_domain_from_path,
    parse_extraction_lines, parse_typed_statement,
)

logger = logging.getLogger("cashew.extractors.obsidian")


class ObsidianExtractor(BaseExtractor):
    """Extract knowledge from Obsidian vault markdown files."""

    @property
    def name(self) -> str:
        return "obsidian"

    def __init__(self):
        # Track processed files and their mtimes
        self._processed: Dict[str, float] = {}
        # Track wikilink relationships for edge creation
        self._note_links: Dict[str, List[str]] = {}

    def extract(self, source_path: str, model_fn: Optional[Callable], 
                db_path: str) -> List[Dict[str, Any]]:
        """Extract knowledge from Obsidian vault."""
        vault_path = Path(source_path)
        
        if not vault_path.exists() or not vault_path.is_dir():
            logger.error(f"Vault path does not exist or is not a directory: {source_path}")
            return []

        # Load ignore patterns
        ignore_file = vault_path / ".obsidianignore"
        ignore_patterns = load_ignore_patterns(ignore_file)
        
        # Find all markdown files
        md_files = []
        for md_file in vault_path.rglob("*.md"):
            if should_ignore(md_file, vault_path, ignore_patterns):
                continue
            md_files.append(md_file)

        nodes = []
        self._note_links.clear()

        # Process each file
        for md_file in md_files:
            rel_path = str(md_file.relative_to(vault_path))
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

            # Parse frontmatter and extract wikilinks
            metadata, body = parse_frontmatter(content)
            wikilinks = extract_wikilinks(content)
            
            # Store wikilinks for edge creation
            note_name = md_file.stem
            self._note_links[note_name] = wikilinks

            # Detect domain from folder structure
            domain = detect_domain_from_path(md_file, vault_path)

            # Prepare source file tag
            source_tag = f"extractor:obsidian:{rel_path}"

            # Extract knowledge using model or fallback to paragraphs
            if model_fn:
                extracted_nodes = self._extract_with_llm(
                    content, metadata, model_fn, domain, source_tag)
            else:
                extracted_nodes = self._extract_paragraphs(
                    body, domain, source_tag)

            nodes.extend(extracted_nodes)
            self._processed[rel_path] = current_mtime

        return nodes

    def _extract_with_llm(self, content: str, metadata: Dict[str, Any], 
                          model_fn: Callable, domain: str, 
                          source_tag: str) -> List[Dict[str, Any]]:
        """Extract knowledge using LLM."""
        # Prepare metadata context
        meta_context = ""
        if metadata:
            tags = metadata.get('tags', [])
            aliases = metadata.get('aliases', [])
            if tags:
                meta_context += f"Tags: {', '.join(tags)}\n"
            if aliases:
                meta_context += f"Aliases: {', '.join(aliases)}\n"

        prompt = f"""Extract key insights, decisions, beliefs, and factual information from this Obsidian note.

{meta_context}

Content:
{content}

Return a list of distinct knowledge statements. Each should be:
- A standalone insight, decision, belief, or fact
- Actionable or memorable
- Free of markdown formatting

Before emitting each statement, ask yourself: should this node exist in the graph forever? If you would not want future-you to read it, drop it. There is no hedging and no padding, either it is worth a permanent node or it is not. Focus on substantive content, not formatting or structure.

{TYPE_TAGGING_INSTRUCTION}"""

        try:
            response = model_fn(prompt)
            logger.debug(f"LLM raw ({source_tag}):\n{response}\n---")
            statements = parse_extraction_lines(response)
            
            nodes_out = []
            for raw in statements:
                # Obsidian historically defaulted untagged notes to "insight"
                # rather than "observation"; preserve that for back-compat
                # when the LLM does not tag.
                node_type, content = parse_typed_statement(
                    raw, default_type="insight")
                if len(content) <= 20:
                    continue
                nodes_out.append({
                    "content": content,
                    "type": node_type,
                    "domain": domain,
                    "source_file": source_tag,
                })
            return nodes_out
        except Exception as e:
            logger.warning(f"LLM extraction failed: {e}")
            # Fallback to paragraph extraction
            _, body = parse_frontmatter(content)
            return self._extract_paragraphs(body, domain, source_tag)

    def _extract_paragraphs(self, content: str, domain: str, 
                            source_tag: str) -> List[Dict[str, Any]]:
        """Fallback extraction using paragraph splitting."""
        paragraphs = split_into_paragraphs(content)
        
        return [{
            "content": para,
            "type": "observation",
            "domain": domain,
            "source_file": source_tag
        } for para in paragraphs]

    def _create_wikilink_edges(self, db_path: str, vault_path: Path):
        """Create edges between notes based on wikilink relationships."""
        if not self._note_links:
            return

        try:
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()

            # Get existing nodes with their source files
            cursor.execute("""
                SELECT id, source_file FROM thought_nodes 
                WHERE source_file LIKE 'extractor:obsidian:%'
            """)
            nodes = {row[1]: row[0] for row in cursor.fetchall()}

            # Create edges for wikilinks
            edges_created = 0
            for note_name, linked_notes in self._note_links.items():
                # Find source node
                source_pattern = f"extractor:obsidian:%{note_name}.md%"
                source_nodes = [node_id for source_file, node_id in nodes.items() 
                               if source_file.endswith(f":{note_name}.md")]
                
                if not source_nodes:
                    continue
                
                source_node = source_nodes[0]  # Use first match
                
                # Create edges to linked notes
                for linked_note in linked_notes:
                    target_pattern = f"extractor:obsidian:%{linked_note}.md%"
                    target_nodes = [node_id for source_file, node_id in nodes.items() 
                                   if source_file.endswith(f":{linked_note}.md")]
                    
                    if not target_nodes:
                        continue
                    
                    target_node = target_nodes[0]  # Use first match
                    
                    # Create bidirectional edge
                    try:
                        cursor.execute("""
                            INSERT OR IGNORE INTO derivation_edges
                            (parent_id, child_id, weight, reasoning, timestamp)
                            VALUES (?, ?, ?, ?, ?)
                        """, (source_node, target_node, 1.0,
                              f"Wikilink: {note_name} -> {linked_note}",
                              datetime.now().isoformat()))

                        # Create reverse edge too
                        cursor.execute("""
                            INSERT OR IGNORE INTO derivation_edges
                            (parent_id, child_id, weight, reasoning, timestamp)
                            VALUES (?, ?, ?, ?, ?)
                        """, (target_node, source_node, 1.0,
                              f"Wikilink: {linked_note} -> {note_name}",
                              datetime.now().isoformat()))
                        
                        edges_created += 2
                    except sqlite3.Error as e:
                        logger.warning(f"Failed to create edge: {e}")

            conn.commit()
            conn.close()
            
            if edges_created > 0:
                logger.info(f"Created {edges_created} wikilink edges")
                
        except sqlite3.Error as e:
            logger.error(f"Failed to create wikilink edges: {e}")

    def post_ingest_hook(self, source_path: str, db_path: str, nodes_created: int):
        """Create wikilink edges after nodes have been ingested."""
        if nodes_created > 0:
            vault_path = Path(source_path)
            self._create_wikilink_edges(db_path, vault_path)

    def get_state(self) -> Dict[str, Any]:
        return {"processed": self._processed}

    def set_state(self, state: Dict[str, Any]):
        self._processed = state.get("processed", {})