#!/usr/bin/env python3
"""
SessionExtractor - Extract knowledge from OpenClaw session JSONL files.

Features:
- Parse JSONL session format
- Incremental processing (tracks line counts)
- Focus on assistant + user content
- Skip tool calls and system messages
- Extract knowledge using model_fn
- .cashewignore support
"""

import json
import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.extractors import BaseExtractor
from extractors.utils import load_ignore_patterns, parse_extraction_lines, should_ignore

logger = logging.getLogger("cashew.extractors.sessions")


class SessionExtractor(BaseExtractor):
    """Extract knowledge from OpenClaw session JSONL files."""

    @property
    def name(self) -> str:
        return "sessions"

    def __init__(self):
        # Track processed files and their line counts
        self._processed: Dict[str, int] = {}

    def extract(self, source_path: str, model_fn: Optional[Callable], 
                db_path: str) -> List[Dict[str, Any]]:
        """Extract knowledge from session directory."""
        source_dir = Path(source_path)
        
        if not source_dir.exists():
            logger.error(f"Session directory does not exist: {source_path}")
            return []

        # Find all .jsonl files
        if source_dir.is_file() and source_dir.suffix == '.jsonl':
            session_files = [source_dir]
        elif source_dir.is_dir():
            ignore_patterns = load_ignore_patterns(source_dir / ".cashewignore")
            session_files = [
                f for f in source_dir.glob("*.jsonl")
                if not should_ignore(f, source_dir, ignore_patterns)
            ]
        else:
            logger.error(f"Invalid source path: {source_path}")
            return []

        nodes = []

        # Process each session file
        for session_file in session_files:
            session_id = session_file.stem
            file_path = str(session_file)
            
            # Get current line count
            try:
                with open(session_file, 'r', encoding='utf-8') as f:
                    current_lines = sum(1 for _ in f)
            except IOError as e:
                logger.warning(f"Could not read {session_file}: {e}")
                continue

            # Check if already processed
            processed_lines = self._processed.get(file_path, 0)
            if processed_lines >= current_lines:
                continue  # No new content

            # Read new lines only
            new_messages = self._read_new_messages(
                session_file, processed_lines, current_lines)
            
            if not new_messages:
                continue

            # Extract knowledge from conversation
            if model_fn:
                extracted_nodes = self._extract_with_llm(
                    new_messages, model_fn, session_id)
            else:
                extracted_nodes = self._extract_simple(
                    new_messages, session_id)

            nodes.extend(extracted_nodes)
            self._processed[file_path] = current_lines

        return nodes

    def _read_new_messages(self, session_file: Path, start_line: int, 
                          end_line: int) -> List[Dict[str, Any]]:
        """Read new messages from JSONL file."""
        messages = []
        
        try:
            with open(session_file, 'r', encoding='utf-8') as f:
                for i, line in enumerate(f):
                    if i < start_line:
                        continue
                    if i >= end_line:
                        break
                    
                    try:
                        message = json.loads(line.strip())
                        # Filter relevant messages
                        if self._is_relevant_message(message):
                            messages.append(message)
                    except json.JSONDecodeError:
                        continue
                        
        except IOError as e:
            logger.warning(f"Error reading {session_file}: {e}")
        
        return messages

    def _is_relevant_message(self, message: Dict[str, Any]) -> bool:
        """Check if message is relevant for knowledge extraction."""
        role = message.get('role', '')
        content = message.get('content', '')
        
        # Skip system messages and empty content
        if role == 'system' or not content:
            return False
        
        # Skip tool calls (they usually have structured content)
        if isinstance(content, dict) or content.startswith('{'):
            return False
        
        # Skip very short messages
        if len(content) < 50:
            return False
        
        # Focus on assistant and user messages
        return role in ['assistant', 'user']

    def _extract_with_llm(self, messages: List[Dict[str, Any]], 
                          model_fn: Callable, session_id: str) -> List[Dict[str, Any]]:
        """Extract knowledge using LLM."""
        if not messages:
            return []

        # Prepare conversation context
        conversation = []
        for msg in messages:
            role = msg.get('role', '')
            content = msg.get('content', '')
            timestamp = msg.get('timestamp', '')
            
            conversation.append(f"{role.upper()}: {content}")

        conv_text = "\n\n".join(conversation)
        
        prompt = f"""Extract key insights, decisions, commitments, and important information from this conversation.

Session: {session_id}

Conversation:
{conv_text}

Return distinct knowledge statements that would be valuable to remember. Each should be:
- A specific insight, decision, commitment, or important fact
- Actionable or memorable for future reference
- Written in a clear, standalone format

Focus on substantive content and skip pleasantries or routine interactions."""

        try:
            response = model_fn(prompt)
            logger.debug(f"LLM raw ({session_id}):\n{response}\n---")
            statements = parse_extraction_lines(response)
            
            # Event clock: use the latest message timestamp in this batch as the
            # referent_time for extracted nodes. Session extractors already parse
            # per-message timestamps — pass them through so imported historical
            # sessions get proper event times (not today's ingest time).
            batch_referent_time = None
            for msg in messages:
                ts = (msg.get('timestamp') or '').strip()
                if ts:
                    batch_referent_time = ts  # last non-empty wins

            return [{
                "content": stmt,
                "type": self._classify_statement(stmt),
                "confidence": 0.75,
                "domain": "conversations",
                "source_file": f"extractor:session:{session_id}",
                "referent_time": batch_referent_time,
            } for stmt in statements if len(stmt) > 20]
            
        except Exception as e:
            logger.warning(f"LLM extraction failed for {session_id}: {e}")
            return self._extract_simple(messages, session_id)

    def _extract_simple(self, messages: List[Dict[str, Any]], 
                        session_id: str) -> List[Dict[str, Any]]:
        """Simple extraction without LLM."""
        nodes = []
        
        for msg in messages:
            content = msg.get('content', '').strip()
            role = msg.get('role', '')
            ts = (msg.get('timestamp') or '').strip() or None

            if len(content) < 100:  # Skip short messages
                continue

            # Extract longer, substantial messages. Per-message timestamp
            # becomes the node's event clock (referent_time).
            nodes.append({
                "content": f"{role}: {content}",
                "type": "observation",
                "confidence": 0.5,
                "domain": "conversations",
                "source_file": f"extractor:session:{session_id}",
                "referent_time": ts,
            })
        
        return nodes

    def _classify_statement(self, statement: str) -> str:
        """Classify extracted statement type."""
        statement_lower = statement.lower()
        
        # Look for decision keywords
        decision_keywords = ['decided', 'will', 'going to', 'plan to', 'agreed']
        if any(keyword in statement_lower for keyword in decision_keywords):
            return "decision"
        
        # Look for insight/learning keywords
        insight_keywords = ['learned', 'realized', 'discovered', 'found that']
        if any(keyword in statement_lower for keyword in insight_keywords):
            return "insight"
        
        # Look for commitment keywords
        commit_keywords = ['commit', 'promise', 'deadline', 'due', 'by']
        if any(keyword in statement_lower for keyword in commit_keywords):
            return "commitment"
        
        # Default to observation
        return "observation"

    def get_state(self) -> Dict[str, Any]:
        return {"processed": self._processed}

    def set_state(self, state: Dict[str, Any]):
        self._processed = state.get("processed", {})