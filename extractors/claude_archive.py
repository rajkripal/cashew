#!/usr/bin/env python3
"""
ClaudeArchiveExtractor - Extract knowledge from claude.ai conversation archives.

Usage:
    cashew ingest claude_archive /path/to/claude/export/
    cashew ingest claude_archive /path/to/claude/export/conversations.json

Input format:
    Claude.ai data export produces a `conversations.json` file — a JSON array of
    conversation objects. Each conversation has:
        - uuid: unique conversation ID
        - name: conversation title (user-given or auto-generated)
        - chat_messages: list of message objects, each with:
            - sender: "human" or "assistant"
            - text: the message text (may contain tool output blocks)
            - content: list of content blocks (text, tool_use, tool_result)
            - created_at: ISO-8601 timestamp
            - parent_message_uuid: threading/parent reference

    Tool-use and tool-result blocks are filtered out. Only substantive human and
    assistant messages are passed to the LLM extraction pipeline.

Features:
    - Incremental processing (tracks processed conversation UUIDs)
    - LLM-based extraction via model_fn (with typed node tagging)
    - Simple fallback extraction when no LLM is available
    - referent_time preservation from conversation timestamps
    - .cashewignore support
    - Handles both the full export directory or a single conversations.json
"""

import json
import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.extractors import BaseExtractor
from extractors.utils import (
    TYPE_TAGGING_INSTRUCTION,
    load_ignore_patterns,
    parse_extraction_lines,
    parse_typed_statement,
    should_ignore,
)

logger = logging.getLogger("cashew.extractors.claude_archive")


class ClaudeArchiveExtractor(BaseExtractor):
    """Extract knowledge from claude.ai conversation archives."""

    @property
    def name(self) -> str:
        return "claude_archive"

    def __init__(self):
        # Track processed conversation UUIDs
        self._processed: Dict[str, str] = {}  # uuid -> updated_at timestamp

    def extract(self, source_path: str, model_fn: Optional[Callable],
                db_path: str) -> List[Dict[str, Any]]:
        """Extract knowledge from claude.ai export directory or file."""
        source = Path(source_path)

        if not source.exists():
            logger.error(f"Source path does not exist: {source_path}")
            return []

        # Locate conversations.json
        if source.is_file() and source.name == "conversations.json":
            conv_file = source
            base_dir = source.parent
        elif source.is_dir():
            conv_file = source / "conversations.json"
            base_dir = source
        else:
            logger.error(f"Expected a claude.ai export directory or conversations.json, got: {source_path}")
            return []

        if not conv_file.exists():
            logger.error(f"conversations.json not found at {conv_file}")
            return []

        # Load conversations
        try:
            with open(conv_file, 'r', encoding='utf-8') as f:
                conversations = json.load(f)
        except (IOError, json.JSONDecodeError) as e:
            logger.error(f"Failed to load {conv_file}: {e}")
            return []

        if not isinstance(conversations, list):
            logger.error(f"Expected JSON array in conversations.json, got {type(conversations).__name__}")
            return []

        # Load ignore patterns
        ignore_patterns = load_ignore_patterns(base_dir / ".cashewignore")

        nodes = []

        for conv in conversations:
            conv_uuid = conv.get("uuid", "")
            if not conv_uuid:
                continue

            # Check ignore patterns
            if should_ignore(Path(conv_uuid), base_dir, ignore_patterns):
                continue

            # Check if already processed (by updated_at to catch re-exports)
            updated_at = conv.get("updated_at", "")
            already_processed = self._processed.get(conv_uuid, "")
            if already_processed and updated_at and updated_at <= already_processed:
                continue

            # Extract messages
            messages = conv.get("chat_messages", [])
            if not messages:
                continue

            # Filter and prepare conversation turns
            turns = self._extract_turns(messages)
            if not turns:
                continue

            # Assign conversation-level metadata
            conv_name = conv.get("name", conv_uuid[:12])

            # Extract knowledge
            if model_fn:
                extracted_nodes = self._extract_with_llm(turns, model_fn, conv_uuid, conv_name)
            else:
                extracted_nodes = self._extract_simple(turns, conv_uuid, conv_name)

            nodes.extend(extracted_nodes)
            self._processed[conv_uuid] = updated_at or conv.get("created_at", "")

        return nodes

    def _extract_turns(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Extract human/assistant turns, filtering out tool use and system content.

        Claude.ai messages have a `content` field which is a list of content blocks.
        Each block has a `type`: "text" (human text), "tool_use", "tool_result", etc.
        The top-level `text` field is a concatenation of all text blocks, which often
        includes tool output interspersed.

        Strategy: walk the content blocks, keep only type="text" blocks for both
        human and assistant messages. This strips tool calls and results cleanly.
        """
        turns = []

        for msg in messages:
            sender = msg.get("sender", "")
            if sender not in ("human", "assistant"):
                continue

            created_at = msg.get("created_at", "")

            # Extract only text content blocks (skip tool_use, tool_result, etc.)
            content_blocks = msg.get("content", [])
            text_parts = []
            for block in content_blocks:
                if isinstance(block, dict) and block.get("type") == "text":
                    text = block.get("text", "").strip()
                    if text:
                        text_parts.append(text)

            if not text_parts:
                # Fall back to the top-level `text` field if content blocks are empty
                full_text = (msg.get("text") or "").strip()
                if not full_text:
                    continue
                text_parts = [full_text]

            # Join the text parts
            combined = "\n\n".join(text_parts)

            # Filter out very short or empty messages
            if len(combined) < 50:
                continue

            # Filter out messages that are entirely tool output artifacts
            # (Claude.ai sometimes renders tool output as text blocks on unsupported devices)
            tool_artifact_indicators = [
                "This block is not supported on your current device yet.",
                "[Tool output truncated]",
            ]
            if any(indicator in combined for indicator in tool_artifact_indicators):
                # Only keep this message if it has substantial non-tool content
                clean_text = combined
                for indicator in tool_artifact_indicators:
                    clean_text = clean_text.replace(indicator, "")
                if len(clean_text.strip()) < 50:
                    continue
                combined = clean_text.strip()

            turns.append({
                "sender": sender,
                "text": combined,
                "created_at": created_at,
            })

        return turns

    def _extract_with_llm(self, turns: List[Dict[str, Any]],
                          model_fn: Callable, conv_uuid: str,
                          conv_name: str) -> List[Dict[str, Any]]:
        """Extract knowledge using LLM."""
        if not turns:
            return []

        # Build conversation transcript
        conversation_lines = []
        batch_referent_time = None
        for turn in turns:
            role_label = "HUMAN" if turn["sender"] == "human" else "CLAUDE"
            conversation_lines.append(f"{role_label}: {turn['text']}")
            ts = (turn.get("created_at") or "").strip()
            if ts:
                batch_referent_time = ts  # last non-empty wins

        conv_text = "\n\n".join(conversation_lines)

        prompt = f"""Extract key insights, decisions, commitments, and important information from this conversation.

Conversation: {conv_name}

{conv_text}

Return distinct knowledge statements that would be valuable to remember. Each should be:
- A specific insight, decision, commitment, or important fact
- Actionable or memorable for future reference
- Written in a clear, standalone format

Before emitting each statement, ask yourself: should this node exist in the graph forever? If you would not want future-you to read it, drop it. There is no hedging and no padding, either it is worth a permanent node or it is not. Skip pleasantries and routine interactions.

{TYPE_TAGGING_INSTRUCTION}"""

        try:
            response = model_fn(prompt)
            logger.debug(f"LLM raw ({conv_uuid[:12]}):\n{response}\n---")
            statements = parse_extraction_lines(response)

            nodes_out = []
            for raw in statements:
                node_type, content = parse_typed_statement(
                    raw, fallback=self._classify_statement)
                if len(content) <= 20:
                    continue
                nodes_out.append({
                    "content": content,
                    "type": node_type,
                    "domain": "claude_conversations",
                    "source_file": f"extractor:claude_archive:{conv_uuid}",
                    "referent_time": batch_referent_time,
                })
            return nodes_out

        except Exception as e:
            logger.warning(f"LLM extraction failed for {conv_uuid[:12]}: {e}")
            return self._extract_simple(turns, conv_uuid, conv_name)

    def _extract_simple(self, turns: List[Dict[str, Any]],
                        conv_uuid: str, conv_name: str) -> List[Dict[str, Any]]:
        """Simple extraction without LLM."""
        nodes = []

        for turn in turns:
            text = turn.get("text", "").strip()
            if len(text) < 50:  # Skip short messages (turns filter already handles this)
                continue

            role = turn.get("sender", "unknown")
            ts = (turn.get("created_at") or "").strip() or None

            nodes.append({
                "content": f"{conv_name} ({role}): {text[:500]}",
                "type": "observation",
                "domain": "claude_conversations",
                "source_file": f"extractor:claude_archive:{conv_uuid}",
                "referent_time": ts,
            })

        return nodes

    def _classify_statement(self, statement: str) -> str:
        """Classify extracted statement type."""
        statement_lower = statement.lower()

        decision_keywords = ['decided', 'will', 'going to', 'plan to', 'agreed']
        if any(keyword in statement_lower for keyword in decision_keywords):
            return "decision"

        insight_keywords = ['learned', 'realized', 'discovered', 'found that']
        if any(keyword in statement_lower for keyword in insight_keywords):
            return "insight"

        commit_keywords = ['commit', 'promise', 'deadline', 'due', 'by']
        if any(keyword in statement_lower for keyword in commit_keywords):
            return "commitment"

        return "observation"

    def get_state(self) -> Dict[str, Any]:
        return {"processed": self._processed}

    def set_state(self, state: Dict[str, Any]):
        self._processed = state.get("processed", {})
