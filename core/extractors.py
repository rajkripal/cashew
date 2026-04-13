#!/usr/bin/env python3
"""
Cashew Extractor Plugin Interface
Domain-specific extractors register here. The orchestrator discovers and runs them.
"""

import json
import logging
import os
import uuid
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger("cashew.extractors")


class BaseExtractor(ABC):
    """Base class for domain-specific knowledge extractors.
    
    Subclass this to create extractors for specific sources
    (meeting transcripts, memos, chat logs, etc.).
    """
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Unique extractor name (e.g. 'granola', 'memo', 'session')."""
        ...

    @abstractmethod
    def extract(self, source_path: str, model_fn: Optional[Callable], db_path: str) -> List[Dict[str, Any]]:
        """Extract knowledge from a source.
        
        Args:
            source_path: Path to the source file/directory to extract from.
            model_fn: LLM function provided by the orchestrator (may be None).
            db_path: Path to the cashew graph database.
            
        Returns:
            List of node dicts, each with keys:
                - content (str, required): The knowledge statement
                - type (str, required): Node type (belief, insight, decision, observation, fact)
                - confidence (float, required): 0.0-1.0
                - domain (str, optional): Domain classification
                - source_file (str, optional): Source identifier
        """
        ...

    def get_state(self) -> Dict[str, Any]:
        """Return extractor state (e.g. which files have been processed).
        Override to track processing state across runs.
        """
        return {}

    def set_state(self, state: Dict[str, Any]):
        """Restore extractor state from a previously saved dict.
        Override to restore processing state.
        """
        pass

    def post_ingest_hook(self, source_path: str, db_path: str, nodes_created: int):
        """Called after nodes have been successfully ingested into the database.
        Override to perform additional processing like edge creation.
        """
        pass


class ExtractorRegistry:
    """Registry for extractor plugins.
    
    Extractors register themselves, then the orchestrator can
    discover and run them.
    """

    def __init__(self, data_dir: str = "./data"):
        self._extractors: Dict[str, BaseExtractor] = {}
        self._state_dir = os.path.join(data_dir, "extractor_state")
        os.makedirs(self._state_dir, exist_ok=True)

    def register(self, extractor: BaseExtractor):
        """Register an extractor plugin."""
        if not isinstance(extractor, BaseExtractor):
            raise TypeError(f"Expected BaseExtractor, got {type(extractor).__name__}")
        name = extractor.name
        if not name:
            raise ValueError("Extractor must have a non-empty name")
        if name in self._extractors:
            raise ValueError(f"Extractor '{name}' already registered")
        self._extractors[name] = extractor
        # Load persisted state
        state = self._load_state(name)
        if state:
            extractor.set_state(state)
        logger.info(f"Registered extractor: {name}")

    def unregister(self, name: str):
        """Remove an extractor from the registry. Raises KeyError if not found."""
        if name not in self._extractors:
            raise KeyError(f"Extractor '{name}' not registered")
        del self._extractors[name]
        logger.info(f"Unregistered extractor: {name}")

    def get(self, name: str) -> BaseExtractor:
        """Get an extractor by name. Raises KeyError if not found."""
        if name not in self._extractors:
            raise KeyError(f"Extractor '{name}' not registered")
        return self._extractors[name]

    def list_extractors(self) -> List[str]:
        """List registered extractor names."""
        return list(self._extractors.keys())

    def run(self, name: str, source_path: str, model_fn: Optional[Callable],
            db_path: str) -> Dict[str, Any]:
        """Run a specific extractor on a source.
        
        Returns:
            Dict with keys: nodes_created (int), errors (list)
        """
        if name not in self._extractors:
            return {"nodes_created": 0, "errors": [f"Extractor '{name}' not found"]}
        extractor = self._extractors[name]

        result = {"nodes_created": 0, "errors": []}
        try:
            nodes = extractor.extract(source_path, model_fn, db_path)
            if nodes:
                created = self._ingest_nodes(nodes, db_path, extractor.name)
                result["nodes_created"] = created
                
                # Call post-ingest hook for additional processing (e.g. edge creation)
                extractor.post_ingest_hook(source_path, db_path, created)
            
            # Persist state after successful run
            self._save_state(name, extractor.get_state())
        except Exception as e:
            logger.error(f"Extractor '{name}' failed: {e}")
            result["errors"].append(str(e))

        return result

    def run_all(self, model_fn: Optional[Callable], db_path: str,
                source_path: str = ".") -> Dict[str, Dict[str, Any]]:
        """Run all registered extractors.
        
        Returns:
            Dict mapping extractor name -> {nodes_created, errors}
        """
        results = {}
        for name in self._extractors:
            results[name] = self.run(name, source_path, model_fn, db_path)
        return results

    def _ingest_nodes(self, nodes: List[Dict[str, Any]], db_path: str,
                      extractor_name: str) -> int:
        """Ingest extracted nodes into the graph database."""
        import sqlite3
        
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        created = 0

        for node in nodes:
            content = node.get("content", "").strip()
            if not content:
                continue

            node_id = str(uuid.uuid4())[:12]
            node_type = node.get("type", "observation")
            confidence = float(node.get("confidence", 0.7))
            domain = node.get("domain", "default")
            source_file = node.get("source_file", f"extractor:{extractor_name}")
            now = datetime.now().isoformat()

            # Dedup: skip if identical content already exists
            cursor.execute(
                "SELECT 1 FROM thought_nodes WHERE content = ? LIMIT 1",
                (content,))
            if cursor.fetchone():
                continue

            try:
                cursor.execute("""
                    INSERT INTO thought_nodes 
                    (id, content, node_type, confidence, domain, source_file, 
                     timestamp, last_accessed, decayed)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, 0)
                """, (node_id, content, node_type, confidence, domain,
                      source_file, now, now))
                created += 1
            except sqlite3.Error as e:
                logger.error(f"Failed to insert node: {e}")

        conn.commit()
        conn.close()
        return created

    def _state_path(self, name: str) -> str:
        return os.path.join(self._state_dir, f"{name}.json")

    def _save_state(self, name: str, state: Dict[str, Any]):
        """Persist extractor state to JSON."""
        if not state:
            return
        try:
            with open(self._state_path(name), 'w') as f:
                json.dump(state, f, indent=2)
        except (IOError, TypeError) as e:
            logger.error(f"Failed to save state for '{name}': {e}")

    def _load_state(self, name: str) -> Optional[Dict[str, Any]]:
        """Load persisted extractor state."""
        path = self._state_path(name)
        if not os.path.exists(path):
            return None
        try:
            with open(path, 'r') as f:
                return json.load(f)
        except (IOError, json.JSONDecodeError) as e:
            logger.error(f"Failed to load state for '{name}': {e}")
            return None
