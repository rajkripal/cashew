#!/usr/bin/env python3
"""
Tests for cashew extractor plugins.
"""

import json
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.extractors import ExtractorRegistry
from extractors import ObsidianExtractor, SessionExtractor, MarkdownDirExtractor
from extractors.utils import (
    parse_frontmatter, extract_wikilinks, load_ignore_patterns,
    should_ignore, split_into_paragraphs
)


class TestExtractorUtils(unittest.TestCase):
    """Test shared utility functions."""

    def test_parse_frontmatter(self):
        """Test YAML frontmatter parsing."""
        content = """---
title: Test Note
tags: [work, project]
aliases: [alias1, alias2]
---

This is the body content."""

        metadata, body = parse_frontmatter(content)

        self.assertEqual(metadata['title'], 'Test Note')
        self.assertEqual(metadata['tags'], ['work', 'project'])
        self.assertEqual(metadata['aliases'], ['alias1', 'alias2'])
        self.assertEqual(body.strip(), 'This is the body content.')

    def test_parse_frontmatter_no_yaml(self):
        """Test content without frontmatter."""
        content = "Just regular content"
        metadata, body = parse_frontmatter(content)

        self.assertEqual(metadata, {})
        self.assertEqual(body, content)

    def test_extract_wikilinks(self):
        """Test wikilink extraction."""
        content = "This links to [[Page One]] and [[Page Two|alias]] references."
        links = extract_wikilinks(content)

        self.assertEqual(set(links), {'Page One', 'Page Two'})

    def test_load_ignore_patterns(self):
        """Test loading gitignore-style patterns."""
        with tempfile.TemporaryDirectory() as temp_dir:
            ignore_file = Path(temp_dir) / ".test_ignore"
            ignore_file.write_text("*.tmp\n# comment\n.secret/\n\n")

            patterns = load_ignore_patterns(ignore_file)
            expected = ['*.tmp', '.secret/']
            self.assertEqual(patterns, expected)

    def test_should_ignore(self):
        """Test ignore pattern matching."""
        base_path = Path("/base")
        patterns = ["*.tmp", "secret/", "*.log"]

        # Should ignore
        self.assertTrue(should_ignore(
            Path("/base/file.tmp"), base_path, patterns))
        self.assertTrue(should_ignore(
            Path("/base/secret/file.txt"), base_path, patterns))

        # Should not ignore
        self.assertFalse(should_ignore(
            Path("/base/file.txt"), base_path, patterns))

    def test_split_into_paragraphs(self):
        """Test paragraph splitting."""
        content = """---
title: Test
---

# Header

This is paragraph one.

This is paragraph two with more content.

## Another Header

```code block```

Short.

This is a longer paragraph that should be included."""

        paragraphs = split_into_paragraphs(content, min_length=20)

        self.assertTrue(any("paragraph one" in p for p in paragraphs))
        self.assertTrue(any("paragraph two" in p for p in paragraphs))
        self.assertTrue(any("longer paragraph" in p for p in paragraphs))
        self.assertFalse(any("Short." in p for p in paragraphs))


class TestObsidianExtractor(unittest.TestCase):
    """Test ObsidianExtractor."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.vault_path = Path(self.temp_dir.name)
        self.db_path = self.vault_path / "test.db"
        self._setup_test_database()

    def tearDown(self):
        self.temp_dir.cleanup()

    def _setup_test_database(self):
        """Create test database with schema."""
        conn = sqlite3.connect(self.db_path)
        conn.execute('''
            CREATE TABLE thought_nodes (
                id TEXT PRIMARY KEY,
                content TEXT NOT NULL,
                node_type TEXT NOT NULL,
                domain TEXT,
                timestamp TEXT,
                access_count INTEGER DEFAULT 0,
                last_accessed TEXT,
                confidence REAL,
                source_file TEXT,
                decayed INTEGER DEFAULT 0,
                metadata TEXT,
                last_updated TEXT,
                mood_state TEXT,
                tags TEXT DEFAULT "",
                permanent INTEGER DEFAULT 0
            )
        ''')
        conn.execute('''
            CREATE TABLE derivation_edges (
                parent_id TEXT,
                child_id TEXT,
                weight REAL,
                reasoning TEXT,
                confidence REAL,
                timestamp TEXT,
                PRIMARY KEY (parent_id, child_id)
            )
        ''')
        conn.commit()
        conn.close()

    def test_basic_extraction(self):
        """Test basic markdown extraction."""
        # Create test files
        note1 = self.vault_path / "note1.md"
        note1.write_text("""---
title: First Note
tags: [work]
---

This is an important insight about the project.

Another paragraph with useful information.""")

        note2 = self.vault_path / "note2.md"
        note2.write_text("This note links to [[note1]] for reference.")

        extractor = ObsidianExtractor()

        # Test without model function (paragraph mode)
        nodes = extractor.extract(str(self.vault_path), None, str(self.db_path))

        self.assertGreater(len(nodes), 0)

        # Check node properties
        for node in nodes:
            self.assertIn('content', node)
            self.assertIn('type', node)
            self.assertIn('source_file', node)
            self.assertTrue(node['source_file'].startswith('extractor:obsidian:'))

    def test_wikilink_edges(self):
        """Test wikilink relationship creation."""
        from core.extractors import ExtractorRegistry

        # Create linked notes
        note1 = self.vault_path / "note1.md"
        note1.write_text("This is note one with sufficient content for extraction processing.")

        note2 = self.vault_path / "note2.md"
        note2.write_text("This note links to [[note1]] and [[note3]] with enough text for processing.")

        note3 = self.vault_path / "note3.md"
        note3.write_text("This links back to [[note2]] with adequate content for the extractor.")

        # Use the registry to properly insert nodes and create edges
        registry = ExtractorRegistry(data_dir=str(self.vault_path))
        registry.register(ObsidianExtractor())
        result = registry.run("obsidian", str(self.vault_path), None, str(self.db_path))

        self.assertGreater(result["nodes_created"], 0)

        # Check that edges were created
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM derivation_edges")
        edge_count = cursor.fetchone()[0]
        conn.close()

        self.assertGreater(edge_count, 0)

    def test_obsidianignore(self):
        """Test .obsidianignore functionality."""
        # Create ignore file
        ignore_file = self.vault_path / ".obsidianignore"
        ignore_file.write_text("secret.md\ntemplates/\n*.tmp")

        # Create files
        (self.vault_path / "normal.md").write_text("Normal content with sufficient length for extraction processing.")
        (self.vault_path / "secret.md").write_text("Secret content that should be ignored by the extractor.")
        templates_dir = self.vault_path / "templates"
        templates_dir.mkdir()
        (templates_dir / "template.md").write_text("Template content that should also be ignored by the extractor.")

        extractor = ObsidianExtractor()
        nodes = extractor.extract(str(self.vault_path), None, str(self.db_path))

        # Check that only normal.md was processed
        source_files = [node['source_file'] for node in nodes]
        self.assertTrue(any('normal.md' in sf for sf in source_files))
        self.assertFalse(any('secret.md' in sf for sf in source_files))
        self.assertFalse(any('template.md' in sf for sf in source_files))

    def test_checkpointing(self):
        """Test file modification time checkpointing."""
        note_file = self.vault_path / "note.md"
        note_file.write_text("Original content with enough text to pass minimum length requirements for extraction.")

        extractor = ObsidianExtractor()

        # First extraction
        nodes1 = extractor.extract(str(self.vault_path), None, str(self.db_path))
        self.assertGreater(len(nodes1), 0)

        # Second extraction without changes
        nodes2 = extractor.extract(str(self.vault_path), None, str(self.db_path))
        self.assertEqual(len(nodes2), 0)  # Should skip unchanged file

        # Modify file and extract again
        note_file.write_text("Modified content with different text that is also long enough for extraction processing.")
        nodes3 = extractor.extract(str(self.vault_path), None, str(self.db_path))
        self.assertGreater(len(nodes3), 0)  # Should process modified file

    def test_llm_extraction(self):
        """Test LLM-based extraction."""
        note_file = self.vault_path / "note.md"
        note_file.write_text("Complex content that needs LLM processing and contains sufficient text for extraction.")

        # Mock model function
        def mock_model_fn(prompt):
            return "Extracted insight 1 with sufficient length\nExtracted insight 2 with also enough content"

        extractor = ObsidianExtractor()
        nodes = extractor.extract(str(self.vault_path), mock_model_fn, str(self.db_path))

        self.assertEqual(len(nodes), 2)
        self.assertEqual(nodes[0]['content'], "Extracted insight 1 with sufficient length")
        self.assertEqual(nodes[1]['content'], "Extracted insight 2 with also enough content")

    def test_llm_debug_logging(self):
        """Raw LLM response is logged at debug level after extraction."""
        note_file = self.vault_path / "note.md"
        note_file.write_text("Content long enough to trigger LLM extraction path.")

        def mock_model_fn(prompt):
            return "raw obsidian response with enough length here"

        with patch("extractors.obsidian.logger") as mock_logger:
            ObsidianExtractor().extract(str(self.vault_path), mock_model_fn, str(self.db_path))
            debug_calls = " ".join(str(c) for c in mock_logger.debug.call_args_list)
            self.assertIn("raw obsidian response", debug_calls)


class TestSessionExtractor(unittest.TestCase):
    """Test SessionExtractor."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.session_dir = Path(self.temp_dir.name)
        self.db_path = self.session_dir / "test.db"
        self._setup_test_database()

    def tearDown(self):
        self.temp_dir.cleanup()

    def _setup_test_database(self):
        """Create test database with schema."""
        conn = sqlite3.connect(self.db_path)
        conn.execute('''
            CREATE TABLE thought_nodes (
                id TEXT PRIMARY KEY,
                content TEXT NOT NULL,
                node_type TEXT NOT NULL,
                domain TEXT,
                timestamp TEXT,
                access_count INTEGER DEFAULT 0,
                last_accessed TEXT,
                confidence REAL,
                source_file TEXT,
                decayed INTEGER DEFAULT 0,
                metadata TEXT,
                last_updated TEXT,
                mood_state TEXT,
                tags TEXT DEFAULT "",
                permanent INTEGER DEFAULT 0
            )
        ''')
        conn.execute('''
            CREATE TABLE derivation_edges (
                parent_id TEXT,
                child_id TEXT,
                weight REAL,
                reasoning TEXT,
                confidence REAL,
                timestamp TEXT,
                PRIMARY KEY (parent_id, child_id)
            )
        ''')
        conn.commit()
        conn.close()

    def test_basic_session_extraction(self):
        """Test basic session file processing."""
        session_file = self.session_dir / "test_session.jsonl"

        messages = [
            {"role": "user", "content": "Tell me about machine learning algorithms", "timestamp": "2024-01-01T10:00:00"},
            {"role": "assistant", "content": "Machine learning algorithms are computational methods that enable systems to learn and improve from data without being explicitly programmed for every task.", "timestamp": "2024-01-01T10:00:01"},
            {"role": "system", "content": "System message", "timestamp": "2024-01-01T10:00:02"},
            {"role": "user", "content": "Hi", "timestamp": "2024-01-01T10:00:03"}
        ]

        with open(session_file, 'w') as f:
            for msg in messages:
                f.write(json.dumps(msg) + "\n")

        extractor = SessionExtractor()
        nodes = extractor.extract(str(self.session_dir), None, str(self.db_path))

        # Should extract assistant message (long enough) but skip system and short user messages
        self.assertGreater(len(nodes), 0)

        # Check source tagging
        for node in nodes:
            self.assertTrue(node['source_file'].startswith('extractor:session:'))

    def test_incremental_extraction(self):
        """Test incremental session processing."""
        session_file = self.session_dir / "growing_session.jsonl"

        # Initial messages
        initial_messages = [
            {"role": "user", "content": "What is the capital of France? This is a longer message to meet the minimum length requirement for extraction processing and analysis.", "timestamp": "2024-01-01T10:00:00"},
            {"role": "assistant", "content": "The capital of France is Paris. Paris is known for its rich history, art, culture, and many famous landmarks like the Eiffel Tower and Louvre Museum.", "timestamp": "2024-01-01T10:00:01"}
        ]

        with open(session_file, 'w') as f:
            for msg in initial_messages:
                f.write(json.dumps(msg) + "\n")

        extractor = SessionExtractor()

        # First extraction
        nodes1 = extractor.extract(str(self.session_dir), None, str(self.db_path))
        self.assertGreater(len(nodes1), 0)

        # Add more messages
        new_messages = [
            {"role": "user", "content": "Tell me about the Eiffel Tower and its historical significance in French culture and tourism industry.", "timestamp": "2024-01-01T10:01:00"},
            {"role": "assistant", "content": "The Eiffel Tower is an iconic symbol of Paris and France, built for the 1889 World's Fair. It has become one of the most visited monuments in the world.", "timestamp": "2024-01-01T10:01:01"}
        ]

        with open(session_file, 'a') as f:
            for msg in new_messages:
                f.write(json.dumps(msg) + "\n")

        # Second extraction should only process new lines
        nodes2 = extractor.extract(str(self.session_dir), None, str(self.db_path))
        self.assertGreater(len(nodes2), 0)

        # Third extraction with no new content
        nodes3 = extractor.extract(str(self.session_dir), None, str(self.db_path))
        self.assertEqual(len(nodes3), 0)

    def test_llm_extraction(self):
        """Test LLM-based extraction for sessions."""
        session_file = self.session_dir / "conversation.jsonl"

        messages = [
            {"role": "user", "content": "I need to decide between React and Vue for my new project. What are the key differences?", "timestamp": "2024-01-01T10:00:00"},
            {"role": "assistant", "content": "Both React and Vue are excellent choices. React has a larger ecosystem and job market, while Vue has a gentler learning curve and more opinionated structure.", "timestamp": "2024-01-01T10:00:01"}
        ]

        with open(session_file, 'w') as f:
            for msg in messages:
                f.write(json.dumps(msg) + "\n")

        # Mock model function
        def mock_model_fn(prompt):
            return "User is deciding between React and Vue\nReact has larger ecosystem\nVue has gentler learning curve"

        extractor = SessionExtractor()
        nodes = extractor.extract(str(self.session_dir), mock_model_fn, str(self.db_path))

        self.assertEqual(len(nodes), 3)
        self.assertTrue(any("React and Vue" in node['content'] for node in nodes))

    def test_llm_debug_logging(self):
        """Raw LLM response is logged at debug level after extraction."""
        session_file = self.session_dir / "session.jsonl"
        messages = [
            {"role": "user", "content": "A" * 60},
            {"role": "assistant", "content": "B" * 60},
        ]
        with open(session_file, "w") as f:
            for m in messages:
                f.write(json.dumps(m) + "\n")

        def mock_model_fn(prompt):
            return "raw session response with enough length here"

        with patch("extractors.sessions.logger") as mock_logger:
            SessionExtractor().extract(str(self.session_dir), mock_model_fn, str(self.db_path))
            debug_calls = " ".join(str(c) for c in mock_logger.debug.call_args_list)
            self.assertIn("raw session response", debug_calls)

    def test_cashewignore(self):
        """Test .cashewignore skips matched session files."""
        def _write(name, text="x"):
            p = self.session_dir / name
            msg = {"role": "assistant", "content": text * 60}
            p.write_text(json.dumps(msg) + "\n", encoding="utf-8")

        _write("keep.jsonl")
        _write("skip.jsonl")
        _write("internal-a.jsonl")
        _write("internal-b.jsonl")
        (self.session_dir / ".cashewignore").write_text(
            "skip.jsonl\ninternal-*.jsonl\n", encoding="utf-8"
        )

        extractor = SessionExtractor()
        extractor.extract(str(self.session_dir), None, str(self.db_path))

        processed = {Path(k).name for k in extractor._processed}
        self.assertIn("keep.jsonl", processed)
        self.assertNotIn("skip.jsonl", processed)
        self.assertNotIn("internal-a.jsonl", processed)
        self.assertNotIn("internal-b.jsonl", processed)


class TestMarkdownDirExtractor(unittest.TestCase):
    """Test MarkdownDirExtractor."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.notes_dir = Path(self.temp_dir.name)
        self.db_path = self.notes_dir / "test.db"
        self._setup_test_database()

    def tearDown(self):
        self.temp_dir.cleanup()

    def _setup_test_database(self):
        """Create test database with schema."""
        conn = sqlite3.connect(self.db_path)
        conn.execute('''
            CREATE TABLE thought_nodes (
                id TEXT PRIMARY KEY,
                content TEXT NOT NULL,
                node_type TEXT NOT NULL,
                domain TEXT,
                timestamp TEXT,
                access_count INTEGER DEFAULT 0,
                last_accessed TEXT,
                confidence REAL,
                source_file TEXT,
                decayed INTEGER DEFAULT 0,
                metadata TEXT,
                last_updated TEXT,
                mood_state TEXT,
                tags TEXT DEFAULT "",
                permanent INTEGER DEFAULT 0
            )
        ''')
        conn.commit()
        conn.close()

    def test_directory_extraction(self):
        """Test recursive directory processing."""
        # Create directory structure
        (self.notes_dir / "project").mkdir()
        (self.notes_dir / "project" / "notes.md").write_text("Project notes content here.")
        (self.notes_dir / "readme.md").write_text("This is the readme file.")

        extractor = MarkdownDirExtractor()
        nodes = extractor.extract(str(self.notes_dir), None, str(self.db_path))

        self.assertGreater(len(nodes), 0)

        # Check domain detection
        domains = set(node['domain'] for node in nodes)
        self.assertIn('project', domains)

    def test_cashewignore(self):
        """Test .cashewignore functionality."""
        # Create ignore file
        ignore_file = self.notes_dir / ".cashewignore"
        ignore_file.write_text("draft.md\nprivate/\n*.bak")

        # Create files
        (self.notes_dir / "normal.md").write_text("Normal content with sufficient length for extraction processing.")
        (self.notes_dir / "draft.md").write_text("Draft content that should be ignored by the extractor.")
        private_dir = self.notes_dir / "private"
        private_dir.mkdir()
        (private_dir / "secret.md").write_text("Secret content in private directory that should be ignored.")

        extractor = MarkdownDirExtractor()
        nodes = extractor.extract(str(self.notes_dir), None, str(self.db_path))

        # Check that only normal.md was processed
        source_files = [node['source_file'] for node in nodes]
        self.assertTrue(any('normal.md' in sf for sf in source_files))
        self.assertFalse(any('draft.md' in sf for sf in source_files))
        self.assertFalse(any('secret.md' in sf for sf in source_files))

    def test_single_file_mode(self):
        """Test processing a single markdown file."""
        single_file = self.notes_dir / "single.md"
        single_file.write_text("This is a single file for testing.")

        extractor = MarkdownDirExtractor()
        nodes = extractor.extract(str(single_file), None, str(self.db_path))

        self.assertGreater(len(nodes), 0)
        self.assertTrue(all('single.md' in node['source_file'] for node in nodes))

    def test_llm_debug_logging(self):
        """Raw LLM response is logged at debug level after extraction."""
        md_file = self.notes_dir / "note.md"
        md_file.write_text("Content long enough to trigger LLM extraction path.")

        def mock_model_fn(prompt):
            return "raw markdown response with enough length here"

        with patch("extractors.markdown_dir.logger") as mock_logger:
            MarkdownDirExtractor().extract(str(self.notes_dir), mock_model_fn, str(self.db_path))
            debug_calls = " ".join(str(c) for c in mock_logger.debug.call_args_list)
            self.assertIn("raw markdown response", debug_calls)


class TestExtractorRegistry(unittest.TestCase):
    """Test the ExtractorRegistry system."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.data_dir = self.temp_dir.name
        self.registry = ExtractorRegistry(self.data_dir)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_extractor_registration(self):
        """Test registering and listing extractors."""
        obsidian_extractor = ObsidianExtractor()
        session_extractor = SessionExtractor()

        self.registry.register(obsidian_extractor)
        self.registry.register(session_extractor)

        extractors = self.registry.list_extractors()
        self.assertIn('obsidian', extractors)
        self.assertIn('sessions', extractors)

    def test_duplicate_registration(self):
        """Test that duplicate registration raises error."""
        extractor1 = ObsidianExtractor()
        extractor2 = ObsidianExtractor()

        self.registry.register(extractor1)

        with self.assertRaises(ValueError):
            self.registry.register(extractor2)

    def test_state_persistence(self):
        """Test that extractor state is persisted."""
        # Create a test file
        test_file = Path(self.data_dir) / "test.md"
        test_file.write_text("This is test content with sufficient length for extraction processing.")
        
        # Create test database
        db_path = Path(self.data_dir) / "test.db"
        conn = sqlite3.connect(db_path)
        conn.execute('''
            CREATE TABLE thought_nodes (
                id TEXT PRIMARY KEY,
                content TEXT NOT NULL,
                node_type TEXT NOT NULL,
                domain TEXT,
                timestamp TEXT,
                access_count INTEGER DEFAULT 0,
                last_accessed TEXT,
                confidence REAL,
                source_file TEXT,
                decayed INTEGER DEFAULT 0,
                metadata TEXT,
                last_updated TEXT,
                mood_state TEXT,
                tags TEXT DEFAULT "",
                permanent INTEGER DEFAULT 0
            )
        ''')
        conn.commit()
        conn.close()
        
        extractor = MarkdownDirExtractor()
        self.registry.register(extractor)
        
        # Run the extractor to save state
        result = self.registry.run("markdown", str(test_file.parent), None, str(db_path))
        
        # Create new registry and extractor
        new_registry = ExtractorRegistry(self.data_dir)
        new_extractor = MarkdownDirExtractor()
        new_registry.register(new_extractor)
        
        # State should be restored - check that the file is in processed state
        self.assertIn("test.md", new_extractor._processed)


class TestCLIIntegration(unittest.TestCase):
    """Test CLI ingest command integration."""

    def test_import_extractors(self):
        """Test that extractors can be imported."""
        from extractors import ObsidianExtractor, SessionExtractor, MarkdownDirExtractor

        self.assertEqual(ObsidianExtractor().name, 'obsidian')
        self.assertEqual(SessionExtractor().name, 'sessions')
        self.assertEqual(MarkdownDirExtractor().name, 'markdown')


if __name__ == '__main__':
    unittest.main()