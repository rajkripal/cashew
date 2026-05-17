#!/usr/bin/env python3
"""
Shared utilities for cashew extractors.
"""

import fnmatch
import logging
import re
import yaml
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger("cashew.extractors.utils")


# LLMs tag statements at extraction time with one of the configured node
# types. The regex is built from core.config so adding a type in config
# automatically makes it valid here. Untagged lines fall back to a
# per-extractor classifier so older prompts still produce typed nodes.
from core.config import config as _cashew_config

_extraction_types = list(_cashew_config._node_type_map.keys())
TYPED_STATEMENT_RE = re.compile(
    r'^\[(' + '|'.join(re.escape(t) for t in _extraction_types) + r')\]\s+(.+)$',
    re.IGNORECASE,
)

# Description text is richer here than the one-liners in config because
# extractors emit format-strict prompts. Keep this hand-tuned, but the
# *set* of types is derived from config above. If config grows a new
# type, add a one-line description here in the same order.
_TYPE_DESCRIPTIONS = {
    "fact": "concrete verifiable context-independent info",
    "observation": "something noticed in context, may be situational",
    "insight": "non-obvious connection requiring reasoning",
    "decision": "choice made between alternatives",
    "commitment": "stated intention or planned action",
    "belief": "held opinion, not objectively verifiable",
}

_lines = [
    f"[{t}] {_TYPE_DESCRIPTIONS.get(t, _cashew_config._node_type_map.get(t, ''))}"
    for t in _extraction_types
]
TYPE_TAGGING_INSTRUCTION = (
    "Output one statement per line. Prefix each with a type tag.\n"
    "Types: " + " | ".join(_lines) + "\n"
    "When uncertain, use [observation].\n"
    "Output typed statements only, one per line."
)

# Shared "permanence test" framing. Each extractor appends its own trailing
# clause (e.g. "Skip pleasantries..." vs "Skip formatting...") since the
# noise it should reject is source-specific.
PERMANENCE_INSTRUCTION = (
    "Before emitting each statement, ask yourself: should this node exist "
    "in the graph forever? If you would not want future-you to read it, "
    "drop it. There is no hedging and no padding, either it is worth a "
    "permanent node or it is not."
)


def extract_typed_nodes_via_llm(
    prompt: str,
    model_fn: Callable[[str], str],
    *,
    domain: str,
    source_file: str,
    debug_label: str = "",
    referent_time: Optional[str] = None,
    min_content_length: int = 20,
    classifier: Optional[Callable[[str], str]] = None,
    default_type: str = "observation",
) -> Optional[List[Dict[str, Any]]]:
    """Run an LLM, parse typed statements, and shape them into node dicts.

    The pattern (model_fn -> parse_extraction_lines -> iterate typed statements
    -> filter by length -> build node dict) is identical across every
    extractor that supports an LLM path. This helper owns it.

    Returns the node list on success, or ``None`` on LLM failure so the
    caller can fall back to a deterministic path. Empty list means the LLM
    ran but produced no statements that passed the length filter — that is
    a legitimate result, not a failure.
    """
    try:
        response = model_fn(prompt)
    except Exception as e:
        logger.warning(
            f"LLM extraction failed{f' ({debug_label})' if debug_label else ''}: {e}"
        )
        return None

    if debug_label:
        logger.debug(f"LLM raw ({debug_label}):\n{response}\n---")
    statements = parse_extraction_lines(response)

    nodes_out: List[Dict[str, Any]] = []
    for raw in statements:
        node_type, content = parse_typed_statement(
            raw, fallback=classifier, default_type=default_type)
        if len(content) <= min_content_length:
            continue
        node: Dict[str, Any] = {
            "content": content,
            "type": node_type,
            "domain": domain,
            "source_file": source_file,
        }
        if referent_time is not None:
            node["referent_time"] = referent_time
        nodes_out.append(node)
    return nodes_out


def parse_typed_statement(
    line: str,
    fallback: Optional[Callable[[str], str]] = None,
    default_type: str = "observation",
) -> Tuple[str, str]:
    """Parse a possibly-typed statement line.

    Tagged lines look like ``[insight] some text`` (case-insensitive). When
    matched, the prefix is stripped and the lowercased type is returned. When
    not matched, ``fallback(line)`` is consulted (if given) — otherwise
    ``default_type`` is used and the line is returned unchanged.
    """
    m = TYPED_STATEMENT_RE.match(line)
    if m:
        return m.group(1).lower(), m.group(2).strip()
    if fallback is not None:
        return fallback(line), line
    return default_type, line


def parse_frontmatter(content: str) -> tuple[Dict[str, Any], str]:
    """Parse YAML frontmatter from markdown content.
    
    Returns:
        (metadata_dict, content_without_frontmatter)
    """
    metadata = {}
    
    if not content.startswith('---'):
        return metadata, content
    
    try:
        # Find the end of frontmatter
        end_match = re.search(r'\n---\s*\n', content)
        if not end_match:
            return metadata, content
        
        frontmatter_text = content[3:end_match.start()]
        body = content[end_match.end():]
        
        metadata = yaml.safe_load(frontmatter_text) or {}
        return metadata, body
    except (yaml.YAMLError, AttributeError):
        return metadata, content


def extract_wikilinks(content: str) -> List[str]:
    """Extract [[wikilink]] references from content.
    
    Returns:
        List of linked page names (without the [[ ]] brackets)
    """
    # Match [[link]] or [[link|alias]] patterns
    pattern = r'\[\[([^\]|]+)(?:\|[^\]]*)?\]\]'
    matches = re.findall(pattern, content)
    return [match.strip() for match in matches]


def load_ignore_patterns(ignore_file: Path) -> List[str]:
    """Load gitignore-style patterns from a file.
    
    Returns:
        List of patterns to ignore
    """
    if not ignore_file.exists():
        return []
    
    patterns = []
    try:
        with open(ignore_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    patterns.append(line)
    except IOError:
        pass
    
    return patterns


def should_ignore(file_path: Path, base_path: Path, patterns: List[str]) -> bool:
    """Check if a file should be ignored based on gitignore-style patterns.
    
    Args:
        file_path: The file to check
        base_path: The root directory for relative patterns
        patterns: List of gitignore-style patterns
        
    Returns:
        True if the file should be ignored
    """
    if not patterns:
        return False
    
    # Get relative path from base
    try:
        rel_path = file_path.relative_to(base_path)
    except ValueError:
        # file_path is not under base_path
        return False
    
    rel_str = str(rel_path)
    rel_posix = rel_str.replace('\\', '/')  # Normalize for cross-platform
    
    for pattern in patterns:
        # Normalize pattern separators
        norm_pattern = pattern.replace('\\', '/')
        
        # Match against relative path
        if fnmatch.fnmatch(rel_posix, norm_pattern):
            return True
        
        # Match against just the filename
        if fnmatch.fnmatch(file_path.name, norm_pattern):
            return True
        
        # Match directory patterns
        if norm_pattern.endswith('/'):
            dir_pattern = norm_pattern[:-1]
            for parent in file_path.parents:
                try:
                    parent_rel = parent.relative_to(base_path)
                    if fnmatch.fnmatch(str(parent_rel), dir_pattern):
                        return True
                except ValueError:
                    break
    
    return False


def split_into_paragraphs(content: str, min_length: int = 20) -> List[str]:
    """Split content into meaningful paragraphs for extraction.
    
    Args:
        content: Text content to split
        min_length: Minimum paragraph length to include
        
    Returns:
        List of paragraph strings
    """
    # Remove frontmatter first
    _, content = parse_frontmatter(content)
    
    # Split on double newlines
    paragraphs = content.split('\n\n')
    
    # Clean and filter paragraphs
    result = []
    for para in paragraphs:
        # Strip whitespace and normalize
        para = para.strip()
        
        # Skip empty, too short, or header-only paragraphs
        if (len(para) < min_length or 
            para.startswith('#') or 
            para.startswith('```') or
            para.startswith('---')):
            continue
        
        # Clean up formatting
        para = re.sub(r'\s+', ' ', para)  # Normalize whitespace
        para = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', para)  # Remove markdown links
        
        if len(para) >= min_length:
            result.append(para)
    
    return result


def parse_extraction_lines(response: str) -> List[str]:
    """Split an LLM extraction response into clean statement lines.

    Drops blank lines and markdown header/preamble lines (`#`, `##`, `---`).
    See issue #13: models intermittently prepend headers like "# Extracted
    Insights" despite prompt instructions to return statements only.
    """
    out: List[str] = []
    for raw in response.split("\n"):
        line = raw.strip()
        if not line:
            continue
        if line.startswith("#") or line.startswith("---"):
            continue
        out.append(line)
    return out


def detect_domain_from_path(file_path: Path, base_path: Path) -> str:
    """Detect domain from file path structure.
    
    Uses the first directory level as domain, falls back to 'default'.
    """
    try:
        rel_path = file_path.relative_to(base_path)
        parts = rel_path.parts
        
        if len(parts) > 1:
            # Use first directory level as domain
            return parts[0]
        else:
            # File is at root level
            return "default"
    except ValueError:
        return "default"