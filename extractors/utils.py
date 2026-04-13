#!/usr/bin/env python3
"""
Shared utilities for cashew extractors.
"""

import fnmatch
import re
import yaml
from pathlib import Path
from typing import Any, Dict, List, Optional


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