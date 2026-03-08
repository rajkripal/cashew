#!/usr/bin/env python3
"""
Generic memory file extractor for Cashew thought-graph
"""

import sqlite3
import json
import re
import hashlib
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple, Optional

class GenericExtractor:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.workspace_path = Path("/Users/bunny/.openclaw/workspace")
        
    def generate_id(self, content: str) -> str:
        """Generate consistent ID for content"""
        return hashlib.sha256(content.encode()).hexdigest()[:12]
    
    def extract_thoughts_from_text(self, text: str, source_file: str) -> List[Dict]:
        """Extract thought-worthy content from text"""
        thoughts = []
        
        # Split into paragraphs and lines
        lines = text.split('\n')
        
        for line in lines:
            line = line.strip()
            if not line or len(line) < 20:  # Skip short lines
                continue
                
            # Remove markdown formatting
            cleaned = re.sub(r'[#*`_\[\]()]+', '', line).strip()
            if len(cleaned) < 20:
                continue
            
            # Determine node type and confidence based on content patterns
            node_type = "core_memory"
            confidence = 0.7
            mood_state = "neutral"
            
            # Pattern matching for different types of content
            if any(pattern in cleaned.lower() for pattern in [
                'i believe', 'my view', 'i think', 'conviction', 'philosophy'
            ]):
                node_type = "belief"
                confidence = 0.8
                
            elif any(pattern in cleaned.lower() for pattern in [
                'insight:', 'realization:', 'aha', 'breakthrough', 'understanding'
            ]):
                node_type = "derived"
                confidence = 0.85
                mood_state = "enlightened"
                
            elif any(pattern in cleaned.lower() for pattern in [
                'i remember', 'when i was', 'childhood', 'growing up', 'first time'
            ]):
                node_type = "core_memory"
                confidence = 0.9
                mood_state = "reflective"
                
            elif any(pattern in cleaned.lower() for pattern in [
                'god', 'prayer', 'church', 'bible', 'faith', 'christian', 'belief'
            ]):
                confidence = 0.85  # Religious content is often high-confidence for Raj
                
            elif any(pattern in cleaned.lower() for pattern in [
                'family', 'mom', 'dad', 'partner', 'marriage', 'love'
            ]):
                confidence = 0.8
                mood_state = "emotional"
                
            # Skip if it looks like metadata or boilerplate
            if any(skip in cleaned.lower() for skip in [
                'todo:', 'note:', 'reminder:', 'timestamp', 'date:'
            ]):
                continue
                
            thoughts.append({
                'content': cleaned,
                'node_type': node_type,
                'confidence': confidence,
                'mood_state': mood_state,
                'source_file': source_file
            })
        
        return thoughts
    
    def add_thoughts_to_db(self, thoughts: List[Dict]):
        """Add extracted thoughts to database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        added_count = 0
        
        for thought in thoughts:
            node_id = self.generate_id(thought['content'])
            
            # Check if node already exists
            cursor.execute("SELECT id FROM thought_nodes WHERE id = ?", (node_id,))
            if cursor.fetchone():
                continue  # Skip duplicates
            
            cursor.execute("""
                INSERT INTO thought_nodes 
                (id, content, node_type, timestamp, confidence, mood_state, metadata, source_file)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                node_id, 
                thought['content'], 
                thought['node_type'], 
                datetime.now().isoformat(),
                thought['confidence'], 
                thought['mood_state'], 
                json.dumps({}),
                thought['source_file']
            ))
            added_count += 1
        
        conn.commit()
        conn.close()
        
        return added_count
    
    def extract_from_file(self, file_path: Path) -> int:
        """Extract thoughts from a single file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
            
            thoughts = self.extract_thoughts_from_text(text, str(file_path.name))
            added_count = self.add_thoughts_to_db(thoughts)
            
            print(f"  {file_path.name}: {added_count} new thoughts extracted")
            return added_count
            
        except Exception as e:
            print(f"  ERROR processing {file_path.name}: {e}")
            return 0
    
    def extract_from_unprocessed_files(self):
        """Extract from all unprocessed memory files"""
        
        # Files that have already been processed (mentioned in the original request)
        processed_files = {
            '2026-03-05.md', '2026-03-06.md', '2026-03-07.md',
            '2026-02-23.md', 'e5-narrative-draft.md', 
            'raj-deep-context.md', 'raj-life-system.md'
        }
        
        memory_dir = self.workspace_path / "memory"
        total_added = 0
        
        print("📚 Extracting from unprocessed memory files...")
        
        # Get all markdown files in memory directory
        md_files = list(memory_dir.glob("**/*.md"))
        
        for file_path in sorted(md_files):
            if file_path.name in processed_files:
                print(f"  {file_path.name}: already processed, skipping")
                continue
                
            added = self.extract_from_file(file_path)
            total_added += added
        
        print(f"\n✅ Total new thoughts extracted: {total_added}")
        return total_added

def main():
    db_path = "/Users/bunny/.openclaw/workspace/cashew/data/graph.db"
    extractor = GenericExtractor(db_path)
    extractor.extract_from_unprocessed_files()

if __name__ == "__main__":
    main()