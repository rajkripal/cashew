#!/usr/bin/env python3
"""
Cashew Context CLI - Manual testing interface for OpenClaw integration
"""

import sys
import argparse
import json
import os
import time
import logging
from pathlib import Path

logger = logging.getLogger("cashew")

# Add the parent directory to the path so we can import cashew modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from integration.openclaw import generate_session_context, extract_from_conversation, run_think_cycle, run_tension_detection
from integration.complete_integration import (
    generate_complete_session_context, extract_from_conversation_complete,
    run_complete_think_cycle, run_complete_sleep_cycle, migrate_to_complete_coverage,
    explain_complete_system, get_complete_system_stats
)
from core.hotspots import create_hotspot, update_hotspot, list_hotspots, get_hotspot


def cmd_context(args):
    """Generate context for current session"""
    hints = args.hints if args.hints else None
    print(f"🔍 Generating context with hints: {hints}")
    print()
    
    t0 = time.time()
    context = generate_session_context(args.db, hints)
    elapsed = time.time() - t0
    
    if context:
        print(context)
        print()
        print("✅ Context generated successfully")
        if args.debug:
            print(f"⏱  Elapsed: {elapsed:.2f}s", file=sys.stderr)
            print(f"📏 Context length: {len(context)} chars", file=sys.stderr)
    else:
        print("❌ No context generated (empty result)")


def cmd_extract(args):
    """Extract from a conversation file"""
    if not args.input:
        print("❌ Error: --input file required for extract command")
        return 1
    
    if not os.path.exists(args.input):
        print(f"❌ Error: File not found: {args.input}")
        return 1
    
    print(f"📖 Reading conversation from: {args.input}")
    
    with open(args.input, 'r') as f:
        conversation_text = f.read()
    
    if not conversation_text.strip():
        print("❌ Error: Empty conversation file")
        return 1
    
    print(f"📝 Conversation length: {len(conversation_text)} characters")
    print("🧠 Extracting insights...")
    print()
    
    t0 = time.time()
    result = extract_from_conversation(args.db, conversation_text, args.session_id)
    elapsed = time.time() - t0
    
    print(json.dumps(result, indent=2))
    
    if result.get("success"):
        print()
        print("✅ Extraction completed successfully")
        print(f"   New nodes: {result['new_nodes']}")
        print(f"   New edges: {result['new_edges']}")
        if args.debug:
            print(f"⏱  Elapsed: {elapsed:.2f}s", file=sys.stderr)
    else:
        print()
        print(f"❌ Extraction failed")
        if args.debug:
            print(f"⏱  Elapsed: {elapsed:.2f}s", file=sys.stderr)
            print(f"🔍 Result: {json.dumps(result, indent=2)}", file=sys.stderr)
        return 1


def cmd_think(args):
    """Run a think cycle"""
    domain = args.domain if args.domain else None
    mode = getattr(args, 'mode', 'general')
    
    if mode == "tension":
        print("⚡ Running tension detection...")
        if domain:
            print(f"   Focused on domain: {domain}")
        print()
        t0 = time.time()
        result = run_tension_detection(args.db, domain)
        elapsed = time.time() - t0
    else:
        if domain:
            print(f"🤔 Running think cycle focused on domain: {domain}")
        else:
            print("🤔 Running general think cycle...")
        print()
        t0 = time.time()
        result = run_think_cycle(args.db, domain)
        elapsed = time.time() - t0
    
    print(json.dumps(result, indent=2))
    
    if result.get("success"):
        print()
        print("✅ Think cycle completed successfully")
        print(f"   Cluster: {result['cluster_topic']}")
        print(f"   New insights: {result['new_nodes']}")
        print(f"   New connections: {result['new_edges']}")
        if args.debug:
            print(f"⏱  Elapsed: {elapsed:.2f}s", file=sys.stderr)
    else:
        print()
        print("❌ Think cycle failed")
        if args.debug:
            print(f"⏱  Elapsed: {elapsed:.2f}s", file=sys.stderr)
        return 1


def cmd_stats(args):
    """Show graph statistics"""
    try:
        import sqlite3
        
        conn = sqlite3.connect(args.db)
        cursor = conn.cursor()
        
        # Count nodes by type
        cursor.execute("""
            SELECT node_type, COUNT(*) 
            FROM thought_nodes 
            WHERE decayed IS NULL OR decayed = 0
            GROUP BY node_type 
            ORDER BY COUNT(*) DESC
        """)
        node_types = cursor.fetchall()
        
        # Count total nodes
        cursor.execute("SELECT COUNT(*) FROM thought_nodes WHERE decayed IS NULL OR decayed = 0")
        total_nodes = cursor.fetchone()[0]
        
        # Count edges
        cursor.execute("SELECT COUNT(*) FROM derivation_edges")
        total_edges = cursor.fetchone()[0]
        
        # Count embeddings
        cursor.execute("""
            SELECT COUNT(*) FROM embeddings e
            JOIN thought_nodes tn ON e.node_id = tn.id
            WHERE tn.decayed IS NULL OR tn.decayed = 0
        """)
        embedded_nodes = cursor.fetchone()[0]
        
        # Recent activity
        cursor.execute("""
            SELECT COUNT(*) FROM thought_nodes 
            WHERE (decayed IS NULL OR decayed = 0)
            AND timestamp > datetime('now', '-7 days')
        """)
        recent_nodes = cursor.fetchone()[0]
        
        conn.close()
        
        print("📊 Cashew Graph Statistics")
        print("=" * 40)
        print(f"Total nodes: {total_nodes}")
        print(f"Total edges: {total_edges}")
        print(f"Embedded nodes: {embedded_nodes}")
        print(f"Recent nodes (7 days): {recent_nodes}")
        print(f"Embedding coverage: {embedded_nodes/total_nodes*100:.1f}%" if total_nodes > 0 else "Embedding coverage: 0%")
        print()
        print("Node types:")
        for node_type, count in node_types:
            print(f"  {node_type}: {count}")
        
    except Exception as e:
        print(f"❌ Error getting stats: {e}")
        return 1


def cmd_sleep(args):
    """Run the sleep/consolidation protocol."""
    import time as _time
    start = _time.time()
    print("😴 Running sleep protocol...")
    
    try:
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__))))
        from core.sleep import SleepProtocol
        
        protocol = SleepProtocol(args.db)
        result = protocol.run_sleep_cycle()
        elapsed = _time.time() - start
        
        print(f"\n✅ Sleep protocol completed in {elapsed:.1f}s")
        if isinstance(result, dict):
            for k, v in result.items():
                print(f"  {k}: {v}")
        else:
            print(f"  Result: {result}")
        
    except Exception as e:
        print(f"❌ Sleep protocol error: {e}")
        if args.debug:
            import traceback
            traceback.print_exc()
        return 1


def cmd_complete_context(args):
    """Generate context with complete coverage retrieval"""
    hints = args.hints if args.hints else None
    method = getattr(args, 'method', 'dfs')
    print(f"🔍 Generating complete context with method: {method}")
    if hints:
        print(f"   Hints: {hints}")
    print()
    
    t0 = time.time()
    try:
        context = generate_complete_session_context(
            args.db, hints, args.domain, method, args.top_k
        )
        elapsed = time.time() - t0
        
        if context:
            print(context)
            print()
            print("✅ Complete context generated successfully")
            if args.debug:
                print(f"⏱  Elapsed: {elapsed:.2f}s", file=sys.stderr)
                print(f"📏 Context length: {len(context)} chars", file=sys.stderr)
        else:
            print("❌ No context generated (empty result)")
    except Exception as e:
        print(f"❌ Error: {e}")
        if args.debug:
            import traceback
            traceback.print_exc()
        return 1


def cmd_complete_extract(args):
    """Extract with placement-aware assignment"""
    if not args.input:
        print("❌ Error: --input file required for extract command")
        return 1
    
    if not os.path.exists(args.input):
        print(f"❌ Error: File not found: {args.input}")
        return 1
    
    print(f"📖 Reading conversation from: {args.input}")
    
    with open(args.input, 'r') as f:
        conversation_text = f.read()
    
    if not conversation_text.strip():
        print("❌ Error: Empty conversation file")
        return 1
    
    print(f"📝 Conversation length: {len(conversation_text)} characters")
    print("🧠 Extracting with placement-aware assignment...")
    print()
    
    t0 = time.time()
    try:
        result = extract_from_conversation_complete(
            args.db, conversation_text, args.session_id or "complete_session"
        )
        elapsed = time.time() - t0
        
        print(json.dumps(result, indent=2))
        
        if result.get("success"):
            print()
            print("✅ Placement-aware extraction completed successfully")
            print(f"   New nodes: {len(result.get('new_nodes', []))}")
            print(f"   Placements: {len(result.get('placements', []))}")
            if args.debug:
                print(f"⏱  Elapsed: {elapsed:.2f}s", file=sys.stderr)
        else:
            print()
            print(f"❌ Extraction failed: {result.get('error', 'Unknown error')}")
            return 1
    except Exception as e:
        print(f"❌ Error: {e}")
        if args.debug:
            import traceback
            traceback.print_exc()
        return 1


def cmd_complete_think(args):
    """Run think cycle with placement-aware assignment"""
    domain = args.domain if args.domain else None
    
    print("⚡ Running complete think cycle with placement...")
    if domain:
        print(f"   Focused on domain: {domain}")
    print()
    
    t0 = time.time()
    try:
        result = run_complete_think_cycle(args.db, domain)
        elapsed = time.time() - t0
        
        print(json.dumps(result, indent=2))
        
        if result.get("success"):
            print()
            print("✅ Complete think cycle completed successfully")
            print(f"   New nodes: {len(result.get('new_nodes', []))}")
            if args.debug:
                print(f"⏱  Elapsed: {elapsed:.2f}s", file=sys.stderr)
        else:
            print()
            print(f"❌ Think cycle failed: {result.get('error', 'Unknown error')}")
            return 1
    except Exception as e:
        print(f"❌ Error: {e}")
        if args.debug:
            import traceback
            traceback.print_exc()
        return 1


def cmd_complete_sleep(args):
    """Run complete sleep cycle with hierarchy evolution"""
    enable_evolution = not getattr(args, 'no_evolution', False)
    
    print("😴 Running complete sleep cycle...")
    print(f"   Hierarchy evolution: {'enabled' if enable_evolution else 'disabled'}")
    print()
    
    t0 = time.time()
    try:
        result = run_complete_sleep_cycle(args.db, enable_evolution)
        elapsed = time.time() - t0
        
        print(json.dumps(result, indent=2))
        
        print()
        print(f"✅ Complete sleep cycle completed in {elapsed:.1f}s")
        print(f"   Total actions: {result.get('total_actions', 0)}")
        print(f"   Coverage: {result.get('coverage_verification', {}).get('coverage_percentage', 'Unknown')}%")
    except Exception as e:
        print(f"❌ Error: {e}")
        if args.debug:
            import traceback
            traceback.print_exc()
        return 1


def cmd_migrate(args):
    """Migrate to complete coverage system"""
    dry_run = getattr(args, 'dry_run', False)
    
    print(f"🔄 Migrating to complete coverage system (dry_run={dry_run})...")
    print()
    
    t0 = time.time()
    try:
        result = migrate_to_complete_coverage(args.db, dry_run)
        elapsed = time.time() - t0
        
        print(json.dumps(result, indent=2))
        
        print()
        if result.get("migration_needed"):
            print(f"✅ Migration completed in {elapsed:.1f}s")
            print(f"   Final coverage: {result.get('final_coverage', {}).get('coverage_percentage', 'Unknown')}%")
        else:
            print("✅ No migration needed - system already has complete coverage")
    except Exception as e:
        print(f"❌ Error: {e}")
        if args.debug:
            import traceback
            traceback.print_exc()
        return 1


def cmd_explain(args):
    """Explain the complete coverage system"""
    query = getattr(args, 'query', 'test query for explanation')
    
    print("📊 Explaining complete coverage system...")
    print()
    
    t0 = time.time()
    try:
        result = explain_complete_system(args.db, query)
        elapsed = time.time() - t0
        
        print(json.dumps(result, indent=2))
        
        print()
        print(f"✅ System explanation completed in {elapsed:.1f}s")
    except Exception as e:
        print(f"❌ Error: {e}")
        if args.debug:
            import traceback
            traceback.print_exc()
        return 1


def cmd_init(args):
    """Initialize a new cashew database"""
    import sqlite3
    from pathlib import Path
    
    db_path = Path(args.db)
    data_dir = db_path.parent
    
    print(f"🗂  Initializing cashew at: {db_path}")
    
    # Create data directory if it doesn't exist
    data_dir.mkdir(parents=True, exist_ok=True)
    print(f"✅ Created data directory: {data_dir}")
    
    if db_path.exists():
        print(f"⚠️  Database already exists: {db_path}")
        response = input("Do you want to overwrite it? [y/N]: ")
        if response.lower() != 'y':
            print("❌ Cancelled")
            return 1
    
    # Create the database with schema
    try:
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()
        
        # Create tables (from core schema)
        cursor.execute('''
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
                mood_state TEXT
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE derivation_edges (
                parent_id TEXT,
                child_id TEXT,
                edge_type TEXT,
                relation TEXT,
                weight REAL,
                reasoning TEXT,
                confidence REAL,
                timestamp TEXT,
                PRIMARY KEY (parent_id, child_id, edge_type),
                FOREIGN KEY (parent_id) REFERENCES thought_nodes(id),
                FOREIGN KEY (child_id) REFERENCES thought_nodes(id)
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE embeddings (
                node_id TEXT PRIMARY KEY,
                vector BLOB NOT NULL,
                model TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                FOREIGN KEY (node_id) REFERENCES thought_nodes(id)
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE hotspots (
                id TEXT PRIMARY KEY,
                content TEXT NOT NULL,
                status TEXT,
                domain TEXT,
                file_pointers TEXT,
                cluster_node_ids TEXT,
                tags TEXT,
                created TEXT,
                last_updated TEXT
            )
        ''')
        
        # Create indexes
        cursor.execute('CREATE INDEX idx_nodes_timestamp ON thought_nodes(timestamp)')
        cursor.execute('CREATE INDEX idx_nodes_domain ON thought_nodes(domain)')
        cursor.execute('CREATE INDEX idx_nodes_type ON thought_nodes(node_type)')
        cursor.execute('CREATE INDEX idx_edges_parent ON derivation_edges(parent_id)')
        cursor.execute('CREATE INDEX idx_edges_child ON derivation_edges(child_id)')
        
        conn.commit()
        conn.close()
        
        print("✅ Database initialized successfully")
        print(f"   Schema: thought_nodes, derivation_edges, embeddings, hotspots")
        print(f"   Indexes: optimized for retrieval")
        print()
        print("🚀 Ready to use! Try:")
        print(f"   cashew stats --db {db_path}")
        print(f"   cashew context 'your topic' --db {db_path}")
        
    except Exception as e:
        print(f"❌ Error initializing database: {e}")
        if args.debug:
            import traceback
            traceback.print_exc()
        return 1


def _migrate_extract_file(db_path: str, content: str, filename: str, session_id: str) -> dict:
    """
    Extract knowledge from a document file for migration.
    Uses a migration-specific prompt that produces semantic summaries
    instead of line-level fragments.
    """
    import json
    import sqlite3
    import hashlib
    from datetime import datetime, timezone
    from core.embeddings import ensure_schema, embed_nodes
    
    ensure_schema(db_path)
    
    # Try LLM extraction with migration-specific prompt
    try:
        import anthropic
        import os
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            return _migrate_extract_heuristic(db_path, content, filename, session_id)
        
        client = anthropic.Anthropic(api_key=api_key)
        
        prompt = f"""You are migrating a personal knowledge document into a thought graph. Extract the KEY ENTITIES and INSIGHTS as self-contained statements.

CRITICAL RULES:
1. Each statement must be SELF-CONTAINED — understandable without reading the source document
2. Include WHAT something is, not just that it exists. "DisClawd is a multi-agent collaboration framework for Discord bots" not just "DisClawd lives at /home/disclawd"
3. Capture projects with their PURPOSE and STATUS, not just file paths
4. Capture decisions with their REASONING, not just the choice
5. Merge related fragments into single comprehensive statements
6. Aim for 5-15 high-quality nodes per document, not 50 fragments

Types: belief, insight, decision, observation, fact

Respond with ONLY a JSON array:
[{{"content": "self-contained statement", "type": "belief|insight|decision|observation|fact", "confidence": 0.8}}]

Document ({filename}):
{content[:8000]}"""
        
        message = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=2000,
            temperature=0.3,
            messages=[{"role": "user", "content": prompt}]
        )
        response = message.content[0].text
        
        # Parse JSON
        cleaned = response.strip()
        if cleaned.startswith('```'):
            lines = cleaned.split('\n')
            lines = [l for l in lines if not l.strip().startswith('```')]
            cleaned = '\n'.join(lines).strip()
        start = cleaned.find('[')
        end = cleaned.rfind(']')
        if start == -1 or end == -1:
            return _migrate_extract_heuristic(db_path, content, filename, session_id)
        
        extractions = json.loads(cleaned[start:end+1])
        
    except Exception as e:
        print(f"   ⚠️  LLM extraction failed ({e}), using heuristic")
        return _migrate_extract_heuristic(db_path, content, filename, session_id)
    
    # Insert nodes
    now = datetime.now(timezone.utc).isoformat()
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    new_nodes = []
    
    for item in extractions:
        node_content = item.get("content", "").strip()
        if not node_content or len(node_content) < 20:
            continue
        node_type = item.get("type", "observation")
        confidence = item.get("confidence", 0.7)
        node_id = hashlib.sha256(f"{node_content}:{now}".encode()).hexdigest()[:12]
        
        try:
            cursor.execute("""
                INSERT OR IGNORE INTO thought_nodes 
                (id, content, node_type, timestamp, confidence, source_file, access_count, metadata, domain)
                VALUES (?, ?, ?, ?, ?, ?, 0, '{}', 'default')
            """, (node_id, node_content, node_type, now, confidence, f"migration:{filename}"))
            new_nodes.append(node_id)
        except Exception:
            continue
    
    conn.commit()
    conn.close()
    
    # Generate embeddings for new nodes
    try:
        embed_nodes(db_path)
    except Exception:
        pass
    
    return {
        "success": True,
        "new_nodes": new_nodes,
        "placements": []
    }


def _migrate_extract_heuristic(db_path: str, content: str, filename: str, session_id: str) -> dict:
    """Fallback heuristic extraction for migration when no LLM available"""
    import sqlite3
    import hashlib
    from datetime import datetime, timezone
    from core.embeddings import ensure_schema, embed_nodes
    
    ensure_schema(db_path)
    
    now = datetime.now(timezone.utc).isoformat()
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    new_nodes = []
    
    # Extract paragraphs (not lines) as semantic units
    paragraphs = [p.strip() for p in content.split('\n\n') if len(p.strip()) > 50]
    
    for para in paragraphs[:20]:  # Cap at 20 per file
        # Skip markdown headers, code blocks, and list fragments
        if para.startswith('```') or para.startswith('---') or para.startswith('|'):
            continue
        # Clean up but keep as paragraph
        clean = ' '.join(para.split())[:500]
        node_id = hashlib.sha256(f"{clean}:{now}".encode()).hexdigest()[:12]
        
        try:
            cursor.execute("""
                INSERT OR IGNORE INTO thought_nodes 
                (id, content, node_type, timestamp, confidence, source_file, access_count, metadata, domain)
                VALUES (?, ?, 'observation', ?, 0.5, ?, 0, '{}', 'default')
            """, (node_id, clean, now, f"migration:{filename}"))
            new_nodes.append(node_id)
        except Exception:
            continue
    
    conn.commit()
    conn.close()
    
    try:
        embed_nodes(db_path)
    except Exception:
        pass
    
    return {
        "success": True,
        "new_nodes": new_nodes,
        "placements": []
    }


def cmd_migrate_files(args):
    """Migrate markdown files to cashew database"""
    import os
    from pathlib import Path
    import glob
    
    if not args.dir:
        print("❌ Error: --dir required for migrate command")
        return 1
    
    source_dir = Path(args.dir)
    if not source_dir.exists():
        print(f"❌ Error: Directory not found: {source_dir}")
        return 1
    
    dry_run = getattr(args, 'dry_run', False)
    
    print(f"📂 Migrating files from: {source_dir}")
    print(f"🗄  Target database: {args.db}")
    print(f"🔍 Dry run: {dry_run}")
    print()
    
    # Find markdown files (use set to avoid duplicates from overlapping globs)
    md_files = sorted(set(source_dir.glob("**/*.md")))
    
    if not md_files:
        print("❌ No markdown files found")
        return 1
    
    print(f"📋 Found {len(md_files)} markdown files:")
    for f in md_files[:5]:  # Show first 5
        print(f"   {f}")
    if len(md_files) > 5:
        print(f"   ... and {len(md_files) - 5} more")
    print()
    
    if dry_run:
        print("🔍 DRY RUN - No changes will be made")
        total_content = 0
        for md_file in md_files:
            try:
                content = md_file.read_text(encoding='utf-8')
                total_content += len(content)
                print(f"   {md_file.name}: {len(content)} chars")
            except Exception as e:
                print(f"   {md_file.name}: Error reading - {e}")
        
        print()
        print(f"📊 Total content: {total_content:,} characters")
        print("   Would extract key statements and build graph")
        print("   Would run semantic similarity wiring")
        print("   Would run consolidation sleep cycle")
        return 0
    
    # Check if database exists
    if not os.path.exists(args.db):
        print("❌ Error: Database not found. Run `cashew init` first.")
        return 1
    
    # Extract from each file using migration-aware extraction
    extracted_count = 0
    errors = 0
    
    for md_file in md_files:
        try:
            print(f"📖 Processing: {md_file.name}")
            content = md_file.read_text(encoding='utf-8')
            
            if len(content.strip()) < 100:
                print(f"   ⚠️  Skipping (too short)")
                continue
            
            # Use migration-specific extraction (semantic summaries, not line fragments)
            result = _migrate_extract_file(
                args.db, 
                content, 
                md_file.name,
                f"migration_{md_file.stem}"
            )
            
            if result.get("success"):
                new_nodes = len(result.get('new_nodes', []))
                placements = len(result.get('placements', []))
                extracted_count += new_nodes
                print(f"   ✅ Extracted {new_nodes} nodes, {placements} placements")
            else:
                errors += 1
                print(f"   ❌ Extraction failed: {result.get('error', 'Unknown')}")
                
        except Exception as e:
            errors += 1
            print(f"   ❌ Error: {e}")
    
    if extracted_count > 0:
        print()
        print(f"🧠 Running consolidation cycle...")
        
        try:
            # Run a sleep cycle to consolidate and connect the extracted knowledge
            result = run_complete_sleep_cycle(args.db)
            if result.get("error"):
                print(f"⚠️  Consolidation had issues: {result['error']}")
                print("   Graph is usable but hierarchy may be incomplete.")
                print("   Run `cashew sleep` later to retry consolidation.")
            else:
                coverage = result.get('coverage_verification', {}).get('coverage_percentage', 'Unknown')
                actions = result.get('total_actions', 0)
                print(f"✅ Consolidation completed ({actions} actions)")
                print(f"   Final coverage: {coverage}%")
        except Exception as e:
            print(f"⚠️  Consolidation skipped: {e}")
            print("   Graph is usable but hierarchy may be incomplete.")
            print("   Run `cashew sleep` later to retry consolidation.")
    
    print()
    print("📊 MIGRATION COMPLETE")
    print(f"   Files processed: {len(md_files)}")
    print(f"   Nodes extracted: {extracted_count}")
    print(f"   Errors: {errors}")
    if errors == 0:
        print("   🎉 All files migrated successfully!")


def cmd_system_stats(args):
    """Show complete system statistics"""
    print("📈 Gathering complete system statistics...")
    print()
    
    t0 = time.time()
    try:
        result = get_complete_system_stats(args.db)
        elapsed = time.time() - t0
        
        print(json.dumps(result, indent=2))
        
        print()
        print(f"✅ Statistics gathered in {elapsed:.1f}s")
        
        # Show summary
        coverage = result.get("coverage", {})
        hierarchy = result.get("hierarchy_statistics", {})
        health = result.get("system_health", {})
        
        print("\n📊 SUMMARY:")
        print(f"   Coverage: {coverage.get('coverage_percentage', 'Unknown')}%")
        print(f"   Total nodes: {result.get('node_statistics', {}).get('total_nodes', 'Unknown')}")
        print(f"   Total hotspots: {hierarchy.get('total_hotspots', 'Unknown')}")
        print(f"   Hierarchy depth: {hierarchy.get('hierarchy_depth', 'Unknown')}")
        print(f"   Complete coverage: {health.get('has_complete_coverage', 'Unknown')}")
        print(f"   Emergent domains: {health.get('emergent_domains', 'Unknown')}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        if args.debug:
            import traceback
            traceback.print_exc()
        return 1


def cmd_hotspot(args):
    """Manage hotspot nodes"""
    action = args.hotspot_action
    
    if action == "create":
        if not args.content:
            print("❌ --content required for create")
            return 1
        
        # Parse file pointers from --files "label:path,label:path"
        file_pointers = {}
        if args.files:
            for pair in args.files.split(","):
                if ":" in pair:
                    label, path = pair.split(":", 1)
                    file_pointers[label.strip()] = path.strip()
        
        # Parse cluster IDs
        cluster_ids = args.cluster.split(",") if args.cluster else []
        tags = args.tags.split(",") if args.tags else []
        
        hotspot_id = create_hotspot(
            db_path=args.db,
            content=args.content,
            status=args.status or "active",
            file_pointers=file_pointers,
            cluster_node_ids=cluster_ids,
            domain=args.domain or "bunny",
            tags=tags
        )
        print(f"✅ Created hotspot: {hotspot_id}")
        print(f"   Content: {args.content[:80]}...")
        print(f"   Status: {args.status or 'active'}")
        print(f"   Files: {file_pointers}")
        print(f"   Cluster: {len(cluster_ids)} nodes")
    
    elif action == "update":
        if not args.id:
            print("❌ --id required for update")
            return 1
        
        file_pointers = None
        if args.files:
            file_pointers = {}
            for pair in args.files.split(","):
                if ":" in pair:
                    label, path = pair.split(":", 1)
                    file_pointers[label.strip()] = path.strip()
        
        add_ids = args.cluster.split(",") if args.cluster else None
        
        success = update_hotspot(
            db_path=args.db,
            hotspot_id=args.id,
            content=args.content,
            status=args.status,
            file_pointers=file_pointers,
            add_cluster_ids=add_ids
        )
        if success:
            print(f"✅ Updated hotspot: {args.id}")
        else:
            print(f"❌ Failed to update hotspot: {args.id}")
            return 1
    
    elif action == "list":
        hotspots = list_hotspots(args.db, args.domain)
        if not hotspots:
            print("No hotspots found.")
            return
        
        print(f"📍 {len(hotspots)} Hotspot(s)")
        print("=" * 60)
        for h in hotspots:
            print(f"\n🔵 [{h['id']}] {h['content'][:80]}")
            print(f"   Status: {h['status']} | Domain: {h['domain']} | Cluster: {h['cluster_size']} nodes")
            if h['file_pointers']:
                for label, path in h['file_pointers'].items():
                    print(f"   📄 {label}: {path}")
            if h['tags']:
                print(f"   🏷  Tags: {', '.join(h['tags'])}")
            print(f"   Updated: {h['last_updated']}")
    
    elif action == "show":
        if not args.id:
            print("❌ --id required for show")
            return 1
        
        h = get_hotspot(args.db, args.id)
        if not h:
            print(f"❌ Hotspot not found: {args.id}")
            return 1
        
        print(f"📍 Hotspot: {h['id']}")
        print(f"   Content: {h['content']}")
        print(f"   Status: {h['status']}")
        print(f"   Domain: {h['domain']}")
        print(f"   Updated: {h['last_updated']}")
        if h['file_pointers']:
            print(f"   Files:")
            for label, path in h['file_pointers'].items():
                print(f"     📄 {label}: {path}")
        if h['tags']:
            print(f"   Tags: {', '.join(h['tags'])}")
        if h['cluster']:
            print(f"   Cluster ({len(h['cluster'])} nodes):")
            for node in h['cluster']:
                print(f"     - [{node['type']}] {node['content'][:60]}...")


def _preprocess_db_flag(argv):
    """Move --db flag before subcommand so argparse can find it as a top-level arg"""
    result = list(argv)
    i = 0
    while i < len(result):
        if result[i] == '--db' and i + 1 < len(result):
            db_flag = result.pop(i)
            db_val = result.pop(i)
            result.insert(0, db_val)
            result.insert(0, db_flag)
            break
        elif result[i].startswith('--db='):
            db_flag = result.pop(i)
            result.insert(0, db_flag)
            break
        i += 1
    return result


def main():
    parser = argparse.ArgumentParser(description="Cashew Context CLI")
    # Import here to avoid circular imports
    import sys
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__))))
    from core.config import get_db_path
    
    parser.add_argument("--db", default=get_db_path(), 
                       help="Database path (default: ./data/graph.db, or CASHEW_DB env var)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--debug", action="store_true", help="Debug output (timing, diagnostics to stderr)")
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Context command
    context_parser = subparsers.add_parser("context", help="Generate context for current session")
    context_parser.add_argument("--hints", nargs="*", 
                                help="Topic hints (e.g., 'work promotion manager')")
    context_parser.set_defaults(func=cmd_context)
    
    # Extract command  
    extract_parser = subparsers.add_parser("extract", help="Extract from a conversation file")
    extract_parser.add_argument("--input", required=True, help="Input conversation file")
    extract_parser.add_argument("--session-id", help="Optional session ID")
    extract_parser.set_defaults(func=cmd_extract)
    
    # Think command
    think_parser = subparsers.add_parser("think", help="Run a think cycle")
    think_parser.add_argument("--domain", help="Focus domain (e.g., 'career')")
    think_parser.add_argument("--mode", choices=["general", "tension"], default="general",
                             help="Think mode: general (default) or tension (find contradictions)")
    think_parser.set_defaults(func=cmd_think)
    
    # Sleep command
    sleep_parser = subparsers.add_parser("sleep", help="Run the sleep/consolidation protocol")
    sleep_parser.set_defaults(func=cmd_sleep)
    
    # Stats command
    stats_parser = subparsers.add_parser("stats", help="Show graph stats")
    stats_parser.set_defaults(func=cmd_stats)
    
    # Hotspot command
    hotspot_parser = subparsers.add_parser("hotspot", help="Manage hotspot nodes")
    hotspot_parser.add_argument("hotspot_action", choices=["create", "update", "list", "show"],
                                help="Hotspot action")
    hotspot_parser.add_argument("--content", help="Hotspot summary content")
    hotspot_parser.add_argument("--status", help="Status string")
    hotspot_parser.add_argument("--files", help="File pointers as 'label:path,label:path'")
    hotspot_parser.add_argument("--cluster", help="Comma-separated cluster node IDs")
    hotspot_parser.add_argument("--tags", help="Comma-separated search tags")
    hotspot_parser.add_argument("--domain", help="Domain (raj/bunny)")
    hotspot_parser.add_argument("--id", help="Hotspot ID (for update/show)")
    hotspot_parser.set_defaults(func=cmd_hotspot)
    
    # === COMPLETE COVERAGE SYSTEM COMMANDS ===
    
    # Complete context command
    complete_context_parser = subparsers.add_parser("complete-context", help="Generate context with complete coverage retrieval")
    complete_context_parser.add_argument("hints", nargs="*", help="Topic hints for context generation")
    complete_context_parser.add_argument("--method", choices=["dfs", "hierarchical", "breadth_first"], 
                                        default="dfs", help="Retrieval method")
    complete_context_parser.add_argument("--top-k", type=int, default=5, help="Number of context items")
    complete_context_parser.add_argument("--domain", help="Optional domain filter")
    complete_context_parser.set_defaults(func=cmd_complete_context)
    
    # Complete extract command
    complete_extract_parser = subparsers.add_parser("complete-extract", help="Extract with placement-aware assignment")
    complete_extract_parser.add_argument("--input", required=True, help="Input conversation file")
    complete_extract_parser.add_argument("--session-id", help="Optional session ID")
    complete_extract_parser.set_defaults(func=cmd_complete_extract)
    
    # Complete think command
    complete_think_parser = subparsers.add_parser("complete-think", help="Run think cycle with placement-aware assignment")
    complete_think_parser.add_argument("--domain", help="Focus domain")
    complete_think_parser.set_defaults(func=cmd_complete_think)
    
    # Complete sleep command  
    complete_sleep_parser = subparsers.add_parser("complete-sleep", help="Run complete sleep cycle with hierarchy evolution")
    complete_sleep_parser.add_argument("--no-evolution", action="store_true", 
                                      help="Disable hierarchy evolution (merge/split/promote/reclassify)")
    complete_sleep_parser.set_defaults(func=cmd_complete_sleep)
    
    # Migration command
    migrate_parser = subparsers.add_parser("migrate", help="Migrate to complete coverage system")
    migrate_parser.add_argument("--dry-run", action="store_true", help="Show what would be done without making changes")
    migrate_parser.set_defaults(func=cmd_migrate)
    
    # Explain command
    explain_parser = subparsers.add_parser("explain", help="Explain the complete coverage system")
    explain_parser.add_argument("--query", default="test query", help="Test query for demonstration")
    explain_parser.set_defaults(func=cmd_explain)
    
    # System stats command
    system_stats_parser = subparsers.add_parser("system-stats", help="Show complete system statistics")
    system_stats_parser.set_defaults(func=cmd_system_stats)
    
    # Init command
    init_parser = subparsers.add_parser("init", help="Initialize a new cashew database")
    init_parser.add_argument("--db", dest="sub_db", default=None, help="Database path (can also be specified before subcommand)")
    init_parser.set_defaults(func=cmd_init)
    
    # Migrate files command
    migrate_files_parser = subparsers.add_parser("migrate-files", help="Migrate markdown files to cashew database")
    migrate_files_parser.add_argument("--dir", required=True, help="Directory containing markdown files")
    migrate_files_parser.add_argument("--dry-run", action="store_true", help="Show what would be migrated without making changes")
    migrate_files_parser.add_argument("--db", dest="sub_db", default=None, help="Database path (can also be specified before subcommand)")
    migrate_files_parser.set_defaults(func=cmd_migrate_files)
    
    args = parser.parse_args(_preprocess_db_flag(sys.argv[1:]))
    
    if not args.command:
        parser.print_help()
        return 1
    
    # Allow --db after subcommand (sub_db overrides if set)
    if hasattr(args, 'sub_db') and args.sub_db is not None:
        args.db = args.sub_db
    
    if args.debug:
        logging.basicConfig(level=logging.DEBUG, format="%(name)s %(levelname)s: %(message)s")
    elif args.verbose:
        logging.basicConfig(level=logging.INFO)
    
    # Commands that don't require existing database
    init_commands = ['init']
    
    if args.command not in init_commands and not os.path.exists(args.db):
        print(f"❌ Error: Database not found: {args.db}")
        print(f"💡 Run `cashew init --db {args.db}` to create a new database")
        return 1
    
    if args.debug:
        db_size = os.path.getsize(args.db)
        print(f"🗄  DB: {args.db} ({db_size/1024:.1f} KB)", file=sys.stderr)
        print(f"🔧 Command: {args.command}", file=sys.stderr)
    
    return args.func(args) or 0


if __name__ == "__main__":
    sys.exit(main())