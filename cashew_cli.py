#!/usr/bin/env python3
"""
Cashew CLI - Command line interface for the cashew thought-graph engine
"""

import sys
import argparse
import json
import os
import time
import logging
import sqlite3
import shutil
from pathlib import Path
import yaml

# Add the current directory to the path so we can import cashew modules
sys.path.insert(0, str(Path(__file__).parent))

from core.config import config, reload_config, get_db_path
from core.embeddings import embed_text

logger = logging.getLogger("cashew")


def cmd_init(args):
    """Initialize a new cashew brain"""
    print("🧠 Initializing Cashew brain...")
    
    # Create config.yaml from template if it doesn't exist
    config_path = Path("config.yaml")
    template_path = Path(__file__).parent / "config.yaml.template"
    
    if not config_path.exists():
        if template_path.exists():
            print(f"📝 Creating config.yaml from template...")
            shutil.copy(template_path, config_path)
        else:
            print("❌ Error: config.yaml.template not found")
            return 1
        
        print("✅ Created config.yaml - edit this file to customize your setup")
    else:
        print("📝 Using existing config.yaml")
    
    # Reload config to pick up any changes
    reload_config("config.yaml")
    
    # Create data directory
    db_path = Path(get_db_path())
    db_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Create logs directory
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    # Create models directory
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    
    print(f"📂 Created directory structure")
    
    # Initialize empty graph database with schema
    print(f"🗄️ Initializing database: {db_path}")
    
    try:
        conn = sqlite3.connect(str(db_path))
        
        # Create schema
        conn.execute('''
            CREATE TABLE IF NOT EXISTS thought_nodes (
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
        
        conn.execute('''
            CREATE TABLE IF NOT EXISTS derivation_edges (
                parent_id TEXT,
                child_id TEXT,
                weight REAL,
                reasoning TEXT,
                confidence REAL,
                timestamp TEXT,
                PRIMARY KEY (parent_id, child_id),
                FOREIGN KEY (parent_id) REFERENCES thought_nodes(id),
                FOREIGN KEY (child_id) REFERENCES thought_nodes(id)
            )
        ''')
        
        conn.execute('''
            CREATE TABLE IF NOT EXISTS embeddings (
                node_id TEXT PRIMARY KEY,
                vector BLOB NOT NULL,
                model TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                FOREIGN KEY (node_id) REFERENCES thought_nodes(id)
            )
        ''')
        
        conn.execute('''
            CREATE TABLE IF NOT EXISTS hotspots (
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
        
        # Create indexes for performance
        conn.execute('CREATE INDEX IF NOT EXISTS idx_nodes_domain ON thought_nodes(domain)')
        conn.execute('CREATE INDEX IF NOT EXISTS idx_nodes_timestamp ON thought_nodes(timestamp)')
        conn.execute('CREATE INDEX IF NOT EXISTS idx_nodes_confidence ON thought_nodes(confidence)')
        conn.execute('CREATE INDEX IF NOT EXISTS idx_edges_parent ON derivation_edges(parent_id)')
        conn.execute('CREATE INDEX IF NOT EXISTS idx_edges_child ON derivation_edges(child_id)')
        
        conn.commit()
        conn.close()
        
        print(f"✅ Database initialized with schema")
        
    except sqlite3.Error as e:
        print(f"❌ Database error: {e}")
        return 1
    
    # Test embedding model (downloads automatically on first use)
    print("📥 Testing embedding model...")
    try:
        test_embedding = embed_text("test")
        if len(test_embedding) == 384:  # Expected dimension for all-MiniLM-L6-v2
            print("✅ Embedding model ready")
        else:
            print(f"⚠️  Warning: Unexpected embedding dimension: {len(test_embedding)}")
    except Exception as e:
        print(f"⚠️  Warning: Could not load embedding model: {e}")
        print("   The model will be downloaded on first use")
    
    print()
    print("🎉 Cashew initialization complete!")
    print()
    print("Next steps:")
    print("1. Edit config.yaml to customize settings")
    print("2. Run 'cashew context --hints \"test\"' to verify")
    print("3. Use 'cashew install-crons' to set up automated maintenance")
    
    return 0


def cmd_install_crons(args):
    """Generate OpenClaw cron job configurations"""
    print("⏰ Generating OpenClaw cron job configurations...")
    
    # Get configuration
    raw_config = config.get_raw_config()
    cron_config = raw_config.get('integration', {}).get('cron', {})
    
    # Define cron job templates
    cron_jobs = {
        'brain-extract': {
            'description': 'Extract knowledge from session history to brain',
            'schedule': cron_config.get('extract_schedule', '0 */2 * * *'),
            'command': 'cashew extract-session',
            'model': cron_config.get('extract_model', 'anthropic/claude-haiku-4-5'),
            'timeout': 300
        },
        'think-cycle': {
            'description': 'Run think cycle for knowledge consolidation',
            'schedule': cron_config.get('think_schedule', '0 6,18 * * *'),
            'command': 'cashew think',
            'model': cron_config.get('think_model', 'anthropic/claude-sonnet-4-20250514'),
            'timeout': 600
        },
        'sleep-cycle': {
            'description': 'Run sleep cycle for deep reorganization',
            'schedule': cron_config.get('sleep_schedule', '0 3 * * *'),
            'command': 'cashew sleep',
            'model': cron_config.get('sleep_model', 'anthropic/claude-sonnet-4-20250514'),
            'timeout': 1800
        },
        'backup': {
            'description': 'Backup graph database',
            'schedule': cron_config.get('backup_schedule', '0 4 * * *'),
            'command': 'cashew backup',
            'model': None,  # No model needed for backup
            'timeout': 120
        },
        'health-check': {
            'description': 'Run health check and basic maintenance',
            'schedule': cron_config.get('health_schedule', '*/30 * * * *'),
            'command': 'cashew health',
            'model': 'anthropic/claude-haiku-4-5',
            'timeout': 60
        }
    }
    
    # Generate cron job configs for OpenClaw
    cron_output = {}
    
    for job_name, job_config in cron_jobs.items():
        openclaw_job = {
            'schedule': job_config['schedule'],
            'command': job_config['command'],
            'description': job_config['description'],
            'timeout': job_config['timeout'],
            'workdir': '${OPENCLAW_WORKSPACE}/cashew'
        }
        
        if job_config['model']:
            openclaw_job['model'] = job_config['model']
            openclaw_job['pty'] = True
        
        cron_output[job_name] = openclaw_job
    
    # Write to file
    output_file = 'cashew-crons.yaml'
    with open(output_file, 'w') as f:
        yaml.dump({
            'cron_jobs': cron_output,
            'instructions': {
                'setup': 'Add these cron job definitions to your OpenClaw config file',
                'path': '~/.openclaw/config/config.yaml under the cron_jobs section',
                'note': 'Adjust schedules and models as needed for your setup'
            }
        }, f, default_flow_style=False, sort_keys=True)
    
    print(f"✅ Generated cron job configurations in {output_file}")
    print()
    print("To install:")
    print("1. Copy the cron job definitions from cashew-crons.yaml")
    print("2. Add them to your OpenClaw config file (~/.openclaw/config/config.yaml)")
    print("3. Restart OpenClaw gateway to pick up the new jobs")
    print()
    print("Jobs generated:")
    for job_name, job_config in cron_output.items():
        print(f"  - {job_name}: {job_config['description']} ({job_config['schedule']})")
    
    return 0


def cmd_pin(args):
    """Pin a node as permanently important"""
    from core.permanence import pin_node
    
    db_path = args.db if args.db else get_db_path()
    
    success = pin_node(db_path, args.node_id)
    
    if success:
        print(f"✅ Node {args.node_id} pinned as permanent")
        return 0
    else:
        print(f"❌ Node {args.node_id} not found")
        return 1


def cmd_unpin(args):
    """Remove permanent pin from a node"""
    from core.permanence import unpin_node
    
    db_path = args.db if args.db else get_db_path()
    
    success = unpin_node(db_path, args.node_id)
    
    if success:
        print(f"✅ Node {args.node_id} unpinned")
        return 0
    else:
        print(f"❌ Node {args.node_id} not found")
        return 1


def cmd_permanent(args):
    """Show permanence statistics or list permanent nodes"""
    from core.permanence import get_permanence_stats, get_permanent_nodes
    
    db_path = args.db if args.db else get_db_path()
    
    if args.stats:
        # Show statistics
        stats = get_permanence_stats(db_path)
        
        print("🔒 Permanence Statistics:")
        print(f"  Total active nodes: {stats['total_active_nodes']}")
        print(f"  Permanent nodes: {stats['total_permanent_nodes']} ({stats['permanent_ratio']:.1%})")
        print(f"    • Auto-permanent: {stats['auto_permanent']}")
        print(f"    • Manually pinned: {stats['manually_pinned']}")
        print(f"  Permanent hotspots: {stats['permanent_hotspots']}/{stats['total_hotspots']} ({stats['hotspot_permanent_ratio']:.1%})")
        
        if stats['permanent_by_domain']:
            print(f"  By domain:")
            for domain, count in sorted(stats['permanent_by_domain'].items()):
                print(f"    • {domain}: {count}")
        
        return 0
        
    else:
        # List permanent nodes
        nodes = get_permanent_nodes(db_path, include_manually_pinned=True)
        
        if not nodes:
            print("No permanent nodes found.")
            return 0
        
        print(f"🔒 Permanent Nodes ({len(nodes)}):")
        for node in nodes[:20]:  # Limit to first 20
            pin_type = "📌" if node['permanent_type'] == 'manually_pinned' else "🔒"
            print(f"  {pin_type} {node['id']} ({node['node_type']}/{node['domain']})")
            print(f"      {node['content']}")
            print(f"      Access: {node['access_count']}, Confidence: {node['confidence']:.3f}")
            print()
        
        if len(nodes) > 20:
            print(f"... and {len(nodes) - 20} more. Use --stats for summary.")
        
        return 0


def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description="Cashew - Persistent thought-graph memory for AI agents",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  cashew init                           # Initialize new brain
  cashew context --hints "work tasks"   # Query for context
  cashew extract --input notes.txt     # Extract knowledge from text
  cashew think                          # Run think cycle
  cashew sleep                          # Run sleep cycle
  cashew stats                          # Show graph statistics
  cashew pin node123                    # Pin node as permanent
  cashew unpin node123                  # Remove permanent pin
  cashew permanent --stats              # Show permanence statistics
  cashew install-crons                  # Generate cron job configs
        """)
    
    parser.add_argument("--config", help="Path to config.yaml file")
    parser.add_argument("--db", help="Database path (overrides config)")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # init command
    init_parser = subparsers.add_parser('init', help='Initialize new cashew brain')
    init_parser.set_defaults(func=cmd_init)
    
    # install-crons command
    crons_parser = subparsers.add_parser('install-crons', help='Generate OpenClaw cron job configs')
    crons_parser.set_defaults(func=cmd_install_crons)
    
    # pin command
    pin_parser = subparsers.add_parser('pin', help='Pin a node as permanently important')
    pin_parser.add_argument('node_id', help='Node ID to pin')
    pin_parser.set_defaults(func=cmd_pin)
    
    # unpin command
    unpin_parser = subparsers.add_parser('unpin', help='Remove permanent pin from a node')
    unpin_parser.add_argument('node_id', help='Node ID to unpin')
    unpin_parser.set_defaults(func=cmd_unpin)
    
    # permanent command  
    permanent_parser = subparsers.add_parser('permanent', help='Show permanence statistics and permanent nodes')
    permanent_parser.add_argument('--stats', action='store_true', help='Show statistics instead of listing nodes')
    permanent_parser.set_defaults(func=cmd_permanent)
    
    # TODO: Add other commands (context, extract, think, sleep, stats, etc.)
    # For now, we'll just implement the essential ones for packaging
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    # Configure logging
    if args.debug:
        logging.basicConfig(level=logging.DEBUG)
    
    # Load configuration
    if args.config:
        reload_config(args.config)
    
    # Override DB path if specified
    if args.db:
        config.db_path = args.db
    
    # Execute command
    try:
        return args.func(args)
    except KeyboardInterrupt:
        print("\n❌ Interrupted")
        return 130
    except Exception as e:
        if args.debug:
            import traceback
            traceback.print_exc()
        else:
            print(f"❌ Error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())