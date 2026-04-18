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
from core.extractors import ExtractorRegistry
from extractors import ObsidianExtractor, SessionExtractor, MarkdownDirExtractor

logger = logging.getLogger("cashew")


def cmd_init(args):
    """Initialize a new cashew brain"""
    print("🧠 Initializing Cashew brain...")

    # Resolve config path from args or default
    config_path = Path(args.config) if args.config else Path("config.yaml")
    config_path.parent.mkdir(parents=True, exist_ok=True)
    template_path = Path(__file__).parent / "config.yaml.template"

    if not config_path.exists():
        if template_path.exists():
            print(f"📝 Creating {config_path} from template...")
            shutil.copy(template_path, config_path)
        else:
            print("❌ Error: config.yaml.template not found")
            return 1

        print(f"✅ Created {config_path} - edit this file to customize your setup")
    else:
        print(f"📝 Using existing {config_path}")

    # Reload config from the resolved path
    reload_config(str(config_path))

    # Resolve db path from args, config, or default
    db_path = Path(args.db) if args.db else Path(get_db_path())
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
                mood_state TEXT,
                tags TEXT DEFAULT ""
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
    
    # Ask about backup preferences
    print()
    enable_backups = False  # Default fallback
    backup_interval = "6h"   # Default fallback
    retention_period = "24h" # Default fallback
    
    try:
        while True:
            backup_choice = input("Would you like to enable automatic database backups? (y/n): ").lower().strip()
            if backup_choice in ['y', 'yes']:
                enable_backups = True
                break
            elif backup_choice in ['n', 'no']:
                enable_backups = False
                break
            else:
                print("Please enter 'y' or 'n'")
        
        if enable_backups:
            # Ask for backup interval
            while True:
                interval_input = input("Backup interval (default: 6h, options: 3h, 6h, 12h, 1d): ").strip()
                if not interval_input:
                    backup_interval = "6h"
                    break
                elif interval_input in ['3h', '6h', '12h', '1d']:
                    backup_interval = interval_input
                    break
                else:
                    print("Please enter a valid interval (3h, 6h, 12h, 1d)")
            
            # Ask for retention period  
            while True:
                retention_input = input("Backup retention period (default: 24h, options: 12h, 24h, 3d, 1w): ").strip()
                if not retention_input:
                    retention_period = "24h"
                    break
                elif retention_input in ['12h', '24h', '3d', '1w']:
                    retention_period = retention_input
                    break
                else:
                    print("Please enter a valid retention period (12h, 24h, 3d, 1w)")
            
            # Update config with backup settings
            print(f"✅ Configured automatic backups: every {backup_interval}, retain for {retention_period}")
            
            # Update config.yaml with backup settings
            reload_config(str(config_path))
            backup_config = {
                'enabled': True,
                'interval': backup_interval,
                'retention_period': retention_period,
                'auto_backup_enabled': True,
                'schedule': _interval_to_cron(backup_interval)
            }
            
            # Read existing config and update backup section
            with open(config_path, 'r') as f:
                config_data = yaml.safe_load(f) or {}
            
            config_data['backup'] = backup_config
            
            with open(config_path, 'w') as f:
                yaml.dump(config_data, f, default_flow_style=False, sort_keys=True)
            
            print(f"📝 Updated {config_path} with backup settings")
        
        else:
            print("⚠️ Automatic backups disabled - you can enable them later by editing config.yaml")
    
    except KeyboardInterrupt:
        print("\n⚠️ Setup interrupted - using default backup settings")
    
    print()
    print("🎉 Cashew initialization complete!")
    print()
    print("Next steps:")
    print("1. Edit config.yaml to customize settings")
    print("2. Run 'cashew context --hints \"test\"' to verify")
    print("3. Use 'cashew install-crons' to set up automated maintenance")
    if enable_backups:
        print("4. Test backup with 'cashew backup'")
    
    return 0


def _interval_to_cron(interval: str) -> str:
    """Convert backup interval to cron schedule."""
    if interval == "3h":
        return "0 */3 * * *"
    elif interval == "6h":
        return "0 */6 * * *" 
    elif interval == "12h":
        return "0 */12 * * *"
    elif interval == "1d":
        return "0 4 * * *"  # Daily at 4 AM
    else:
        return "0 */6 * * *"  # Default to 6h


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
            'schedule': cron_config.get('backup_schedule', '0 */6 * * *'),
            'command': 'cashew backup --retention 24h',
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


def cmd_backup(args):
    """Create database backup and manage retention"""
    from core.backup import create_backup, cleanup_old_backups, get_backup_stats, parse_retention_period
    
    db_path = args.db if args.db else get_db_path()
    
    if args.stats:
        # Show backup statistics
        stats = get_backup_stats()
        
        print("💾 Backup Statistics:")
        print(f"  Total backups: {stats['count']}")
        if stats['count'] > 0:
            print(f"  Total size: {_human_readable_size_from_bytes(stats['total_size'])}")
            print(f"  Oldest backup: {stats['oldest'].strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"  Newest backup: {stats['newest'].strftime('%Y-%m-%d %H:%M:%S')}")
            print()
            print("Recent backups:")
            for backup in stats['files'][-10:]:  # Show last 10
                backup_path = Path(backup['path'])
                size_str = _human_readable_size_from_bytes(backup['size'])
                created_str = backup['created'].strftime('%Y-%m-%d %H:%M')
                print(f"  📄 {backup_path.name} ({size_str}, {created_str})")
        else:
            print("  No backups found")
        
        return 0
    
    elif args.cleanup:
        # Clean up old backups
        retention_hours = parse_retention_period(args.retention or "24h")
        deleted = cleanup_old_backups(retention_hours=retention_hours)
        
        if deleted:
            print(f"🗑️ Cleaned up {len(deleted)} old backups (retention: {args.retention or '24h'})")
            for deleted_file in deleted:
                print(f"  • {Path(deleted_file).name}")
        else:
            print("✅ No old backups to clean up")
        
        return 0
    
    else:
        # Create backup
        print(f"💾 Creating database backup...")
        
        backup_path = create_backup(db_path)
        if backup_path:
            print(f"✅ Backup created: {Path(backup_path).name}")
            
            # Auto-cleanup if retention specified
            if args.retention:
                retention_hours = parse_retention_period(args.retention)
                deleted = cleanup_old_backups(retention_hours=retention_hours)
                if deleted:
                    print(f"🗑️ Cleaned up {len(deleted)} old backups")
            
            return 0
        else:
            print("❌ Backup failed")
            return 1


def cmd_ingest(args):
    """Ingest knowledge using extractor plugins"""
    db_path = args.db if args.db else get_db_path()
    
    # Initialize registry and register extractors
    data_dir = str(Path(db_path).parent)
    registry = ExtractorRegistry(data_dir=data_dir)
    registry.register(ObsidianExtractor())
    registry.register(SessionExtractor())
    registry.register(MarkdownDirExtractor())
    
    if args.list:
        # List available extractors
        extractors = registry.list_extractors()
        print("📦 Available Extractors:")
        for name in extractors:
            print(f"  • {name}")
        return 0
    
    if not args.extractor:
        print("❌ Error: extractor type required")
        print("Use --list to see available extractors")
        return 1
    
    if not args.path:
        print("❌ Error: source path required")
        return 1
    
    # Get model function if configured
    model_fn = None
    if not args.no_llm:
        try:
            from scripts.cashew_context import _build_model_fn
            model_fn = _build_model_fn()
            if model_fn is None:
                logger.warning("No LLM gateway found, using fallback extraction")
        except ImportError:
            logger.warning("No LLM integration available, using fallback extraction")
    
    # Run the extractor
    print(f"🧠 Running {args.extractor} extractor on {args.path}...")
    
    try:
        # Prepare kwargs for extractor-specific options
        extractor_kwargs = {}
        if args.no_llm:
            extractor_kwargs['skip_llm'] = True

        result = registry.run(args.extractor, args.path, model_fn, db_path, **extractor_kwargs)
        
        if result["errors"]:
            print("❌ Extraction completed with errors:")
            for error in result["errors"]:
                print(f"  • {error}")
            return 1
        else:
            nodes_created = result["nodes_created"]
            print(f"✅ Extraction complete: {nodes_created} nodes created")
            return 0
            
    except KeyError as e:
        print(f"❌ Error: Unknown extractor '{args.extractor}'")
        print("Use --list to see available extractors")
        return 1
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1


def cmd_serve(args):
    """Run the warm daemon that keeps the embedding model loaded."""
    import logging as _logging
    _logging.basicConfig(level=_logging.INFO, format="%(asctime)s %(name)s: %(message)s")
    from core.daemon import serve
    serve(socket_path=args.socket, warm=not args.no_warm)
    return 0


def _human_readable_size_from_bytes(size_bytes: int) -> str:
    """Convert bytes to human readable size."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024:
            return f"{size_bytes:.1f}{unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f}TB"


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
  cashew ingest obsidian /path/to/vault # Extract from Obsidian vault
  cashew ingest sessions /path/to/logs  # Extract from session logs
  cashew ingest markdown /path/to/notes # Extract from markdown directory
  cashew ingest --list                  # Show available extractors
  cashew think                          # Run think cycle
  cashew sleep                          # Run sleep cycle
  cashew backup                         # Create database backup
  cashew backup --stats                 # Show backup statistics
  cashew backup --cleanup --retention 3d # Clean old backups
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
    init_parser.add_argument('--db', dest='sub_db', default=None, help='Database path')
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
    
    
    # backup command
    backup_parser = subparsers.add_parser('backup', help='Create database backup and manage retention')
    backup_parser.add_argument('--stats', action='store_true', help='Show backup statistics instead of creating backup')
    backup_parser.add_argument('--cleanup', action='store_true', help='Clean up old backups instead of creating backup')
    backup_parser.add_argument('--retention', help='Retention period for cleanup (e.g., "24h", "3d", "1w")')
    backup_parser.set_defaults(func=cmd_backup)
    
    # ingest command
    ingest_parser = subparsers.add_parser('ingest', help='Ingest knowledge using extractor plugins')
    ingest_parser.add_argument('extractor', nargs='?', choices=['obsidian', 'sessions', 'markdown'],
                               help='Extractor type to use')
    ingest_parser.add_argument('path', nargs='?', help='Source path to extract from')
    ingest_parser.add_argument('--list', action='store_true', help='List available extractors')
    ingest_parser.add_argument('--no-llm', action='store_true', help='Disable LLM extraction, use fallback only')
    ingest_parser.set_defaults(func=cmd_ingest)
    
    # Core workflow commands (delegated to cashew_context.py functions)
    from scripts.cashew_context import (
        cmd_context, cmd_extract, cmd_think, cmd_sleep, cmd_stats,
        cmd_migrate_files, cmd_complete_context, cmd_complete_sleep,
        cmd_system_stats
    )
    
    # context command
    context_parser = subparsers.add_parser('context', help='Query brain for relevant context')
    context_parser.add_argument('--hints', nargs='*', help='Topic hints (e.g., "work promotion")')
    context_parser.set_defaults(func=cmd_context)
    
    # extract command
    extract_parser = subparsers.add_parser('extract', help='Extract knowledge from text')
    extract_parser.add_argument('--input', required=True, help='Input text file')
    extract_parser.add_argument('--session-id', help='Optional session ID')
    extract_parser.add_argument('--referent-time', dest='referent_time', default=None,
                                help='UTC ISO8601 event time (tz-aware) applied to all extracted nodes')
    extract_parser.add_argument('--infer-referent-time', dest='infer_referent_time',
                                action='store_true', default=False,
                                help='Opt-in LLM inference of per-node event time. Default OFF.')
    extract_parser.set_defaults(func=cmd_extract)
    
    # think command
    think_parser = subparsers.add_parser('think', help='Run a think cycle')
    think_parser.add_argument('--domain', help='Focus domain')
    think_parser.add_argument('--mode', default='general', choices=['general', 'tension'],
                              help='Think mode')
    think_parser.set_defaults(func=cmd_think)
    
    # sleep command
    sleep_parser = subparsers.add_parser('sleep', help='Run sleep/consolidation protocol')
    sleep_parser.add_argument('--eps', type=float, default=None,
                              help='Clustering threshold (0.0-1.0). Higher = looser clusters')
    sleep_parser.set_defaults(func=cmd_sleep)
    
    # stats command
    stats_parser = subparsers.add_parser('stats', help='Show graph statistics')
    stats_parser.set_defaults(func=cmd_stats)
    
    # system-stats command
    sys_stats_parser = subparsers.add_parser('system-stats', help='Show complete system statistics')
    sys_stats_parser.set_defaults(func=cmd_system_stats)
    
    # migrate-files command
    mf_parser = subparsers.add_parser('migrate-files', help='Migrate markdown files to cashew')
    mf_parser.add_argument('--dir', required=True, help='Directory containing markdown files')
    mf_parser.add_argument('--dry-run', action='store_true', help='Preview without changes')
    mf_parser.add_argument('--db', dest='sub_db', default=None, help='Database path')
    mf_parser.set_defaults(func=cmd_migrate_files)
    
    # complete-context command
    cc_parser = subparsers.add_parser('complete-context', help='Context with complete coverage retrieval')
    cc_parser.add_argument('--hints', nargs='*', help='Topic hints')
    cc_parser.add_argument('--method', default='dfs', choices=['dfs', 'hierarchical', 'breadth_first'])
    cc_parser.set_defaults(func=cmd_complete_context)
    
    # complete-sleep command
    cs_parser = subparsers.add_parser('complete-sleep', help='Full sleep with hierarchy evolution')
    cs_parser.add_argument('--no-evolution', action='store_true', help='Skip hierarchy evolution')
    cs_parser.set_defaults(func=cmd_complete_sleep)

    serve_parser = subparsers.add_parser('serve', help='Run the warm daemon (keeps model + sqlite-vec loaded)')
    serve_parser.add_argument('--socket', help='Unix socket path (default: $CASHEW_SOCKET or ~/.cashew/daemon.sock)')
    serve_parser.add_argument('--no-warm', action='store_true', help='Skip loading the embedding model at startup')
    serve_parser.set_defaults(func=cmd_serve)

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
    
    # Allow --db after subcommand (sub_db overrides top-level --db)
    if hasattr(args, 'sub_db') and args.sub_db is not None:
        args.db = args.sub_db
    
    # Ensure args.db is always set (needed by delegated commands from cashew_context.py)
    if args.db:
        config.db_path = args.db
    else:
        args.db = get_db_path()
    
    # Ensure args.verbose and args.debug exist for delegated commands
    if not hasattr(args, 'verbose'):
        args.verbose = False
    
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