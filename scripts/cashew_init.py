#!/usr/bin/env python3
"""
Cashew Interactive Setup CLI
Interactive setup wizard for the cashew thought-graph memory engine
"""

import os
import sys
import yaml
import argparse
import subprocess
import json
import shutil
import requests
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

# Ensure repo root on path so core.db imports work when run as a script.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from core import db as cdb  # noqa: E402

# ANSI color codes for pretty output
class Colors:
    GREEN = '\033[92m'
    BLUE = '\033[94m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    DIM = '\033[2m'
    RESET = '\033[0m'

def print_header(text: str):
    """Print a colorful header"""
    print(f"\n{Colors.BOLD}{Colors.BLUE}🥜 {text}{Colors.RESET}")

def print_success(text: str):
    """Print success message"""
    print(f"{Colors.GREEN}✅ {text}{Colors.RESET}")

def print_info(text: str):
    """Print info message"""
    print(f"{Colors.BLUE}ℹ️  {text}{Colors.RESET}")

def print_warning(text: str):
    """Print warning message"""
    print(f"{Colors.YELLOW}⚠️  {text}{Colors.RESET}")

def print_error(text: str):
    """Print error message"""
    print(f"{Colors.RED}❌ {text}{Colors.RESET}")

def prompt_with_default(question: str, default: str, options: Optional[List[str]] = None) -> str:
    """Prompt user with a default value"""
    if options:
        options_str = f" ({'/'.join(options)})"
    else:
        options_str = ""
    
    prompt = f"{Colors.YELLOW}? {question}{options_str} [{Colors.BOLD}{default}{Colors.RESET}{Colors.YELLOW}]: {Colors.RESET}"
    
    while True:
        response = input(prompt).strip()
        if not response:
            return default
        
        if options and response not in options:
            print_error(f"Please choose from: {', '.join(options)}")
            continue
            
        return response

def prompt_yes_no(question: str, default: bool = True) -> bool:
    """Prompt user for yes/no with default"""
    default_str = "Y/n" if default else "y/N"
    prompt = f"{Colors.YELLOW}? {question} [{default_str}]: {Colors.RESET}"
    
    while True:
        response = input(prompt).strip().lower()
        if not response:
            return default
        if response in ['y', 'yes', 'true']:
            return True
        if response in ['n', 'no', 'false']:
            return False
        print_error("Please answer y/n")

def prompt_list(question: str, default_list: List[str], separator: str = ",") -> List[str]:
    """Prompt user for a list of values"""
    default_str = separator.join(default_list)
    prompt = f"{Colors.YELLOW}? {question} [{Colors.BOLD}{default_str}{Colors.RESET}{Colors.YELLOW}]: {Colors.RESET}"
    
    response = input(prompt).strip()
    if not response:
        return default_list
    
    return [item.strip() for item in response.split(separator) if item.strip()]

def create_database_schema(db_path: str):
    """Create the SQLite database with proper schema"""
    print_info(f"Creating database at {db_path}")
    
    # Ensure parent directory exists
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    
    conn = cdb.connect(db_path)
    cursor = conn.cursor()
    
    # Core schema matching the test fixtures
    cursor.execute('''
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
            metadata TEXT DEFAULT '{}',
            last_updated TEXT,
            mood_state TEXT,
            permanent INTEGER DEFAULT 0,
            tags TEXT
        )
    ''')
    
    cursor.execute('''
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
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS embeddings (
            node_id TEXT PRIMARY KEY,
            vector BLOB NOT NULL,
            model TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            FOREIGN KEY (node_id) REFERENCES thought_nodes(id)
        )
    ''')
    
    cursor.execute('''
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
    
    # Metrics table for performance tracking
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS metrics (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            metric_name TEXT NOT NULL,
            metric_value REAL NOT NULL,
            tags TEXT
        )
    ''')
    
    conn.commit()
    conn.close()
    
    print_success(f"Database schema created at {db_path}")

def build_config(interactive: bool = True, config_path: str = None, global_config: bool = False, data_dir_override: str = None) -> Dict[str, Any]:
    """Build configuration dictionary through prompts or defaults"""
    config = {}
    
    # Determine data directory based on config location or override
    if data_dir_override:
        # Explicit override takes precedence
        default_data_dir = data_dir_override
    elif config_path:
        # When custom config path is specified, put data relative to config location
        config_parent = Path(config_path).parent
        default_data_dir = str(config_parent / "data")
    elif global_config:
        # Global install uses ~/.cashew/data
        default_data_dir = str(Path.home() / ".cashew" / "data")
    else:
        # Default to global behavior for backward compatibility
        default_data_dir = str(Path.home() / ".cashew" / "data")
    
    if interactive:
        print_header("Cashew Setup Wizard")
        print("Welcome! Let's configure your personal thought-graph memory engine.\n")
        
        # Data directory
        data_dir = prompt_with_default(
            "Where should cashew store your data?",
            default_data_dir
        )
        db_path = str(Path(data_dir) / "graph.db")
        
        # Embedding model
        print_info("\nEmbedding models convert text to vectors for similarity search:")
        print("  • local: Uses sentence-transformers (private, no API key)")
        print("  • openai: Uses OpenAI embeddings (requires OPENAI_API_KEY)")
        
        embedding_provider = prompt_with_default(
            "Which embedding model?",
            "local",
            ["local", "openai"]
        )
        
        # LLM provider
        print_info("\nLLM providers handle extraction and reasoning:")
        print("  • anthropic: Claude models (requires ANTHROPIC_API_KEY)")
        print("  • openai: GPT models (requires OPENAI_API_KEY)")
        print("  • ollama: Local models (fully private, no API key)")
        print("  • claude_code: Headless Claude Code CLI (runs under Max plan, no key needed)")

        llm_provider = prompt_with_default(
            "Which LLM provider?",
            "claude_code",
            ["anthropic", "openai", "ollama", "claude_code"]
        )
        
        # API key prompt (only if needed)
        api_key = None
        if embedding_provider == "openai" or llm_provider in ["anthropic", "openai"]:
            if llm_provider == "anthropic":
                key_name = "ANTHROPIC_API_KEY"
            else:
                key_name = "OPENAI_API_KEY"
            
            existing_key = os.getenv(key_name)
            if existing_key:
                print_info(f"{key_name} already set in environment")
            else:
                api_key = input(f"{Colors.YELLOW}? Enter your {key_name} (leave blank to set later): {Colors.RESET}").strip()
        
        # Sleep cycle frequency
        print_info("\nSleep cycles perform memory consolidation and reorganization")
        
        sleep_frequency = prompt_with_default(
            "How often should sleep cycles run?",
            "6h",
            ["6h", "12h", "manual"]
        )
        
        # Domain separation
        use_domains = prompt_yes_no(
            "Enable domain separation? (recommended for organizing different life areas)",
            True
        )
        
        domains = ["personal", "work"]
        if use_domains:
            domains = prompt_list(
                "What domains would you like to separate? (comma-separated)",
                ["personal", "work"]
            )
    
    else:
        # Non-interactive defaults
        data_dir = default_data_dir
        db_path = str(Path(data_dir) / "graph.db")
        embedding_provider = "local"
        llm_provider = "anthropic"
        api_key = None
        sleep_frequency = "6h"
        use_domains = True
        domains = ["personal", "work"]
    
    # Build the configuration with paths relative to config location when possible
    if config_path and not data_dir_override and not global_config:
        # Use relative paths when config is in a custom location (for portability)
        config_parent = Path(config_path).parent
        db_rel_path = str(Path(data_dir).relative_to(config_parent) / "graph.db")
        backup_rel_path = str(Path(data_dir).relative_to(config_parent) / "backups")
        models_rel_path = str(Path(data_dir).relative_to(config_parent) / "models")
        logs_rel_path = str(Path(data_dir).relative_to(config_parent) / "logs" / "cashew.log")
    else:
        # Use absolute paths for global configs, local configs, or when data dir is explicitly overridden
        db_rel_path = db_path
        backup_rel_path = str(Path(data_dir) / "backups")
        models_rel_path = str(Path(data_dir) / "models")
        logs_rel_path = str(Path(data_dir) / "logs" / "cashew.log")
    
    # Build the configuration
    config = {
        'database': {
            'path': db_rel_path,
            'backup_dir': backup_rel_path,
            'auto_backup': True
        },
        'models': {
            'embedding': {
                'name': 'all-MiniLM-L6-v2' if embedding_provider == 'local' else 'text-embedding-ada-002',
                'provider': 'sentence-transformers' if embedding_provider == 'local' else 'openai',
                'cache_dir': models_rel_path
            }
        },
        'domains': {
            'default': 'general',
            'user': 'user',
            'ai': 'ai',
            'classifications': domains if use_domains else ['general'],
            'auto_classify': use_domains,
            'separation_enabled': use_domains
        },
        'node_types': {
            'core': {
                'belief': 'a held opinion or conviction',
                'insight': 'a non-obvious connection or pattern discovered',
                'decision': 'a choice made between alternatives',
                'commitment': 'a stated intention or planned action',
                'observation': 'a factual pattern noticed',
                'fact': 'a concrete verifiable fact'
            },
            'custom': {}
        },
        'performance': {
            'token_budget': 2000,
            'top_k_results': 10,
            'walk_depth': 2,
            'similarity_threshold': 0.3,
            'access_weight': 0.2,
            'temporal_weight': 0.1,
            'think_cycle_nodes': 5,
            'clustering_eps': 0.35,
            'novelty_threshold': 0.82,
            'confidence_threshold': 0.7,
            'max_think_iterations': 3
        },
        'integration': {},
        'logging': {
            'level': 'INFO',
            'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            'file': logs_rel_path
        },
        'gc': {
            'mode': 'soft',
            'threshold': 0.05,
            'grace_days': 7,
            'protect_types': ['seed', 'core_memory'],
            'think_cycle_penalty': 1.5
        },
        'features': {
            'auto_extraction': True,
            'think_cycles': True,
            'sleep_cycles': True,
            'decay_pruning': True,
            'pattern_detection': True
        }
    }
    
    # Add sleep cycle schedule based on frequency
    if sleep_frequency == "6h":
        cron_schedule = "0 */6 * * *"  # Every 6 hours
    elif sleep_frequency == "12h":
        cron_schedule = "0 */12 * * *"  # Every 12 hours
    else:
        cron_schedule = None  # Manual
    
    config['sleep'] = {
        'enabled': sleep_frequency != "manual",
        'schedule': cron_schedule,
        'frequency': sleep_frequency
    }
    
    # Store API key hint if provided
    if api_key:
        config['setup'] = {
            'api_key_provided': True,
            'llm_provider': llm_provider
        }
    
    return config, db_path, api_key, data_dir

def write_config_file(config: Dict[str, Any], config_path: str):
    """Write configuration to YAML file"""
    print_info(f"Writing config to {config_path}")
    
    # Ensure parent directory exists
    Path(config_path).parent.mkdir(parents=True, exist_ok=True)
    
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, indent=2, sort_keys=False)
    
    print_success(f"Configuration saved to {config_path}")

def detect_scheduling_backends() -> Dict[str, bool]:
    """Detect available scheduling backends"""
    backends = {}
    
    # Check system crontab
    backends['crontab'] = shutil.which('crontab') is not None
    
    # Manual is always available
    backends['manual'] = True
    
    return backends

def generate_cron_entries(config_path: str, data_dir: str, frequency: str) -> List[str]:
    """Generate cron entries for cashew cycles"""
    entries = []
    
    if frequency == "manual":
        return entries
    
    # Determine frequency pattern
    if frequency == "6h":
        hour_pattern = "*/6"
    elif frequency == "12h":
        hour_pattern = "*/12"
    else:
        hour_pattern = "*/6"  # default fallback
    
    # Use absolute paths
    config_abs_path = Path(config_path).resolve()
    cashew_dir = Path(__file__).parent.parent.resolve()  # cashew project root
    log_dir = Path(data_dir) / "logs"
    
    # Ensure log directory exists
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate cron entries with environment variables and full paths
    sleep_entry = f"0 {hour_pattern} * * * CASHEW_CONFIG_PATH={config_abs_path} cd {cashew_dir} && python3 scripts/cashew_context.py sleep 2>&1 >> {log_dir}/sleep.log"
    think_entry = f"30 {hour_pattern} * * * CASHEW_CONFIG_PATH={config_abs_path} cd {cashew_dir} && python3 scripts/cashew_context.py think 2>&1 >> {log_dir}/think.log"
    
    entries.extend([sleep_entry, think_entry])
    
    return entries

def install_system_crontab(entries: List[str], dry_run: bool = False) -> bool:
    """Install cron entries to system crontab"""
    if not entries:
        return True
    
    if dry_run:
        print_info("Would install these cron entries:")
        for entry in entries:
            print(f"  {entry}")
        return True
    
    try:
        # Read current crontab
        try:
            result = subprocess.run(['crontab', '-l'], capture_output=True, text=True)
            existing_crontab = result.stdout if result.returncode == 0 else ""
        except subprocess.CalledProcessError:
            existing_crontab = ""
        
        # Check for existing cashew entries to avoid duplicates
        existing_lines = existing_crontab.strip().split('\n') if existing_crontab.strip() else []
        cashew_marker = "# cashew"
        
        # Remove old cashew entries
        new_lines = [line for line in existing_lines if cashew_marker not in line]
        
        # Add new entries with marker
        for entry in entries:
            new_lines.append(f"{entry}  {cashew_marker}")
        
        new_crontab = '\n'.join(new_lines) + '\n' if new_lines else ''
        
        # Write new crontab
        process = subprocess.run(['crontab', '-'], input=new_crontab, text=True, capture_output=True)
        if process.returncode != 0:
            print_error(f"Failed to install crontab: {process.stderr}")
            return False
        
        print_success(f"Installed {len(entries)} cron entries")
        return True
        
    except Exception as e:
        print_error(f"Failed to install crontab: {e}")
        return False

def setup_scheduling(config: Dict[str, Any], config_path: str, data_dir: str, interactive: bool = True, dry_run: bool = False) -> bool:
    """Set up scheduling for sleep/think cycles"""
    sleep_config = config.get('sleep', {})
    frequency = sleep_config.get('frequency', 'manual')
    
    if frequency == 'manual':
        if interactive:
            print_info("\nManual mode selected. Run these commands periodically:")
            print("  cashew sleep    # consolidation (recommended: every 6 hours)")
            print("  cashew think    # insight generation (recommended: every 6 hours)")
        return True
    
    # Detect available backends
    backends = detect_scheduling_backends()
    
    if interactive:
        print_header("Schedule Setup")
        print("How should we schedule sleep/think cycles?")
        
        options = []
        if backends['crontab']:
            options.append("system")
            print("  ● system crontab (recommended — works everywhere)")
        
        options.append("manual")
        print("  ○ manual — I'll run `cashew sleep` myself")

        if not options or options == ['manual']:
            print_warning("No automatic scheduling backends available")
            selected_backend = "manual"
        else:
            default_backend = "system" if backends['crontab'] else "manual"
            selected_backend = prompt_with_default(
                "Which scheduling method?",
                default_backend,
                options
            )
    else:
        # Non-interactive: auto-detect best option
        if backends['crontab']:
            selected_backend = "system"
        else:
            selected_backend = "manual"
    
    # Generate cron entries
    entries = generate_cron_entries(config_path, data_dir, frequency)
    
    # Set up scheduling based on selected backend
    success = True
    if selected_backend == "system":
        success = install_system_crontab(entries, dry_run)
    elif selected_backend == "manual":
        print_info("\nManual mode selected. Run these commands periodically:")
        print("  cashew sleep    # consolidation (recommended: every 6 hours)")
        print("  cashew think    # insight generation (recommended: every 6 hours)")
    
    return success

def print_next_steps(data_dir: str, config_path: str, api_key: Optional[str], llm_provider: str):
    """Print next steps for the user"""
    print_header("Setup Complete! 🎉")
    
    print(f"Your cashew brain is ready at: {Colors.BOLD}{data_dir}{Colors.RESET}")
    print(f"Configuration saved to: {Colors.BOLD}{config_path}{Colors.RESET}")
    
    print(f"\n{Colors.BOLD}Next steps:{Colors.RESET}")
    
    if api_key:
        if llm_provider == "anthropic":
            print(f"1. {Colors.GREEN}Export your API key:{Colors.RESET}")
            print(f"   export ANTHROPIC_API_KEY='{api_key}'")
        else:
            print(f"1. {Colors.GREEN}Export your API key:{Colors.RESET}")
            print(f"   export OPENAI_API_KEY='{api_key}'")
    else:
        if llm_provider in ["anthropic", "openai"]:
            key_name = "ANTHROPIC_API_KEY" if llm_provider == "anthropic" else "OPENAI_API_KEY"
            print(f"1. {Colors.YELLOW}Set your API key:{Colors.RESET}")
            print(f"   export {key_name}='your-key-here'")
    
    print(f"\n2. {Colors.GREEN}Try your first extraction:{Colors.RESET}")
    print(f"   python scripts/cashew_context.py extract --input 'I learned that SQLite is perfect for local AI apps'")
    
    print(f"\n3. {Colors.GREEN}Search your knowledge:{Colors.RESET}")
    print(f"   python scripts/cashew_context.py context --hints 'databases AI'")
    
    print(f"\n4. {Colors.GREEN}Run a think cycle:{Colors.RESET}")
    print(f"   python scripts/cashew_context.py think")
    
    print(f"\n📖 {Colors.BLUE}Learn more at:{Colors.RESET} https://github.com/jugaad-lab/cashew")

def main():
    """Main setup function"""
    parser = argparse.ArgumentParser(description="Cashew Interactive Setup")
    parser.add_argument('--non-interactive', action='store_true', 
                       help='Use all defaults without prompting')
    parser.add_argument('--config-path', default=None,
                       help='Where to write config file (default: ./config.yaml or ~/.cashew/config.yaml)')
    parser.add_argument('--global', dest='global_config', action='store_true',
                       help='Install config globally at ~/.cashew/config.yaml')
    parser.add_argument('--data-dir', default=None,
                       help='Explicit data directory override (overrides location-based defaults)')
    
    args = parser.parse_args()
    
    # Determine config path
    if args.config_path:
        config_path = args.config_path
    elif args.global_config:
        config_path = str(Path.home() / ".cashew" / "config.yaml")
    else:
        # Default: project root if we're in a cashew project, otherwise global
        if Path("core/config.py").exists():
            config_path = "./config.yaml"
        else:
            config_path = str(Path.home() / ".cashew" / "config.yaml")
    
    try:
        # Build configuration
        config, db_path, api_key, data_dir = build_config(
            interactive=not args.non_interactive,
            config_path=config_path,
            global_config=args.global_config,
            data_dir_override=args.data_dir
        )
        
        # Create directories
        Path(data_dir).mkdir(parents=True, exist_ok=True)
        Path(data_dir, "logs").mkdir(exist_ok=True)
        Path(data_dir, "backups").mkdir(exist_ok=True)
        Path(data_dir, "models").mkdir(exist_ok=True)
        
        # Create database with schema (use absolute path for creation)
        create_database_schema(str(Path(data_dir) / "graph.db"))
        
        # Write configuration
        write_config_file(config, config_path)
        
        # Set up scheduling
        setup_success = setup_scheduling(
            config, 
            config_path, 
            data_dir, 
            interactive=not args.non_interactive,
            dry_run=False
        )
        
        if not setup_success:
            print_warning("Scheduling setup failed, but cashew is still functional")
        
        if not args.non_interactive:
            print_next_steps(data_dir, config_path, api_key, config.get('setup', {}).get('llm_provider', 'unknown'))
        else:
            print_success(f"Cashew initialized with defaults at {data_dir}")
            
    except KeyboardInterrupt:
        print_warning("\nSetup cancelled by user")
        sys.exit(1)
    except Exception as e:
        print_error(f"Setup failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()