#!/usr/bin/env python3
"""
Cashew Configuration Module
Centralized configuration with YAML file and environment variable overrides
"""

import os
import yaml
import re
from typing import Optional, Dict, Any
from pathlib import Path

# Default configuration values (fallbacks if no config file)
DEFAULT_TOKEN_BUDGET = 2000
DEFAULT_DB_PATH = "./data/graph.db"
DEFAULT_TOP_K = 10
DEFAULT_WALK_DEPTH = 2
DEFAULT_EMBEDDING_MODEL = 'all-MiniLM-L6-v2'
DEFAULT_THINK_CYCLE_NODES = 5
DEFAULT_ACCESS_WEIGHT = 0.2
DEFAULT_TEMPORAL_WEIGHT = 0.1
DEFAULT_SIMILARITY_THRESHOLD = 0.3
DEFAULT_USER_DOMAIN = "user"
DEFAULT_AI_DOMAIN = "ai"

def _expand_env_vars(value: Any) -> Any:
    """Recursively expand environment variables in configuration values"""
    if isinstance(value, str):
        # Handle ${VAR:-default} syntax
        def replace_env_var(match):
            env_expr = match.group(1)
            if ':-' in env_expr:
                var_name, default_value = env_expr.split(':-', 1)
                return os.getenv(var_name, default_value)
            else:
                return os.getenv(env_expr, '')
        
        # Expand ${VAR} and ${VAR:-default} patterns
        value = re.sub(r'\$\{([^}]+)\}', replace_env_var, value)
        
    elif isinstance(value, dict):
        return {k: _expand_env_vars(v) for k, v in value.items()}
    elif isinstance(value, list):
        return [_expand_env_vars(item) for item in value]
    
    return value


class CashewConfig:
    """Configuration class with YAML file and environment variable support"""
    
    def __init__(self, config_path: Optional[str] = None):
        # Check environment variables for config overrides
        env_config_path = os.environ.get('CASHEW_CONFIG_PATH')
        self.config_path = config_path or env_config_path or self._find_config_file()
        self._raw_config = {}
        self._load_config()
    
    def _find_config_file(self) -> Optional[str]:
        """Find config.yaml in order: ./config.yaml → ~/.cashew/config.yaml → parent directories"""
        # First try current directory
        local_config = Path.cwd() / "config.yaml"
        if local_config.exists():
            return str(local_config)
        
        # Then try global config
        global_config = Path.home() / ".cashew" / "config.yaml"
        if global_config.exists():
            return str(global_config)
        
        # Finally try parent directories (backward compatibility)
        current = Path.cwd()
        for path in list(current.parents):
            config_file = path / "config.yaml"
            if config_file.exists():
                return str(config_file)
        
        return None
    
    def _load_config(self):
        """Load configuration from YAML file with environment variable expansion"""
        # Start with defaults
        config_data = self._get_default_config()
        
        # Load from YAML file if available
        if self.config_path and Path(self.config_path).exists():
            try:
                with open(self.config_path, 'r') as f:
                    file_config = yaml.safe_load(f) or {}
                    # Deep merge file config with defaults
                    config_data = self._deep_merge(config_data, file_config)
            except (yaml.YAMLError, IOError) as e:
                print(f"Warning: Error loading config file {self.config_path}: {e}")
        
        # Expand environment variables
        self._raw_config = _expand_env_vars(config_data)
        
        # Extract commonly used values for convenience
        self._extract_config_values()
        
        # Ensure paths are absolute and expanded
        self._expand_paths()
        
        # Validation
        self._validate_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration structure"""
        return {
            'database': {
                'path': DEFAULT_DB_PATH,
                'backup_dir': './data/backups',
                'auto_backup': True
            },
            'models': {
                'embedding': {
                    'name': DEFAULT_EMBEDDING_MODEL,
                    'provider': 'sentence-transformers',
                    'cache_dir': './models'
                },
                # LLM section removed - cashew doesn't call LLMs directly
                # LLM access is provided by the orchestrator via model_fn parameters
            },
            'domains': {
                'default': 'general',
                'user': DEFAULT_USER_DOMAIN,
                'ai': DEFAULT_AI_DOMAIN,
                'classifications': ['personal', 'work', 'projects', 'learning', 'system'],
                'auto_classify': True
            },
            'node_types': {
                'core': {
                    'belief': 'a held opinion or conviction',
                    'insight': 'a non-obvious connection or pattern discovered',
                    'decision': 'a choice made between alternatives',
                    'commitment': 'a stated intention or planned action',
                    'observation': 'a factual pattern noticed',
                    'fact': 'a concrete verifiable fact',
                },
                'custom': {},
            },
            'performance': {
                'token_budget': DEFAULT_TOKEN_BUDGET,
                'top_k_results': DEFAULT_TOP_K,
                'walk_depth': DEFAULT_WALK_DEPTH,
                'similarity_threshold': DEFAULT_SIMILARITY_THRESHOLD,
                'access_weight': DEFAULT_ACCESS_WEIGHT,
                'temporal_weight': DEFAULT_TEMPORAL_WEIGHT,
                'think_cycle_nodes': DEFAULT_THINK_CYCLE_NODES,
                'clustering_eps': 0.35,
                'novelty_threshold': 0.82,
                'max_think_iterations': 3
            },
            'integration': {},
            'logging': {
                'level': 'INFO',
                'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                'file': './logs/cashew.log'
            },
            'node_types': {
                'core': [
                    {'belief': 'a held opinion or conviction'},
                    {'insight': 'a non-obvious connection or pattern discovered'},
                    {'decision': 'a choice made between alternatives'},
                    {'commitment': 'a stated intention or planned action'},
                    {'observation': 'a factual pattern noticed'},
                    {'fact': 'a concrete verifiable fact'},
                ],
            },
            'gc': {
                'mode': 'soft',           # soft | hard | off
                'threshold': 0.05,        # relevance score below which nodes are eligible
                'grace_days': 7,          # days since last_accessed before eligible
                
                'protect_types': ['seed', 'core_memory'],
                'think_cycle_penalty': 1.5,  # multiplier on threshold for think-cycle nodes
            },
            'sleep': {
                'enabled': True,
                'frequency': '6h',
                'schedule': '0 */6 * * *'
            },
            'features': {
                'auto_extraction': True,
                'think_cycles': True,
                'sleep_cycles': True,
                'decay_pruning': True,
                'pattern_detection': True
            }
        }
    
    def _deep_merge(self, base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """Deep merge two dictionaries, with override taking precedence"""
        result = base.copy()
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value
        return result
    
    def _extract_config_values(self):
        """Extract commonly used config values for convenience"""
        # Database configuration (environment variable overrides config file)
        self.db_path = os.environ.get('CASHEW_DB_PATH') or self._raw_config['database']['path']
        self.backup_dir = self._raw_config['database']['backup_dir']
        self.auto_backup = self._raw_config['database']['auto_backup']
        
        # Performance configuration (env vars override YAML)
        perf = self._raw_config['performance']
        self.token_budget = int(os.environ.get('CASHEW_TOKEN_BUDGET', perf['token_budget']))
        self.top_k = int(os.environ.get('CASHEW_TOP_K', perf['top_k_results']))
        self.walk_depth = int(os.environ.get('CASHEW_WALK_DEPTH', perf['walk_depth']))
        self.similarity_threshold = float(perf['similarity_threshold'])
        self.access_weight = float(os.environ.get('CASHEW_ACCESS_WEIGHT', perf['access_weight']))
        self.temporal_weight = float(os.environ.get('CASHEW_TEMPORAL_WEIGHT', perf['temporal_weight']))
        self.think_cycle_nodes = int(perf['think_cycle_nodes'])
        self.clustering_eps = float(perf.get('clustering_eps', 0.35))
        self.novelty_threshold = float(perf.get('novelty_threshold', 0.82))
        
        # Model configuration (env var override)
        self.embedding_model = os.environ.get('CASHEW_EMBEDDING_MODEL', self._raw_config['models']['embedding']['name'])
        # LLM properties removed - cashew doesn't call LLMs directly
        
        # Domain configuration
        domains = self._raw_config['domains']
        self.default_domain = domains['default']
        self.user_domain = domains['user']
        self.ai_domain = domains['ai']
        self.domain_classifications = domains['classifications']
        
        # Node type taxonomy
        nt_config = self._raw_config.get('node_types', {})
        core_types = nt_config.get('core', {}) or {}
        custom_types = nt_config.get('custom', {}) or {}
        # Handle YAML list-of-dicts format: [{k: v}, ...] -> {k: v, ...}
        if isinstance(core_types, list):
            merged = {}
            for item in core_types:
                if isinstance(item, dict):
                    merged.update(item)
            core_types = merged
        if isinstance(custom_types, list):
            merged = {}
            for item in custom_types:
                if isinstance(item, dict):
                    merged.update(item)
            custom_types = merged
        # Merge: custom can override core descriptions
        self._node_type_map = {**core_types, **custom_types}
        # System types are always valid but not in extraction prompts
        self._system_types = {'dream', 'tension', 'core_memory', 'derived'}
        self.node_type_names = set(self._node_type_map.keys()) | self._system_types
        
        # GC configuration
        gc = self._raw_config.get('gc', {})
        self.gc_mode = gc.get('mode', 'soft')
        self.gc_threshold = float(gc.get('threshold', 0.05))
        self.gc_grace_days = int(gc.get('grace_days', 7))
        self.gc_protect_types = list(gc.get('protect_types', ['seed', 'core_memory']))
        self.gc_think_cycle_penalty = float(gc.get('think_cycle_penalty', 1.5))

        # (node type taxonomy loaded above via _node_type_map)
        
        # Think configuration
        think_config = self._raw_config.get('think', {})
        self.think_enabled = think_config.get('enabled', True)
        self.think_frequency = think_config.get('frequency', '12h')
        self.think_schedule = think_config.get('schedule', '0 5,17 * * *')
        
        # Sleep configuration
        sleep_config = self._raw_config.get('sleep', {})
        self.sleep_enabled = sleep_config.get('enabled', True)
        self.sleep_frequency = sleep_config.get('frequency', '6h')
        self.sleep_schedule = sleep_config.get('schedule', '0 */6 * * *')
        
        # Extract configuration
        extract_config = self._raw_config.get('extract', {})
        self.extract_enabled = extract_config.get('enabled', True)
        self.extract_frequency = extract_config.get('frequency', '2h')
        self.extract_schedule = extract_config.get('schedule', '0 */2 * * *')
        
        # Backup configuration
        backup_config = self._raw_config.get('backup', {})
        self.backup_schedule = backup_config.get('schedule', '0 */6 * * *')
        self.backup_frequency = backup_config.get('frequency', '6h')
        
        # Feature flags
        features = self._raw_config.get('features', {})
        self.auto_extraction = features.get('auto_extraction', True)
        self.think_cycles = features.get('think_cycles', True)
        self.sleep_cycles = features.get('sleep_cycles', True)
        self.decay_pruning = features.get('decay_pruning', True)
        self.pattern_detection = features.get('pattern_detection', True)
    
    def _expand_paths(self):
        """Expand relative paths to absolute paths, relative to config file location"""
        # Determine base directory for relative path resolution
        if self.config_path:
            base_dir = Path(self.config_path).parent
        else:
            base_dir = Path.cwd()
        
        # Make database path absolute
        if not Path(self.db_path).is_absolute():
            self.db_path = str((base_dir / self.db_path).resolve())
        
        # Make backup dir absolute
        if not Path(self.backup_dir).is_absolute():
            self.backup_dir = str((base_dir / self.backup_dir).resolve())
        
        # Expand other paths in the config
        models = self._raw_config.get('models', {})
        embedding = models.get('embedding', {})
        cache_dir = embedding.get('cache_dir', './models')
        if not Path(cache_dir).is_absolute():
            embedding['cache_dir'] = str((base_dir / cache_dir).resolve())
        
        # Expand log file path
        logging_config = self._raw_config.get('logging', {})
        log_file = logging_config.get('file', './logs/cashew.log')
        if not Path(log_file).is_absolute():
            expanded_log_path = str((base_dir / log_file).resolve())
            logging_config['file'] = expanded_log_path
            # Also update the extracted value
            Path(expanded_log_path).parent.mkdir(parents=True, exist_ok=True)
    
    def _validate_config(self):
        """Validate configuration values"""
        if self.token_budget <= 0:
            raise ValueError(f"Token budget must be positive, got {self.token_budget}")
        
        if self.top_k <= 0:
            raise ValueError(f"Top K must be positive, got {self.top_k}")
        
        if self.walk_depth < 0:
            raise ValueError(f"Walk depth must be non-negative, got {self.walk_depth}")
        
        if self.think_cycle_nodes <= 0:
            raise ValueError(f"Think cycle nodes must be positive, got {self.think_cycle_nodes}")
        
        if not 0 <= self.similarity_threshold <= 1:
            raise ValueError(f"Similarity threshold must be in [0,1], got {self.similarity_threshold}")
        
        # Validate scoring weights
        if self.access_weight + self.temporal_weight > 1.0:
            raise ValueError(f"Access weight ({self.access_weight}) + temporal weight ({self.temporal_weight}) "
                           f"must not exceed 1.0")
    
    def get_scoring_weights(self) -> dict:
        """Get the current scoring weights for hybrid retrieval"""
        # Calculate embedding weight as remainder to ensure weights sum to 1.0
        embedding_weight = 1.0 - self.access_weight - self.temporal_weight
        
        return {
            'embedding': embedding_weight,
            'access': self.access_weight,
            'temporal': self.temporal_weight
        }
    
    @property
    def node_type_prompt_fragment(self) -> str:
        """Generate the type classification block for LLM extraction prompts."""
        lines = []
        for name, desc in self._node_type_map.items():
            lines.append(f'- "{name}": {desc}')
        return '\n'.join(lines)

    @property
    def node_type_pipe_list(self) -> str:
        """Pipe-separated list of valid extraction types for JSON examples."""
        return '|'.join(self._node_type_map.keys())

    def validate_node_type(self, node_type: str) -> str:
        """Validate and return a node type, falling back to 'observation' if invalid."""
        if node_type in self.node_type_names:
            return node_type
        return 'observation'

    def get_raw_config(self) -> Dict[str, Any]:
        """Get the raw configuration dictionary"""
        return self._raw_config
    
    def get_domain_mapping(self) -> Dict[str, str]:
        """Get domain mapping for backward compatibility"""
        return {
            'raj': self.user_domain,    # Map old 'raj' domain to user domain
            'bunny': self.ai_domain,    # Map old 'bunny' domain to AI domain
            'user': self.user_domain,   # Standard mapping
            'ai': self.ai_domain        # Standard mapping
        }
    
    def map_domain(self, domain: str) -> str:
        """Map a domain name using the current configuration"""
        mapping = self.get_domain_mapping()
        return mapping.get(domain, domain)
    
    def to_dict(self) -> dict:
        """Export configuration as dictionary"""
        return {
            'db_path': self.db_path,
            'token_budget': self.token_budget,
            'top_k': self.top_k,
            'walk_depth': self.walk_depth,
            'embedding_model': self.embedding_model,
            'think_cycle_nodes': self.think_cycle_nodes,
            'access_weight': self.access_weight,
            'temporal_weight': self.temporal_weight,
            'similarity_threshold': self.similarity_threshold,
            'user_domain': self.user_domain,
            'ai_domain': self.ai_domain,
            'scoring_weights': self.get_scoring_weights()
        }
    
    def __repr__(self) -> str:
        """String representation of configuration"""
        return f"CashewConfig({self.to_dict()})"

# Global configuration instance
config = CashewConfig()

# Convenience functions for accessing config values
def get_db_path() -> str:
    """Get the database path"""
    return config.db_path

def get_token_budget() -> int:
    """Get the current token budget for context injection"""
    return config.token_budget

def get_top_k() -> int:
    """Get the number of top results to retrieve"""
    return config.top_k

def get_walk_depth() -> int:
    """Get the graph walk depth for context expansion"""
    return config.walk_depth

def get_embedding_model() -> str:
    """Get the embedding model identifier"""
    return config.embedding_model

def get_think_cycle_nodes() -> int:
    """Get the number of nodes to use in think cycles"""
    return config.think_cycle_nodes

def get_scoring_weights() -> dict:
    """Get the current scoring weights for hybrid retrieval"""
    return config.get_scoring_weights()

def get_user_domain() -> str:
    """Get the user domain name (replaces 'raj')"""
    return config.user_domain

def get_ai_domain() -> str:
    """Get the AI domain name (replaces 'bunny')"""
    return config.ai_domain

def map_domain(domain: str) -> str:
    """Map a domain name using the current configuration"""
    return config.map_domain(domain)

def get_gc_config() -> dict:
    """Get the GC policy configuration"""
    return {
        'mode': config.gc_mode,
        'threshold': config.gc_threshold,
        'grace_days': config.gc_grace_days,
        'protect_types': config.gc_protect_types,
        'think_cycle_penalty': config.gc_think_cycle_penalty,
    }


def get_think_config() -> dict:
    """Get the think cycle configuration"""
    return {
        'enabled': config.think_enabled,
        'frequency': config.think_frequency,
        'schedule': config.think_schedule
    }

def get_sleep_config() -> dict:
    """Get the sleep cycle configuration"""
    return {
        'enabled': config.sleep_enabled,
        'frequency': config.sleep_frequency,
        'schedule': config.sleep_schedule
    }

def get_extract_config() -> dict:
    """Get the extraction schedule configuration"""
    return {
        'enabled': config.extract_enabled,
        'frequency': config.extract_frequency,
        'schedule': config.extract_schedule
    }

def get_backup_config() -> dict:
    """Get the backup schedule configuration"""
    return {
        'frequency': config.backup_frequency,
        'schedule': config.backup_schedule
    }

def get_feature_flags() -> dict:
    """Get the feature flags configuration"""
    return {
        'auto_extraction': config.auto_extraction,
        'think_cycles': config.think_cycles,
        'sleep_cycles': config.sleep_cycles,
        'decay_pruning': config.decay_pruning,
        'pattern_detection': config.pattern_detection
    }

def is_feature_enabled(feature_name: str) -> bool:
    """Check if a feature is enabled"""
    return getattr(config, feature_name, False)

def reload_config(config_path: Optional[str] = None):
    """Reload configuration from file"""
    global config
    config = CashewConfig(config_path)

if __name__ == "__main__":
    import json
    print("Cashew Configuration:")
    print(json.dumps(config.to_dict(), indent=2))