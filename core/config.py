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
        self.config_path = config_path or self._find_config_file()
        self._raw_config = {}
        self._load_config()
    
    def _find_config_file(self) -> Optional[str]:
        """Find config.yaml in current directory or parent directories"""
        current = Path.cwd()
        for path in [current] + list(current.parents):
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
                    'decision': 'a commitment or choice made',
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
                'confidence_threshold': 0.7,
                'max_think_iterations': 3
            },
            'integration': {
                'openclaw': {
                    'auth_profile_path': '${HOME}/.openclaw/agents/${OPENCLAW_AGENT:-main}/agent/auth-profiles.json',
                    'workspace_path': '${HOME}/.openclaw/workspace',
                    'session_dir': '${HOME}/.openclaw/sessions'
                }
            },
            'logging': {
                'level': 'INFO',
                'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                'file': './logs/cashew.log'
            },
            'node_types': {
                'core': [
                    {'belief': 'a held opinion or conviction'},
                    {'insight': 'a non-obvious connection or pattern discovered'},
                    {'decision': 'a commitment or choice made'},
                    {'observation': 'a factual pattern noticed'},
                    {'fact': 'a concrete verifiable fact'},
                ],
            },
            'features': {
                'auto_extraction': True,
                'think_cycles': True,
                'sleep_cycles': True,
                'decay_pruning': True,
                'hotspot_generation': True,
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
        # Database configuration
        self.db_path = self._raw_config['database']['path']
        self.backup_dir = self._raw_config['database']['backup_dir']
        self.auto_backup = self._raw_config['database']['auto_backup']
        
        # Performance configuration
        perf = self._raw_config['performance']
        self.token_budget = int(perf['token_budget'])
        self.top_k = int(perf['top_k_results'])
        self.walk_depth = int(perf['walk_depth'])
        self.similarity_threshold = float(perf['similarity_threshold'])
        self.access_weight = float(perf['access_weight'])
        self.temporal_weight = float(perf['temporal_weight'])
        self.think_cycle_nodes = int(perf['think_cycle_nodes'])
        self.clustering_eps = float(perf.get('clustering_eps', 0.35))
        self.novelty_threshold = float(perf.get('novelty_threshold', 0.82))
        self.confidence_threshold = float(perf.get('confidence_threshold', 0.7))
        
        # Model configuration
        self.embedding_model = self._raw_config['models']['embedding']['name']
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
        self._system_types = {'hotspot', 'dream', 'tension', 'core_memory', 'derived'}
        self.node_type_names = set(self._node_type_map.keys()) | self._system_types
        
        # Integration configuration
        openclaw = self._raw_config['integration']['openclaw']
        self.auth_profile_path = openclaw['auth_profile_path']
        self.workspace_path = openclaw['workspace_path']
        self.session_dir = openclaw.get('session_dir', '${HOME}/.openclaw/sessions')

        # (node type taxonomy loaded above via _node_type_map)
    
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

def reload_config(config_path: Optional[str] = None):
    """Reload configuration from file"""
    global config
    config = CashewConfig(config_path)

if __name__ == "__main__":
    import json
    print("Cashew Configuration:")
    print(json.dumps(config.to_dict(), indent=2))