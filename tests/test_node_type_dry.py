"""Guard against drift between core.config node types and the extractor
prompts/regex that consume them.

Background: scripts/cashew_context.py once hand-rolled its extraction
prompt with a hardcoded list ("fact, observation, insight, decision,
belief") that omitted "commitment" even though the type was defined in
config. Result: zero commitment-typed nodes for months. These tests fail
loudly if anything reintroduces that drift.
"""

import re

from core.config import config
from extractors.utils import TYPED_STATEMENT_RE, TYPE_TAGGING_INSTRUCTION


def test_typed_statement_re_matches_every_configured_type():
    """The extractor regex must accept every type in core.config."""
    for type_name in config._node_type_map.keys():
        sample = f"[{type_name}] some statement text"
        m = TYPED_STATEMENT_RE.match(sample)
        assert m is not None, f"regex rejected configured type {type_name!r}"
        assert m.group(1).lower() == type_name


def test_type_tagging_instruction_mentions_every_configured_type():
    """The prompt fragment must list every type so the LLM can use it."""
    for type_name in config._node_type_map.keys():
        token = f"[{type_name}]"
        assert token in TYPE_TAGGING_INSTRUCTION, (
            f"TYPE_TAGGING_INSTRUCTION omits {token}; "
            f"add a description and rebuild the prompt fragment from config."
        )


def test_node_type_prompt_fragment_includes_commitment_when_configured():
    """commitment is a first-class type. If config has it, the helper must
    surface it. Guards against regressions in config.node_type_prompt_fragment."""
    if "commitment" not in config._node_type_map:
        return  # config genuinely doesn't define it, nothing to guard
    fragment = config.node_type_prompt_fragment
    assert '"commitment"' in fragment


def test_validate_node_type_accepts_commitment_when_configured():
    if "commitment" not in config._node_type_map:
        return
    assert config.validate_node_type("commitment") == "commitment"


def test_migration_extraction_prompt_uses_config_helper():
    """scripts/cashew_context.py's migration prompt should not hardcode the
    type list. Specifically, the historical drift string must be gone."""
    from pathlib import Path
    src = Path(__file__).parent.parent / "scripts" / "cashew_context.py"
    text = src.read_text()
    # Old hand-rolled list that dropped commitment
    assert 'one of "fact", "observation", "insight", "decision", "belief"' not in text, (
        "scripts/cashew_context.py reintroduced the hand-rolled type list. "
        "Use config.node_type_prompt_fragment instead."
    )
    # New helper must be referenced
    assert "node_type_prompt_fragment" in text, (
        "scripts/cashew_context.py extraction prompt should reference "
        "config.node_type_prompt_fragment."
    )
