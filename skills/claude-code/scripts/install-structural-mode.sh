#!/usr/bin/env bash
# Install cashew "structural mode": wire the context-injection hook into Claude
# Code so the brain is queried on every substantive prompt automatically, instead
# of relying on the model to remember the CLAUDE.md "query first" instruction.
#
# Usage: install-structural-mode.sh [path/to/.claude/settings.json]
#   defaults to ~/.claude/settings.json (project-local: pass .claude/settings.json)
#
# Idempotent and non-destructive: merges into existing hooks, never clobbers.
set -euo pipefail

HOOK="$(cd "$(dirname "${BASH_SOURCE[0]}")/../hooks" && pwd)/inject_context.py"
SETTINGS="${1:-$HOME/.claude/settings.json}"

python3 - "$SETTINGS" "$HOOK" <<'PY'
import json, os, sys
settings_path, hook = sys.argv[1], sys.argv[2]
os.makedirs(os.path.dirname(os.path.abspath(settings_path)), exist_ok=True)
try:
    with open(settings_path) as f:
        data = json.load(f)
except (FileNotFoundError, json.JSONDecodeError):
    data = {}

cmd = f"python3 {hook}"
ups = data.setdefault("hooks", {}).setdefault("UserPromptSubmit", [])
if any(h.get("command") == cmd for entry in ups for h in entry.get("hooks", [])):
    print("cashew structural mode already installed — nothing to do")
    sys.exit(0)

ups.append({"matcher": "", "hooks": [{"type": "command", "command": cmd}]})
with open(settings_path, "w") as f:
    json.dump(data, f, indent=2)
print(f"✅ installed cashew UserPromptSubmit hook")
print(f"   hook:     {hook}")
print(f"   settings: {settings_path}")
PY

echo
echo "Structural mode is on. Make sure:"
echo "  • CASHEW_DB points at your brain (e.g. export CASHEW_DB=~/.cashew/graph.db)"
echo "  • optional but recommended: run 'cashew serve' so injection stays ~sub-second"
echo "  • tune the relevance gate with CASHEW_HOOK_RELEVANCE (default 0.83)"
