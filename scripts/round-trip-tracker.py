#!/usr/bin/env python3
"""
Round-trip tracker: measures task completion rate per exchange.

Reads session history JSON (piped from sessions_history), identifies task chains,
counts exchanges per task, checks if brain context was used, and appends to a
tracking log.

Usage (from cron/manual):
  # Feed session history JSON via stdin
  cat /tmp/session-history.json | python3 scripts/round-trip-tracker.py

  # Or just analyze the latest log
  python3 scripts/round-trip-tracker.py --report
  
  # Process from JSONL file directly
  python3 scripts/round-trip-tracker.py --from-jsonl path/to/session.jsonl
"""

import json
import sys
import re
from datetime import datetime, timezone, timedelta
from pathlib import Path
from collections import defaultdict

TRACKING_LOG = Path(__file__).parent.parent / "cashew" / "data" / "round-trip-log.jsonl"
REPORT_LAST_N = 50

# Signals that brain context was pulled (updated for toolCall structure)
BRAIN_SIGNALS = [
    "cashew_context.py context",
    "cashew_context.py retrieve", 
    "context --hints",
    "memory_search",
    "GRAPH OVERVIEW",
    "RELEVANT CONTEXT",
]

# Signals a clarification was needed
CLARIFICATION_SIGNALS = [
    "can you clarify", "what do you mean", "which one", "do you want",
    "could you specify", "are you referring", "want me to", "should i",
    "did you mean", "are you talking about",
]


def extract_text(content):
    """Extract plain text from message content, handling both string and list-of-objects formats."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                # Handle both OpenClaw format (toolCall/toolResult) and Claude Code format (tool_use/tool_result)
                if item.get("type") == "text":
                    parts.append(item.get("text", ""))
                elif item.get("type") in ("toolCall", "tool_use"):
                    # Include tool call info for brain signal detection
                    # OpenClaw: name + arguments; Claude Code: name + input
                    tool_name = item.get("name", "")
                    args = item.get("input", item.get("arguments", {}))
                    if isinstance(args, dict):
                        # Serialize arguments for pattern matching
                        args_str = json.dumps(args, separators=(',', ':'))
                        parts.append(f"{tool_name} {args_str}")
                    elif isinstance(args, str):
                        parts.append(f"{tool_name} {args}")
                elif item.get("type") in ("toolResult", "tool_result"):
                    # Include tool results for brain context detection
                    content_part = item.get("content", "")
                    if isinstance(content_part, list):
                        for sub_item in content_part:
                            if isinstance(sub_item, dict) and sub_item.get("type") == "text":
                                parts.append(sub_item.get("text", ""))
                    elif isinstance(content_part, str):
                        parts.append(content_part)
            elif isinstance(item, str):
                parts.append(item)
        return " ".join(parts)
    return str(content)


def is_user_message(msg):
    return msg.get("role") == "user"


def is_assistant_message(msg):
    return msg.get("role") == "assistant"


def is_system_message(msg):
    return msg.get("role") == "system"


# Signals specific to the SessionStart startup hook injection.
STARTUP_HOOK_SIGNALS = [
    "Brain Context (auto-loaded on init)",
    "Context generated successfully",
]


def has_brain_context(messages, start_idx, end_idx, session_brain_preamble=False):
    """Check if brain was queried in this exchange window.

    session_brain_preamble: True if the startup hook fired before the first
    chain started (i.e. brain context was injected at session level and applies
    to all chains).
    """
    if session_brain_preamble:
        return True
    for msg in messages[start_idx:end_idx]:
        content = extract_text(msg.get("content", ""))
        if any(sig in content for sig in BRAIN_SIGNALS):
            return True
    return False


def detect_session_brain_preamble(messages, first_chain_idx):
    """Return True if the startup hook injected brain context before the first chain.

    Hook messages use role="hook" so chain logic skips them, but we scan them
    here to detect session-level brain context injection.
    """
    for msg in messages[:first_chain_idx + 1]:
        if msg.get("role") not in ("hook", "user", "assistant", "system"):
            continue
        content = extract_text(msg.get("content", ""))
        if any(sig in content for sig in STARTUP_HOOK_SIGNALS + BRAIN_SIGNALS):
            return True
    return False


def is_cron_or_system_noise(content):
    """Filter out cron job notifications, system events, and heartbeats."""
    content_lower = content.strip().lower()
    
    # Cron job notifications
    if re.search(r'\[.*\] a cron job ".*" just (completed|started|failed)', content_lower):
        return True
    
    # System event messages
    if content.startswith('[') and '] ' in content[:50]:  # Timestamp prefix
        if any(keyword in content_lower for keyword in [
            'cron job', 'compaction', 'heartbeat', 'system:', 'session started'
        ]):
            return True
    
    return False


def get_user_text(msg):
    """Extract the actual user text, stripping metadata wrappers."""
    content = extract_text(msg.get("content", ""))
    
    # Skip cron/system noise entirely
    if is_cron_or_system_noise(content):
        return ""
    
    # Strip OpenClaw conversation_label metadata blocks
    # Pattern: 'Conversation info (untrusted metadata):\n```json\n{...}\n```\n\nActual message'
    content = re.sub(
        r'Conversation info \(untrusted metadata\):\s*```json\s*\{[^}]*\}\s*```\s*',
        '', content, flags=re.DOTALL
    ).strip()
    
    # Strip system event prefixes
    content = re.sub(r'^System:\s*\[.*?\]\s*', '', content).strip()
    
    return content


def is_substantive_request(content):
    """Filter out greetings, heartbeats, acknowledgments, and system noise."""
    content_lower = content.strip().lower()
    
    # Empty after noise filtering
    if not content or len(content_lower) < 10:
        return False
    
    # Heartbeat prompts
    if "read heartbeat.md" in content_lower:
        return False
    if "heartbeat_ok" in content_lower:
        return False
    
    # Compaction/system messages
    if content_lower.startswith("before compaction"):
        return False
    if content_lower == "compaction":
        return False
    
    # Pure acknowledgments
    skip_exact = [
        r"^(hi|hey|hello|yo|sup|thanks|thank you|ok|okay|k|yep|yes|no|nah|cool|nice|great|got it|sounds good)\.?!?$",
    ]
    for pat in skip_exact:
        if re.match(pat, content_lower):
            return False
    
    # Media-only messages with no text
    if content_lower.startswith("[media attached") and len(content_lower.split("\n")[-1].strip()) < 10:
        return False
    
    return True


def is_noise_response(content):
    """Filter out assistant responses that aren't real task completions."""
    content_lower = content.strip().lower()
    if content_lower in ("no_reply", "heartbeat_ok"):
        return True
    return False


def estimate_tokens(text):
    """Rough token estimate: ~4 chars per token for English text."""
    return max(1, len(text) // 4)


def extract_task_chains(messages):
    """
    Identify task chains from message history.
    A task chain starts with a substantive user message and ends when
    the next substantive user message begins (or messages end).
    
    Returns list of:
    {
        "task_summary": first 120 chars of user request,
        "exchanges": number of user<>assistant rounds,
        "brain_used": bool,
        "clarifications_needed": count of clarifying questions from assistant,
        "first_try_success": bool (1 exchange, no clarifications),
        "timestamp": ISO timestamp of first message,
        "tokens_estimate": estimated total tokens in this task chain,
        "tokens_per_exchange": estimated tokens per exchange,
        "user_tokens": estimated user tokens,
        "assistant_tokens": estimated assistant tokens,
    }
    """
    chains = []
    current_chain_start = None
    current_exchanges = 0
    current_clarifications = 0
    current_task = ""
    current_timestamp = ""
    current_user_tokens = 0
    current_assistant_tokens = 0
    chain_start_idx = 0

    # Find the first substantive user message index so we can check the preamble.
    first_chain_idx = next(
        (i for i, m in enumerate(messages)
         if is_user_message(m) and is_substantive_request(get_user_text(m))),
        len(messages),
    )
    session_brain_preamble = detect_session_brain_preamble(messages, first_chain_idx)

    for i, msg in enumerate(messages):
        # Skip system messages entirely
        if is_system_message(msg):
            continue
        
        if is_user_message(msg):
            user_text = get_user_text(msg)
            
            if not is_substantive_request(user_text):
                continue
            
            # Close existing chain if we have one
            if current_chain_start is not None and current_exchanges > 0:
                total_tokens = current_user_tokens + current_assistant_tokens
                tpe = total_tokens // max(1, current_exchanges)
                
                # Filter out cron noise in task summaries
                if not current_task.startswith("[cron:"):
                    chains.append({
                        "task_summary": current_task[:120],
                        "exchanges": current_exchanges,
                        "brain_used": has_brain_context(messages, chain_start_idx, i, session_brain_preamble),
                        "clarifications_needed": current_clarifications,
                        "first_try_success": current_exchanges == 1 and current_clarifications == 0,
                        "timestamp": current_timestamp,
                        "tokens_estimate": total_tokens,
                        "tokens_per_exchange": tpe,
                        "user_tokens": current_user_tokens,
                        "assistant_tokens": current_assistant_tokens,
                    })
            
            # Start new chain
            current_chain_start = i
            chain_start_idx = i
            current_exchanges = 0
            current_clarifications = 0
            current_user_tokens = 0
            current_assistant_tokens = 0
            current_task = user_text
            current_timestamp = msg.get("timestamp", datetime.now(timezone.utc).isoformat())
            current_user_tokens += estimate_tokens(user_text)
        
        elif is_assistant_message(msg) and current_chain_start is not None:
            assistant_text = extract_text(msg.get("content", ""))
            
            # Skip noise responses
            if is_noise_response(assistant_text):
                continue
            
            current_exchanges += 1
            current_assistant_tokens += estimate_tokens(assistant_text)
            content_lower = assistant_text.lower()
            if any(sig in content_lower for sig in CLARIFICATION_SIGNALS):
                current_clarifications += 1
        
        elif is_user_message(msg) and current_chain_start is not None:
            # Count follow-up user tokens in the chain
            user_text = get_user_text(msg)
            if user_text and not is_substantive_request(user_text):
                current_user_tokens += estimate_tokens(user_text)

    # Close final chain
    if current_chain_start is not None and current_exchanges > 0:
        total_tokens = current_user_tokens + current_assistant_tokens
        tpe = total_tokens // max(1, current_exchanges)
        
        # Filter out cron noise in task summaries
        if not current_task.startswith("[cron:"):
            chains.append({
                "task_summary": current_task[:120],
                "exchanges": current_exchanges,
                "brain_used": has_brain_context(messages, chain_start_idx, len(messages), session_brain_preamble),
                "clarifications_needed": current_clarifications,
                "first_try_success": current_exchanges == 1 and current_clarifications == 0,
                "timestamp": current_timestamp,
                "tokens_estimate": total_tokens,
                "tokens_per_exchange": tpe,
                "user_tokens": current_user_tokens,
                "assistant_tokens": current_assistant_tokens,
            })

    return chains


def deduplicate_chains(new_chains, existing_timestamps):
    """Skip chains with timestamps we've already logged."""
    return [c for c in new_chains if c.get("timestamp") not in existing_timestamps]


def get_existing_timestamps():
    """Load timestamps of already-logged chains to prevent duplicates."""
    if not TRACKING_LOG.exists():
        return set()
    timestamps = set()
    with open(TRACKING_LOG) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    entry = json.loads(line)
                    timestamps.add(entry.get("timestamp"))
                except json.JSONDecodeError:
                    continue
    return timestamps


def append_to_log(chains):
    """Append task chains to JSONL tracking log."""
    TRACKING_LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(TRACKING_LOG, "a") as f:
        for chain in chains:
            chain["logged_at"] = datetime.now(timezone.utc).isoformat()
            f.write(json.dumps(chain) + "\n")
    return len(chains)


def weekly_trend_analysis(entries):
    """Analyze trends over weekly buckets."""
    if not entries:
        return
    
    # Group by weeks
    weekly_buckets = defaultdict(list)
    for entry in entries:
        timestamp = entry.get("timestamp")
        try:
            if isinstance(timestamp, str):
                dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
            else:
                dt = datetime.fromtimestamp(timestamp / 1000, tz=timezone.utc)
            
            # Get Monday of that week
            week_start = dt - timedelta(days=dt.weekday())
            week_key = week_start.strftime("%Y-%m-%d")
            weekly_buckets[week_key].append(entry)
        except (ValueError, TypeError):
            continue
    
    if len(weekly_buckets) < 2:
        return
    
    print(f"\n=== Weekly Trend Analysis ===")
    for week in sorted(weekly_buckets.keys())[-4:]:  # Last 4 weeks
        week_entries = weekly_buckets[week]
        avg_exchanges = sum(e["exchanges"] for e in week_entries) / len(week_entries)
        brain_usage = sum(1 for e in week_entries if e["brain_used"]) / len(week_entries) * 100
        first_try = sum(1 for e in week_entries if e["first_try_success"]) / len(week_entries) * 100
        
        print(f"  Week {week}: {len(week_entries)} tasks, "
              f"{avg_exchanges:.1f} avg exchanges, "
              f"{brain_usage:.0f}% brain usage, "
              f"{first_try:.0f}% first-try success")


def generate_report():
    """Generate summary stats from tracking log."""
    if not TRACKING_LOG.exists():
        print("No tracking data yet.")
        return

    entries = []
    with open(TRACKING_LOG) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    entry = json.loads(line)
                    # Skip cron noise entries that may have slipped through
                    if not entry.get("task_summary", "").startswith("[cron:"):
                        entries.append(entry)
                except json.JSONDecodeError:
                    continue

    if not entries:
        print("No tracking data yet.")
        return

    recent = entries[-REPORT_LAST_N:]
    
    with_brain = [e for e in recent if e["brain_used"]]
    without_brain = [e for e in recent if not e["brain_used"]]

    def stats(subset, label):
        if not subset:
            print(f"\n{label}: no data")
            return
        avg_exchanges = sum(e["exchanges"] for e in subset) / len(subset)
        first_try_rate = sum(1 for e in subset if e["first_try_success"]) / len(subset) * 100
        avg_clarifications = sum(e["clarifications_needed"] for e in subset) / len(subset)
        
        # Token stats (graceful for old entries without token data)
        token_entries = [e for e in subset if e.get("tokens_estimate")]
        avg_tokens = sum(e["tokens_estimate"] for e in token_entries) / len(token_entries) if token_entries else 0
        avg_tpe = sum(e["tokens_per_exchange"] for e in token_entries) / len(token_entries) if token_entries else 0
        
        print(f"\n{label} (n={len(subset)}):")
        print(f"  Avg exchanges/task:    {avg_exchanges:.1f}")
        print(f"  First-try success:     {first_try_rate:.0f}%")
        print(f"  Avg clarifications:    {avg_clarifications:.1f}")
        if token_entries:
            print(f"  Avg tokens/task:       {avg_tokens:,.0f}")
            print(f"  Avg tokens/exchange:   {avg_tpe:,.0f}")

    print(f"=== Round-Trip Report (last {len(recent)} tasks) ===")
    print(f"Total tracked: {len(entries)}")
    stats(recent, "Overall")
    stats(with_brain, "With brain context")
    stats(without_brain, "Without brain context")

    # Delta
    if with_brain and without_brain:
        brain_avg = sum(e["exchanges"] for e in with_brain) / len(with_brain)
        no_brain_avg = sum(e["exchanges"] for e in without_brain) / len(without_brain)
        delta = no_brain_avg - brain_avg
        print(f"\n  Brain saves: {delta:+.1f} exchanges/task on average")

    # Efficiency analysis
    efficiency_entries = [e for e in recent if e.get("user_tokens") and e.get("assistant_tokens")]
    if efficiency_entries:
        print(f"\n=== Efficiency Analysis (n={len(efficiency_entries)}) ===")
        
        # Output tokens per input token
        ratios = []
        for e in efficiency_entries:
            user_t = e["user_tokens"]
            assistant_t = e["assistant_tokens"]
            if user_t > 0:
                ratios.append(assistant_t / user_t)
        
        if ratios:
            avg_ratio = sum(ratios) / len(ratios)
            print(f"  Avg output/input token ratio: {avg_ratio:.2f}")
        
        # Brain context ROI
        brain_eff = [e for e in efficiency_entries if e["brain_used"]]
        no_brain_eff = [e for e in efficiency_entries if not e["brain_used"]]
        
        if brain_eff and no_brain_eff:
            brain_ratio = sum(e["assistant_tokens"] / max(1, e["user_tokens"]) for e in brain_eff) / len(brain_eff)
            no_brain_ratio = sum(e["assistant_tokens"] / max(1, e["user_tokens"]) for e in no_brain_eff) / len(no_brain_eff)
            roi = brain_ratio - no_brain_ratio
            print(f"  Brain context ROI: {roi:+.2f} (richer responses when brain used)")
    
    # Weekly trend analysis
    weekly_trend_analysis(entries)


def clean_log_file():
    """Remove cron noise entries from the log file."""
    if not TRACKING_LOG.exists():
        print("No log file to clean.")
        return 0
    
    clean_entries = []
    removed_count = 0
    
    with open(TRACKING_LOG) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    entry = json.loads(line)
                    if entry.get("task_summary", "").startswith("[cron:"):
                        removed_count += 1
                    else:
                        clean_entries.append(entry)
                except json.JSONDecodeError:
                    continue
    
    # Rewrite the file with clean entries
    with open(TRACKING_LOG, "w") as f:
        for entry in clean_entries:
            f.write(json.dumps(entry) + "\n")
    
    print(f"Cleaned log file: removed {removed_count} cron noise entries, kept {len(clean_entries)} real entries")
    return removed_count


def load_from_jsonl(jsonl_path):
    """Load messages from an OpenClaw session JSONL file directly.
    Much faster than going through sessions_history API."""
    messages = []
    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            
            # OpenClaw JSONL format: {"type": "message", "message": {"role": "...", "content": ...}}
            # Claude Code JSONL format: {"type": "user"/"assistant", "message": {"role": "...", "content": ...}}
            # Claude Code attachment format: {"type": "attachment", "attachment": {"content": "..."}, ...}
            # Note: attachment entries have no "message" field; content is in entry["attachment"]["content"].
            entry_type = entry.get("type", "")
            if entry_type == "message":
                msg = entry.get("message", {})
                role = msg.get("role")
                content = msg.get("content", "")
            elif entry_type in ("user", "assistant"):
                msg = entry.get("message", {})
                role = msg.get("role") or entry_type
                content = msg.get("content", "")
            elif entry_type == "attachment" and entry.get("attachment"):
                # Hook results (e.g. startup brain context from SessionStart hook).
                # Content lives in attachment.content, not in message.content.
                att = entry.get("attachment", {})
                content = att.get("content", "") or att.get("stdout", "")
                role = "hook"  # custom role so chain logic skips it but brain detection sees it
            else:
                continue
            if role in ("user", "assistant", "system", "hook"):
                messages.append({
                    "role": role,
                    "content": content,
                    "timestamp": entry.get("timestamp", ""),
                })
    return messages


if __name__ == "__main__":
    if "--report" in sys.argv:
        generate_report()
    elif "--clean-log" in sys.argv:
        clean_log_file()
    else:
        # Check for --from-jsonl flag (direct file read, no API needed)
        jsonl_path = None
        for i, arg in enumerate(sys.argv):
            if arg == "--from-jsonl" and i + 1 < len(sys.argv):
                jsonl_path = sys.argv[i + 1]
        
        if jsonl_path:
            messages = load_from_jsonl(jsonl_path)
            print(f"Loaded {len(messages)} messages from {jsonl_path}")
        else:
            # Read session history from stdin
            try:
                data = json.load(sys.stdin)
            except json.JSONDecodeError:
                print("Error: pipe valid session history JSON to stdin", file=sys.stderr)
                sys.exit(1)
            messages = data if isinstance(data, list) else data.get("messages", [])

        chains = extract_task_chains(messages)
        
        # Deduplicate against existing log
        existing = get_existing_timestamps()
        chains = deduplicate_chains(chains, existing)
        
        if chains:
            count = append_to_log(chains)
            print(f"Logged {count} task chains")
            for c in chains:
                brain = "🧠" if c["brain_used"] else "  "
                ftry = "✅" if c["first_try_success"] else f"❌({c['exchanges']}x)"
                print(f"  {brain} {ftry} {c['task_summary'][:80]}")
        else:
            print("No new task chains found")