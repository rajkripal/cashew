# Contributing to Cashew

Thanks for taking a look. Cashew is a small project with a clear design intent, so a short guide saves everyone time.

## Running tests

```bash
pip install -e ".[dev]"
pytest
```

Full suite runs in under 30 seconds. Every PR should leave it green.

## What to send

- **Bug fix**: include a regression test that fails before your change and passes after. No test, no merge.
- **New feature**: open an issue first if it's non-trivial. Spares us both from wasted work if the shape is off.
- **Refactor without behavior change**: keep the diff focused. If you find yourself touching unrelated files, split the PR.

## Code style

See `.claude/skills/code-style/SKILL.md` if you're working via Claude Code — the short version is: modular, elegant, DRY, testable. Concretely:

- One concept per file. Narrow public interface, helpers underscore-prefixed.
- Early returns over nested branches. Type hints on public signatures.
- Tests ship with the code, not in a follow-up.
- Comments explain *why*, not *what*.

## PR shape

- Subject: `<area>: <what changed>`, imperative, under 70 chars.
- Body: 1-3 short paragraphs. What's broken, what the fix is, why it's safe. Skip section headers for small PRs.
- No AI-generated footer or marketing language.

## Design constraints to respect

Cashew has a few load-bearing decisions that shouldn't be reversed without discussion:

- **Dumb graph, smart reasoning layer.** The graph stores connections; meaning lives in the LLM that reads it.
- **No typed edges.** Ablation-tested and removed. Don't add them back without numbers.
- **Organic decay is the forgetting mechanism.** Don't build structures that fight decay.
- **Single SQLite file.** No external servers, no separate indexes.

If a PR touches these, explain the motivation in the body.

## Filing issues

Bug reports should include: what you ran, what you expected, what happened, and the smallest reproduction you can produce. Python version and OS help.
