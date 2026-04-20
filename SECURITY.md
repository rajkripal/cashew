# Security

If you've found a security issue in cashew, please do not file a public issue. Report it privately to **rajkripal.danday@gmail.com** with enough detail to reproduce.

Expect an acknowledgement within 72 hours. Valid reports get a fix and a coordinated disclosure; I'll credit you in the release notes unless you'd rather stay anonymous.

## Scope

In scope:
- Anything that lets an attacker read or modify another user's graph
- Code execution via malicious input to the extractor, ingest paths, or retrieval
- Secrets leaking from config or environment into logs / the graph

Out of scope:
- Attacks that require existing local shell access to the machine running cashew
- Dependency CVEs without a working exploit path through cashew itself
