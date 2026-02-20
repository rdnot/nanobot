## Purpose

This file gives targeted, actionable guidance for AI coding agents working on nanobot.
Keep recommendations concrete and grounded in the repository's code and workflows.

## Big picture (why & how)

- nanobot is an ultra-lightweight agent framework: core runtime is in `nanobot/agent`.
- Message flow: inbound message -> `MessageBus` -> `AgentLoop` builds context (`nanobot/agent/context.py`) -> provider chat -> tool calls -> tool execution -> outbound via `ChannelManager` (`nanobot/channels/manager.py`).
- Skills extend behavior via `skills/*/SKILL.md` and are progressively loaded by `nanobot/agent/skills.py`.

## Key files to inspect first

- [nanobot/agent/context.py](nanobot/agent/context.py): system prompt & context assembly.
- [nanobot/agent/loop.py](nanobot/agent/loop.py): core processing loop, tool registration, and command handling.
- [nanobot/agent/skills.py](nanobot/agent/skills.py): skill loading, frontmatter metadata, and availability checks.
- [nanobot/agent/tools/registry.py](nanobot/agent/tools/registry.py): dynamic tool registration and execution contract.
- [nanobot/providers/registry.py](nanobot/providers/registry.py): canonical provider metadata and how provider selection/matching works.
- [nanobot/cli/commands.py](nanobot/cli/commands.py): important developer CLI commands (`onboard`, `gateway`, `agent`, `rerun`).
- [README.md](README.md) and [workspace/AGENTS.md](workspace/AGENTS.md): user-facing quickstart and bootstrap guidance (the agent reads these into system prompts).

## Project-specific conventions and patterns

- Bootstrap documents (AGENTS.md, SOUL.md, USER.md, TOOLS.md) are loaded into the system prompt. Prefer edits to these when changing agent behavior.
- Skills use YAML-like frontmatter for metadata. `SkillsLoader` exposes availability checks by binary (`requires.bins`) and env vars (`requires.env`).
- Progressive skill loading: include short summaries in the system prompt; the agent uses the `read_file` tool to load full `SKILL.md` on demand.
- Tools must implement `to_schema()` and `validate_params()`; `ToolRegistry.execute()` returns a string (errors are surfaced as strings). See `nanobot/agent/tools/`.
- Persistent memory: workspace `memory/MEMORY.md` (long-term) and `memory/HISTORY.md` (append-only). Agents write to these files to persist facts.
- Message sending: prefer returning text in assistant responses. Use the `message` tool only when you need to send outgoing messages to a channel (e.g., WhatsApp). See `ContextBuilder._get_identity()` guidance.
- Shell execution: `ExecTool` respects `restrict_to_workspace`. Tests and tooling assume workspace-restricted exec by default; avoid running arbitrary system commands outside the workspace.

## Configuration & providers

- Config path: `~/.nanobot/config.json` (see `nanobot/config/loader.py:get_config_path`).
- Provider selection uses `providers/registry.py`. To add a provider, add a `ProviderSpec` entry and corresponding config schema.
- Common startup flows: `pip install -e .`, then `nanobot onboard` to create workspace templates and `nanobot gateway` (or `nanobot agent`) to run.

## Developer workflows (how to build, run, debug)

- Install editable: `pip install -e .`
- Initialize workspace (creates AGENTS.md, SOUL.md, memory files): `nanobot onboard`.
- Run gateway (channels + cron + agent): `nanobot gateway`.
- Run a single interactive agent session: `nanobot agent -m "Hello!"` or use the CLI in README.
- Tests: run `pytest` from the repo root (tests are under `tests/`).

## Things an AI agent should do when editing code here

- Prefer changing bootstrap files (`AGENTS.md`, `SOUL.md`, `USER.md`) for agent behavior tweaks rather than deep code edits when possible.
- When adding a new tool: register it in `nanobot/agent/tools/registry.py` via `ToolRegistry.register()`, implement `to_schema()` and `validate_params()`, and include it in `AgentLoop._register_default_tools()` if it should be available by default.
- When adding a new provider: update `nanobot/providers/registry.py` and the config schema in `nanobot/config/schema.py`.
- When adding skills: create `skills/<name>/SKILL.md` with frontmatter describing `requires` (bins/env) and optional `always: true` to auto-load.

## Quick examples (copyable)

- Create workspace and start gateway:

  pip install -e .
  nanobot onboard
  nanobot gateway

- Run a one-off agent message:

  nanobot agent -m "Summarize today's news" 

## Gotchas & guardrails

- Do not assume message tool is equivalent to normal assistant response; use it only to send outbound messages.
- Use `read_file` to let the agent inspect `SKILL.md` files instead of pasting full skill contents into prompts.
- Respect `restrict_to_workspace` when running shell commands — it's intentionally enforced in `ExecTool`.

---

If anything in this guidance is unclear or you'd like additional examples (e.g., how to add a provider or a new tool), tell me which area to expand.
