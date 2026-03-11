# explenation.md

This document describes how this repository implements a Recursive Language
Model (RLM) workflow with OpenCode. It highlights the local agents, commands,
skills, and rules that govern how the system works.

## Purpose of the repo

- Provide an OpenCode-based workflow to analyze scientific PDF corpora.
- Use a persistent Python REPL to load, chunk, and manage long documents.
- Delegate chunk-level extraction to a subagent, then synthesize results with
  strict citation safeguards.

## OpenCode configuration (opencode.json)

OpenCode uses a project config to define agents, commands, permissions, and
instruction files. See `opencode.json`.

### Agents

Defined in `opencode.json`:

- `research-write` (primary): full tool access; writes LaTeX/BibTeX output and
  orchestrates the RLM workflow.
- `research-read` (primary): read-only analysis mode for safe exploration.
- `rlm-subcall` (subagent): chunk-level extractor; no tool use beyond read.
- `paper-analyzer` (subagent): deep single-paper analysis from a file or chunk.

Agent prompts live in `prompts/research-write.md`, `prompts/research-read.md`,
and `prompts/paper-analyzer.md`. The `rlm-subcall` agent is defined in
`.opencode/agents/rlm-subcall.md`.

Notes from OpenCode docs:

- Agents are either primary or subagents; primary agents are selected via Tab,
  subagents via `@mention` or command configuration.
- Agents can be defined in `opencode.json` or via markdown in
  `.opencode/agents/`.

### Commands

Project commands are defined in two places:

1. `opencode.json` (custom commands):
   - `/rlm` (orchestration flow)
   - `/research` (corpus-grounded answer)
   - `/write-section` (LaTeX section + BibTeX)
   - `/cite-check` (citation verification)
   - `/bibtex` (generate entries)
   - `/deep-dive` (single paper analysis)
2. `.opencode/commands/rlm.md` (a command file for the RLM workflow)

OpenCode docs confirm commands can be defined in `opencode.json` or
`.opencode/commands/*.md`, and that `$ARGUMENTS` expands command input.

### Skills

The repository defines a local skill at `.opencode/skills/rlm/SKILL.md`.
Per OpenCode docs, skills live at `.opencode/skills/<name>/SKILL.md` and are
loaded on demand by the `skill` tool.

The `rlm` skill is a workflow guide that explains:

- How to load a PDF corpus into the REPL
- How to list/search/cite papers
- How to chunk and delegate to `@rlm-subcall`
- How to synthesize results with verified citations

## RLM workflow architecture in this repo

The RLM workflow mirrors the RLM paper's design: the prompt is treated as an
external environment, not shoved into the model context. The persistent REPL
holds the corpus and exposes helpers, while the main agent writes code to
inspect and decompose the prompt, and calls subagents on chunks.

Core components:

- REPL: `.opencode/skills/rlm/scripts/rlm_repl.py`
  - Loads PDFs, extracts text, and builds a corpus state.
  - Persists state in `.opencode/rlm_state/state.pkl`.
  - Writes chunk files to `.opencode/rlm_state/chunks/`.
- Subagent: `@rlm-subcall`
  - Reads chunk files and returns JSON with quotes and metadata.
- Root agent: `research-write`
  - Orchestrates chunk selection, subcalls, and final synthesis.

### Diagram (repo-level flow)

```text
User prompt / task
        |
        v
research-write (root agent)
        |
        |  bash: rlm_repl.py load-corpus / exec helpers
        v
Persistent REPL (state + corpus + chunks)
        |
        |  per-chunk file paths
        v
@rlm-subcall (chunk extractor)
        |
        v
JSON evidence + quotes + metadata
        |
        v
research-write synthesizes + cites
```

### RLM paper summary (how the model works)

From the RLM paper (arXiv 2512.24601, HTML version). The summary below is
paraphrased and references paper sections for traceability:

- Definition: RLMs are inference-time scaffolds that treat the prompt as an
  external environment, allowing dense processing without loading the full
  prompt into the model context. (Abstract, Section 2)
- Core loop: the model operates in a persistent REPL, writes code, executes it,
  inspects metadata about outputs, and stops when a final variable is set.
  (Section 2)
- Prompt-as-environment: the model gets a symbolic handle to the prompt and
  inspects or transforms it via code, instead of copying it into context.
  (Sections 1-2)
- Symbolic recursion: code running in the environment can invoke sub-calls over
  programmatically constructed slices and stitch results. (Section 2, Section 4.1)
- Advantages vs compaction: because the input remains accessible in the
  environment, RLMs avoid lossy summarization for dense tasks. (Section 1)
- Scaling behavior: RLMs degrade more slowly on long, complex tasks and can
  operate well beyond base model context windows. (Section 4)
- Sub-call importance: REPL alone helps long inputs, but recursive sub-calls
  significantly improve information-dense tasks. (Section 4)
- Costs and variance: median costs can be comparable to base models, but
  trajectory length variance can raise tail costs. (Section 4)
- Limitations: the paper uses synchronous REPL sub-calls and depth-1 recursion;
  deeper recursion and async/sandboxed execution are open directions. (Section 6)

## How the RLM workflow runs end-to-end

1. Place PDFs in `context/`.
2. Load the corpus with the REPL:
   - `python .opencode/skills/rlm/scripts/rlm_repl.py load-corpus ./context/`
3. List and search papers:
   - `python .opencode/skills/rlm/scripts/rlm_repl.py exec -c "print(list_papers())"`
   - `python .opencode/skills/rlm/scripts/rlm_repl.py exec -c "print(find_papers('keyword'))"`
4. For large corpora, use chunk files in `.opencode/rlm_state/chunks/`.
5. Pass each chunk file path to `@rlm-subcall` with a query.
6. Synthesize results in the root agent with verified citations.

## Quick run checklist (common tasks)

### First-time setup

1. Install Python deps:
   - `pip install -r requirements.txt`
2. (Optional) generate sample data:
   - `python examples/generate_sample_data.py`

### Analyze a corpus (typical)

1. Put PDFs in `context/`.
2. Load corpus:
   - `python .opencode/skills/rlm/scripts/rlm_repl.py load-corpus ./context/`
3. Inspect:
   - `python .opencode/skills/rlm/scripts/rlm_repl.py status`
   - `python .opencode/skills/rlm/scripts/rlm_repl.py exec -c "print(list_papers())"`
4. Search:
   - `python .opencode/skills/rlm/scripts/rlm_repl.py exec -c "print(find_papers('keyword'))"`
5. Delegate chunks to `@rlm-subcall`, synthesize in `research-write`.

### Deep-dive a single paper

1. Load one PDF:
   - `python .opencode/skills/rlm/scripts/rlm_repl.py pdf ./context/paper.pdf`
2. Peek:
   - `python .opencode/skills/rlm/scripts/rlm_repl.py exec -c "print(peek(0, 3000))"`
3. Use `@paper-analyzer` for structured extraction.

### Citation check

1. Use `/cite-check` with LaTeX text in OpenCode.
2. Validate that every `\textcite{}`/`\parencite{}` key exists in the corpus.

## Local rules and safety constraints

Project rules live in `AGENTS.md` and are loaded by OpenCode per the rules
system. Key constraints:

- Strict anti-hallucination: never fabricate authors, years, or claims; only
  cite papers present in the loaded corpus.
- Use REPL helpers (`list_papers`, `find_papers`, `cite`, `search_claim`) to
  ground claims.
- Output defaults to LaTeX with BibTeX (APA 7), using `\textcite{}` and
  `\parencite{}`.
- Python style rules follow `rlm_repl.py` conventions (imports, typing,
  `pathlib.Path`, explicit UTF-8, clear error handling with `RlmReplError`).

Additional project instruction files are included via `opencode.json`:

- `AGENTS.md`
- `knowledge/*.md` (research guidelines, incident response, SRE practices)

## Other notable files

- `README.md`: overview, quick start, commands, and architecture diagram.
- `examples/generate_sample_data.py`: generates sample logs and datasets for
  testing the RLM workflow.
- `context/`: user-provided PDF corpus (data, not code).

## Quick mental model

- The root agent does orchestration and synthesis.
- The REPL is the persistent environment holding the prompt/corpus.
- The subagent extracts evidence from chunks only.
- Local rules (AGENTS.md) and OpenCode rules enforce no-hallucination output.
