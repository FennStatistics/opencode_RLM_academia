#!/usr/bin/env python3
"""Persistent mini-REPL for RLM-style scientific literature workflows in OpenCode.

This script provides a *stateful* Python environment across invocations by
saving a pickle file to disk. It is intentionally small.

Designed for scientific research workflows: analysing corpora of PDF papers,
extracting claims, managing citations, and supporting academic writing.

Requires:
  pip install pdfplumber

Typical flow:
  # Load a full corpus directory of PDFs:
  python rlm_repl.py load-corpus ./context/

  # Load a single PDF:
  python rlm_repl.py pdf path/to/paper.pdf

  # Execute code repeatedly (state persists):
  python rlm_repl.py exec -c 'print(list_papers())'
  python rlm_repl.py exec -c 'print(find_papers("climate"))'
  python rlm_repl.py exec -c 'print(cite("Rockström 2009"))'

The script injects these variables into the exec environment:
  - context: dict with keys {path, loaded_at, content, papers}
  - content: string alias for context['content'] (all text concatenated)
  - papers: list of {filename, bibtex_key, header, text, char_start, char_end}
  - buffers: list[str] for storing intermediate results

Research helpers injected:
  - list_papers() -> formatted list of loaded papers
  - find_papers(keyword) -> papers matching author/title keyword
  - cite(author_year) -> passages around mentions of a reference
  - search_claim(claim) -> passages relevant to a claim
  - extract_references() -> parse reference sections from papers
  - get_paper(index_or_keyword) -> full text of one paper
  - peek(start, end) -> slice of concatenated content
  - grep(pattern, ...) -> regex search with context
  - grep_count(pattern) -> count matches
  - find_lines(pattern, ...) -> matching lines with numbers
  - chunk_indices(size, overlap) -> chunk boundaries
  - write_chunks(out_dir, size, overlap) -> split content to files
  - add_buffer(text) -> store intermediate results
  - stats() -> corpus overview

Security note:
  This runs arbitrary Python via exec. Treat it like running code you wrote.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import pickle
import re
import sys
import textwrap
import time
import traceback
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


DEFAULT_STATE_PATH = Path(".opencode/rlm_state/state.pkl")
DEFAULT_MAX_OUTPUT_CHARS = 8000

# Separator used between papers in the concatenated corpus
PAPER_SEPARATOR = "\n\n{'='*80}\n"


class RlmReplError(RuntimeError):
    pass


def _ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _load_state(state_path: Path) -> Dict[str, Any]:
    if not state_path.exists():
        raise RlmReplError(
            f"No state found at {state_path}. "
            f"Run: python rlm_repl.py load-corpus <dir>  OR  python rlm_repl.py pdf <file.pdf>"
        )
    with state_path.open("rb") as f:
        state = pickle.load(f)
    if not isinstance(state, dict):
        raise RlmReplError(f"Corrupt state file: {state_path}")
    return state


def _save_state(state: Dict[str, Any], state_path: Path) -> None:
    _ensure_parent_dir(state_path)
    tmp_path = state_path.with_suffix(state_path.suffix + ".tmp")
    with tmp_path.open("wb") as f:
        pickle.dump(state, f, protocol=pickle.HIGHEST_PROTOCOL)
    tmp_path.replace(state_path)


def _read_text_file(path: Path, max_bytes: int | None = None) -> str:
    if not path.exists():
        raise RlmReplError(f"File does not exist: {path}")
    with path.open("rb") as f:
        data = f.read() if max_bytes is None else f.read(max_bytes)
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        return data.decode("utf-8", errors="replace")


def _extract_pdf_text(path: Path) -> str:
    """Extract text from a PDF using pdfplumber."""
    try:
        import pdfplumber
    except ImportError:
        raise RlmReplError(
            "pdfplumber is not installed. Run: pip install pdfplumber"
        )
    text_parts: List[str] = []
    with pdfplumber.open(str(path)) as pdf:
        for page in pdf.pages:
            t = page.extract_text()
            if t:
                text_parts.append(t)
    return "\n".join(text_parts)


def _parse_filename_metadata(filename: str) -> Dict[str, Any]:
    """
    Parse Zotero-style filenames into metadata.
    Patterns like:
      'Cook et al. - 2013 - Quantifying the consensus on anthropogenic...pdf'
      'Smith - 2020 - Title.pdf'
      'Author1 und Author2 - 2021 - Title.pdf'
    """
    stem = Path(filename).stem
    # Try: Authors - Year - Title
    m = re.match(r'^(.+?)\s+-\s+(\d{4})\s+-\s+(.+)$', stem)
    if m:
        authors_raw, year, title = m.group(1), m.group(2), m.group(3)
    else:
        authors_raw, year, title = stem, "unknown", stem

    # Build a BibTeX key: firstauthorlastname + year + firstmeaningfulword
    # Extract first author's last name
    first_author = re.split(r'\s+et\s+al\.?|,|\s+und\s+|\s+and\s+', authors_raw)[0].strip()
    # Take last word as surname
    first_author_last = first_author.split()[-1].lower() if first_author.split() else "unknown"
    # Remove non-alphanumeric
    first_author_last = re.sub(r'[^a-z0-9]', '', first_author_last)

    # First meaningful word from title (skip articles/prepositions)
    stop_words = {'a', 'an', 'the', 'of', 'in', 'on', 'at', 'to', 'for',
                  'and', 'or', 'but', 'is', 'are', 'was', 'were', 'with'}
    title_words = [w.lower() for w in re.split(r'\W+', title) if w]
    first_content_word = next(
        (w for w in title_words if w not in stop_words and len(w) > 2),
        title_words[0] if title_words else 'paper'
    )
    first_content_word = re.sub(r'[^a-z0-9]', '', first_content_word)

    bibtex_key = f"{first_author_last}{year}{first_content_word}"

    return {
        "filename": filename,
        "authors_raw": authors_raw,
        "year": year,
        "title": title,
        "bibtex_key": bibtex_key,
    }


def _truncate(s: str, max_chars: int) -> str:
    if max_chars <= 0:
        return ""
    if len(s) <= max_chars:
        return s
    return s[:max_chars] + f"\n... [truncated to {max_chars} chars] ...\n"


def _is_pickleable(value: Any) -> bool:
    try:
        pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)
        return True
    except Exception:
        return False


def _filter_pickleable(d: Dict[str, Any]) -> Tuple[Dict[str, Any], List[str]]:
    kept: Dict[str, Any] = {}
    dropped: List[str] = []
    for k, v in d.items():
        if _is_pickleable(v):
            kept[k] = v
        else:
            dropped.append(k)
    return kept, dropped


def _save_incremental(papers: List[Dict], corpus_dir: Path, state_path: Path) -> None:
    """Rebuild concatenated content from current papers list and save state.

    Called after each PDF so that partial progress survives a timeout or
    interruption. On the next run, load-corpus will skip already-loaded files
    and resume from where it left off.
    """
    separator = "\n\n" + "=" * 80 + "\n\n"
    content_parts = [p["header"] + "\n\n" + p["text"] for p in papers]
    full_content = separator.join(content_parts)
    # Annotate char positions
    pos = 0
    for i, part in enumerate(content_parts):
        papers[i]["char_start"] = pos
        papers[i]["char_end"] = pos + len(part)
        pos += len(part) + len(separator)
    state: Dict[str, Any] = {
        "version": 2,
        "context": {
            "path": str(corpus_dir),
            "loaded_at": time.time(),
            "content": full_content,
            "papers": list(papers),  # copy so later mutations don't affect saved state
        },
        "buffers": [],
        "globals": {},
    }
    _save_state(state, state_path)


def _write_paper_chunks(
    paper: Dict,
    chunks_dir: Path,
    size: int = 50_000, # 150_000,
    overlap: int = 2_000,
) -> List[str]:
    """Write chunk file(s) for a single paper to chunks_dir immediately after loading.

    Files are named: <chunks_dir>/<bibtex_key>_chunk_<NNNN>.txt
    Uses overlapping windows so claims near chunk boundaries are not cut off.
    Returns list of written file paths.
    """
    bibtex_key = paper.get("metadata", {}).get("bibtex_key", "paper")
    # Sanitise key for use as a filename (remove characters unsafe on Windows/POSIX)
    safe_key = re.sub(r'[<>:"/\\|?*]', '_', bibtex_key)
    text = paper["header"] + "\n\n" + paper["text"]
    n = len(text)
    paths: List[str] = []

    if n == 0:
        return paths

    chunks_dir.mkdir(parents=True, exist_ok=True)
    step = max(1, size - overlap)
    chunk_idx = 0
    for start in range(0, n, step):
        end = min(n, start + size)
        chunk_path = chunks_dir / f"{safe_key}_chunk_{chunk_idx:04d}.txt"
        chunk_path.write_text(text[start:end], encoding="utf-8")
        paths.append(str(chunk_path))
        chunk_idx += 1
        if end >= n:
            break

    return paths


def _make_helpers(context_ref: Dict[str, Any], buffers_ref: List[str]):
    """Create helper functions for scientific research workflows."""

    # ------------------------------------------------------------------ #
    # Basic text helpers (kept from original for chunking/grep support)   #
    # ------------------------------------------------------------------ #

    def peek(start: int = 0, end: int = 1000) -> str:
        """Return a slice of the concatenated corpus content."""
        return context_ref.get("content", "")[start:end]

    def grep(
        pattern: str,
        max_matches: int = 20,
        window: int = 200,
        flags: int = 0,
    ) -> List[Dict[str, Any]]:
        """Search for pattern in corpus content, return matches with context."""
        content = context_ref.get("content", "")
        out: List[Dict[str, Any]] = []
        for m in re.finditer(pattern, content, flags):
            start, end = m.span()
            snippet_start = max(0, start - window)
            snippet_end = min(len(content), end + window)
            out.append({
                "match": m.group(0),
                "span": (start, end),
                "snippet": content[snippet_start:snippet_end],
            })
            if len(out) >= max_matches:
                break
        return out

    def grep_count(pattern: str, flags: int = 0) -> int:
        """Count occurrences of pattern in corpus."""
        content = context_ref.get("content", "")
        return len(re.findall(pattern, content, flags))

    def find_lines(
        pattern: str,
        max_matches: int = 100,
        flags: int = 0,
    ) -> List[Dict[str, Any]]:
        """Find lines matching pattern, return with line numbers."""
        content = context_ref.get("content", "")
        lines = content.splitlines()
        out: List[Dict[str, Any]] = []
        regex = re.compile(pattern, flags)
        for i, line in enumerate(lines, 1):
            if regex.search(line):
                out.append({"line_number": i, "content": line})
                if len(out) >= max_matches:
                    break
        return out

    def chunk_indices(size: int = 50_000, overlap: int = 2000) -> List[Tuple[int, int]]:
        """Calculate chunk boundaries for the concatenated corpus."""
        if size <= 0:
            raise ValueError("size must be > 0")
        if overlap < 0:
            raise ValueError("overlap must be >= 0")
        if overlap >= size:
            raise ValueError("overlap must be < size")
        content = context_ref.get("content", "")
        n = len(content)
        spans: List[Tuple[int, int]] = []
        step = size - overlap
        for start in range(0, n, step):
            end = min(n, start + size)
            spans.append((start, end))
            if end >= n:
                break
        return spans

    def write_chunks(
        out_dir: str,
        size: int = 50_000,
        overlap: int = 2000,
        prefix: str = "chunk",
        encoding: str = "utf-8",
    ) -> List[str]:
        """Write corpus chunks to files and return paths."""
        content = context_ref.get("content", "")
        spans = chunk_indices(size=size, overlap=overlap)
        out_path = Path(out_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        paths: List[str] = []
        for i, (s, e) in enumerate(spans):
            p = out_path / f"{prefix}_{i:04d}.txt"
            p.write_text(content[s:e], encoding=encoding)
            paths.append(str(p))
        return paths

    def add_buffer(text: str) -> None:
        """Add text to the buffers list for later synthesis."""
        buffers_ref.append(str(text))

    # ------------------------------------------------------------------ #
    # Research-specific helpers                                            #
    # ------------------------------------------------------------------ #

    def list_papers() -> str:
        """List all papers loaded in the corpus with their index and BibTeX key."""
        papers = context_ref.get("papers", [])
        if not papers:
            return "No papers loaded. Use load-corpus or pdf command first."
        lines = [f"Loaded corpus: {len(papers)} paper(s)\n"]
        for i, p in enumerate(papers):
            meta = p.get("metadata", {})
            lines.append(
                f"[{i:03d}] {meta.get('bibtex_key', '?')}  |  "
                f"{meta.get('authors_raw', '?')} ({meta.get('year', '?')})  —  "
                f"{meta.get('title', p.get('filename', '?'))[:80]}"
            )
        return "\n".join(lines)

    def find_papers(keyword: str, max_results: int = 20) -> List[Dict[str, Any]]:
        """Find papers matching a keyword in filename, authors, or title."""
        papers = context_ref.get("papers", [])
        keyword_lower = keyword.lower()
        results = []
        for i, p in enumerate(papers):
            meta = p.get("metadata", {})
            searchable = " ".join([
                p.get("filename", ""),
                meta.get("authors_raw", ""),
                meta.get("title", ""),
                meta.get("bibtex_key", ""),
            ]).lower()
            if keyword_lower in searchable:
                results.append({
                    "index": i,
                    "bibtex_key": meta.get("bibtex_key"),
                    "authors": meta.get("authors_raw"),
                    "year": meta.get("year"),
                    "title": meta.get("title"),
                    "filename": p.get("filename"),
                })
                if len(results) >= max_results:
                    break
        return results

    def get_paper(index_or_keyword) -> str:
        """Get the full extracted text of a paper by index (int) or keyword (str)."""
        papers = context_ref.get("papers", [])
        if not papers:
            return "No papers loaded."
        if isinstance(index_or_keyword, int):
            if 0 <= index_or_keyword < len(papers):
                p = papers[index_or_keyword]
                return p.get("header", "") + "\n\n" + p.get("text", "")
            return f"Index {index_or_keyword} out of range (0–{len(papers)-1})"
        # keyword search
        kw = str(index_or_keyword).lower()
        for p in papers:
            meta = p.get("metadata", {})
            searchable = " ".join([
                p.get("filename", ""),
                meta.get("authors_raw", ""),
                meta.get("title", ""),
                meta.get("bibtex_key", ""),
            ]).lower()
            if kw in searchable:
                return p.get("header", "") + "\n\n" + p.get("text", "")
        return f"No paper found matching '{index_or_keyword}'"

    def cite(author_year: str, window: int = 400, max_matches: int = 10) -> List[Dict[str, Any]]:
        """
        Find passages in the corpus that mention a specific reference.
        E.g. cite('Rockström 2009') or cite('Cook et al., 2013')
        """
        content = context_ref.get("content", "")
        # Build a flexible regex from the input
        # Allow for partial matches: "Rockström" and "2009" anywhere nearby
        parts = re.split(r'[\s,]+', author_year.strip())
        pattern = r'(?i)' + r'.{0,50}'.join(re.escape(p) for p in parts if p)
        results = []
        for m in re.finditer(pattern, content):
            start, end = m.span()
            snippet_start = max(0, start - window)
            snippet_end = min(len(content), end + window)
            results.append({
                "match": m.group(0),
                "context": content[snippet_start:snippet_end],
            })
            if len(results) >= max_matches:
                break
        return results

    def search_claim(claim: str, window: int = 300, max_matches: int = 15) -> List[Dict[str, Any]]:
        """
        Find passages in the corpus relevant to a claim or concept.
        Searches for key terms from the claim.
        """
        content = context_ref.get("content", "")
        # Extract meaningful words (>4 chars, skip stopwords)
        stop_words = {
            'that', 'this', 'with', 'from', 'they', 'have', 'been',
            'will', 'would', 'could', 'should', 'which', 'their',
            'there', 'about', 'more', 'also', 'some', 'when', 'than',
        }
        words = [
            w.lower() for w in re.findall(r'\b\w+\b', claim)
            if len(w) > 4 and w.lower() not in stop_words
        ]
        if not words:
            return []
        # Search for any two keywords within a window
        results = []
        pattern = r'(?i)' + r'|'.join(re.escape(w) for w in words[:6])
        for m in re.finditer(pattern, content):
            start, end = m.span()
            snippet_start = max(0, start - window)
            snippet_end = min(len(content), end + window)
            results.append({
                "match": m.group(0),
                "context": content[snippet_start:snippet_end],
            })
            if len(results) >= max_matches:
                break
        return results

    def extract_references(paper_index: Optional[int] = None, max_refs: int = 50) -> List[str]:
        """
        Extract reference list entries from a paper (or all papers).
        Returns raw reference strings as they appear in the text.
        paper_index: if None, scans all papers; otherwise scans one paper.
        """
        papers = context_ref.get("papers", [])
        if not papers:
            return []

        targets = [papers[paper_index]] if paper_index is not None else papers
        all_refs: List[str] = []

        for p in targets:
            text = p.get("text", "")
            # Find reference section: look for "References", "Bibliography", "Literature"
            ref_match = re.search(
                r'\n(?:References|Bibliography|Literature Cited|Works Cited)\s*\n',
                text, re.IGNORECASE
            )
            if ref_match:
                ref_text = text[ref_match.end():]
            else:
                # Fall back: look for lines starting with [1] or author-year patterns
                ref_text = text

            # Extract individual reference lines (heuristic)
            ref_lines = []
            for line in ref_text.splitlines():
                line = line.strip()
                if len(line) > 20 and re.search(r'\b(19|20)\d{2}\b', line):
                    ref_lines.append(line)
                    if len(ref_lines) >= max_refs:
                        break
            all_refs.extend(ref_lines)

        return all_refs[:max_refs]

    def stats() -> Dict[str, Any]:
        """Return corpus statistics."""
        content = context_ref.get("content", "")
        papers = context_ref.get("papers", [])
        lines = content.splitlines()
        return {
            "paper_count": len(papers),
            "total_chars": len(content),
            "total_lines": len(lines),
            "avg_paper_chars": len(content) // max(len(papers), 1),
            "papers": [
                {
                    "index": i,
                    "bibtex_key": p.get("metadata", {}).get("bibtex_key"),
                    "chars": len(p.get("text", "")),
                }
                for i, p in enumerate(papers)
            ] if len(papers) <= 20 else f"({len(papers)} papers — use list_papers() for full list)",
        }

    return {
        "peek": peek,
        "grep": grep,
        "grep_count": grep_count,
        "find_lines": find_lines,
        "chunk_indices": chunk_indices,
        "write_chunks": write_chunks,
        "add_buffer": add_buffer,
        "list_papers": list_papers,
        "find_papers": find_papers,
        "get_paper": get_paper,
        "cite": cite,
        "search_claim": search_claim,
        "extract_references": extract_references,
        "stats": stats,
    }


# ------------------------------------------------------------------ #
# Commands                                                             #
# ------------------------------------------------------------------ #

def cmd_pdf(args: argparse.Namespace) -> int:
    """Load a single PDF file into the REPL state."""
    state_path = Path(args.state)
    pdf_path = Path(args.pdf_path)

    if not pdf_path.exists():
        raise RlmReplError(f"PDF file not found: {pdf_path}")

    print(f"Extracting text from: {pdf_path} ...", file=sys.stderr)
    text = _extract_pdf_text(pdf_path)
    meta = _parse_filename_metadata(pdf_path.name)

    paper = {
        "filename": pdf_path.name,
        "path": str(pdf_path),
        "metadata": meta,
        "header": f"=== PAPER: {meta['authors_raw']} ({meta['year']}) — {meta['title']} ===",
        "text": text,
    }

    content = paper["header"] + "\n\n" + text
    state: Dict[str, Any] = {
        "version": 2,
        "context": {
            "path": str(pdf_path),
            "loaded_at": time.time(),
            "content": content,
            "papers": [paper],
        },
        "buffers": [],
        "globals": {},
    }
    _save_state(state, state_path)

    print(f"Loaded 1 paper: {meta['bibtex_key']} ({len(text):,} chars)")
    return 0


def cmd_load_corpus(args: argparse.Namespace) -> int:
    """Load all PDFs in a directory into the REPL state.

    Supports resuming an interrupted run: if a partial state.pkl already exists,
    already-loaded PDFs are skipped and extraction continues from where it left off.
    State is saved after every PDF so that timeouts do not lose progress.
    """
    state_path = Path(args.state)
    corpus_dir = Path(args.corpus_dir)

    if not corpus_dir.is_dir():
        raise RlmReplError(f"Directory not found: {corpus_dir}")

    pdf_files = sorted(corpus_dir.glob("*.pdf"))
    if not pdf_files:
        raise RlmReplError(f"No PDF files found in: {corpus_dir}")

    # --- Resume from partial state if it exists ---
    papers: List[Dict[str, Any]] = []
    already_loaded: set = set()
    if state_path.exists():
        try:
            existing = _load_state(state_path)
            papers = list(existing.get("context", {}).get("papers", []))
            already_loaded = {p["filename"] for p in papers}
            if already_loaded:
                print(
                    f"Resuming: {len(papers)} paper(s) already loaded, "
                    f"{len(pdf_files) - len(already_loaded)} remaining.",
                    file=sys.stderr,
                )
        except Exception as e:
            print(f"WARNING: Could not read existing state ({e}); starting fresh.", file=sys.stderr)
            papers, already_loaded = [], set()

    failed: List[str] = []
    new_count = 0

    for pdf_path in pdf_files:
        if pdf_path.name in already_loaded:
            print(f"  Skipping (already loaded): {pdf_path.name}", file=sys.stderr)
            continue

        idx = len(papers)
        print(
            f"  [{idx + 1}/{len(pdf_files)}] Extracting: {pdf_path.name} ...",
            file=sys.stderr,
        )
        try:
            text = _extract_pdf_text(pdf_path)
            meta = _parse_filename_metadata(pdf_path.name)
            header = (
                f"=== PAPER [{idx:03d}]: {meta['authors_raw']} "
                f"({meta['year']}) — {meta['title']} ==="
            )
            paper = {
                "filename": pdf_path.name,
                "path": str(pdf_path),
                "metadata": meta,
                "header": header,
                "text": text,
                "char_start": 0,  # updated by _save_incremental
                "char_end": 0,
            }
            papers.append(paper)
            new_count += 1
            # Save after every PDF so partial progress survives a timeout
            _save_incremental(papers, corpus_dir, state_path)
            # Write per-paper chunk file(s) unless disabled
            if getattr(args, "write_chunks", True):
                chunk_paths = _write_paper_chunks(
                    paper,
                    Path(getattr(args, "chunks_dir", ".opencode/rlm_state/chunks")),
                    size=getattr(args, "chunk_size", 50_000),
                    overlap=getattr(args, "chunk_overlap", 2_000),
                )
                print(
                    f"      Saved. ({len(papers)} total, {len(text):,} chars)"
                    f"  |  {len(chunk_paths)} chunk(s) → chunks/",
                    file=sys.stderr,
                )
            else:
                print(
                    f"      Saved. ({len(papers)} total, {len(text):,} chars)",
                    file=sys.stderr,
                )
        except Exception as e:
            print(f"  WARNING: Failed to extract {pdf_path.name}: {e}", file=sys.stderr)
            failed.append(pdf_path.name)

    # Final save ensures char positions are consistent after all PDFs
    if new_count > 0:
        _save_incremental(papers, corpus_dir, state_path)

    total_chars = sum(len(p["text"]) for p in papers)
    print(f"\nCorpus ready: {len(papers)} papers ({total_chars:,} chars total)")
    if new_count > 0:
        print(f"  New this run: {new_count} paper(s)")
    if len(already_loaded) > 0:
        print(f"  Previously loaded: {len(already_loaded)} paper(s) (skipped)")
    if failed:
        print(f"  Failed to extract {len(failed)} file(s): {', '.join(failed)}")
    print("\nPapers loaded:")
    for i, p in enumerate(papers):
        print(f"  [{i:03d}] {p['metadata']['bibtex_key']:40s}  {len(p['text']):>8,} chars")
    return 0


def cmd_init(args: argparse.Namespace) -> int:
    """Legacy: Initialize state from a plain text file (kept for compatibility)."""
    state_path = Path(args.state)
    ctx_path = Path(args.context)

    # If it's a PDF, redirect to pdf command
    if ctx_path.suffix.lower() == ".pdf":
        args.pdf_path = str(ctx_path)
        return cmd_pdf(args)

    content = _read_text_file(ctx_path, max_bytes=args.max_bytes)

    # Wrap in minimal paper structure
    meta = _parse_filename_metadata(ctx_path.name)
    paper = {
        "filename": ctx_path.name,
        "path": str(ctx_path),
        "metadata": meta,
        "header": f"=== FILE: {ctx_path.name} ===",
        "text": content,
        "char_start": 0,
        "char_end": len(content),
    }

    state: Dict[str, Any] = {
        "version": 2,
        "context": {
            "path": str(ctx_path),
            "loaded_at": time.time(),
            "content": paper["header"] + "\n\n" + content,
            "papers": [paper],
        },
        "buffers": [],
        "globals": {},
    }
    _save_state(state, state_path)

    print(f"Initialized state at: {state_path}")
    print(f"Loaded: {ctx_path} ({len(content):,} chars)")
    return 0


def cmd_status(args: argparse.Namespace) -> int:
    state = _load_state(Path(args.state))
    ctx = state.get("context", {})
    content = ctx.get("content", "")
    papers = ctx.get("papers", [])
    buffers = state.get("buffers", [])
    g = state.get("globals", {})

    print("RLM REPL status")
    print(f"  State file   : {args.state}")
    print(f"  Source       : {ctx.get('path')}")
    print(f"  Papers loaded: {len(papers)}")
    print(f"  Total chars  : {len(content):,}")
    print(f"  Buffers      : {len(buffers)}")
    print(f"  Persisted vars: {len(g)}")
    if args.show_vars and g:
        for k in sorted(g.keys()):
            print(f"    - {k}")
    if papers:
        print("\n  Papers:")
        for i, p in enumerate(papers[:20]):
            meta = p.get("metadata", {})
            print(f"    [{i:03d}] {meta.get('bibtex_key', '?'):40s}  {len(p.get('text','')): >8,} chars")
        if len(papers) > 20:
            print(f"    ... and {len(papers) - 20} more")
    return 0


def cmd_reset(args: argparse.Namespace) -> int:
    state_path = Path(args.state)
    if state_path.exists():
        state_path.unlink()
        print(f"Deleted state: {state_path}")
    else:
        print(f"No state to delete at: {state_path}")

    chunks_dir = state_path.parent / "chunks"
    if chunks_dir.exists():
        import shutil
        shutil.rmtree(chunks_dir)
        print(f"Deleted chunks directory: {chunks_dir}")
    return 0


def cmd_export_buffers(args: argparse.Namespace) -> int:
    state = _load_state(Path(args.state))
    buffers = state.get("buffers", [])
    out_path = Path(args.out)
    _ensure_parent_dir(out_path)
    out_path.write_text("\n\n".join(str(b) for b in buffers), encoding="utf-8")
    print(f"Wrote {len(buffers)} buffers to: {out_path}")
    return 0


def cmd_exec(args: argparse.Namespace) -> int:
    state_path = Path(args.state)
    state = _load_state(state_path)

    ctx = state.get("context")
    if not isinstance(ctx, dict) or "content" not in ctx:
        raise RlmReplError("State is missing a valid 'context'. Re-run load-corpus or pdf.")

    buffers = state.setdefault("buffers", [])
    if not isinstance(buffers, list):
        buffers = []
        state["buffers"] = buffers

    persisted = state.setdefault("globals", {})
    if not isinstance(persisted, dict):
        persisted = {}
        state["globals"] = persisted

    code = args.code
    if code is None:
        code = sys.stdin.read()

    env: Dict[str, Any] = dict(persisted)
    env["context"] = ctx
    env["content"] = ctx.get("content", "")
    env["papers"] = ctx.get("papers", [])
    env["buffers"] = buffers

    helpers = _make_helpers(ctx, buffers)
    env.update(helpers)

    stdout_buf = io.StringIO()
    stderr_buf = io.StringIO()

    try:
        with redirect_stdout(stdout_buf), redirect_stderr(stderr_buf):
            exec(code, env, env)
    except Exception:
        traceback.print_exc(file=stderr_buf)

    # Pull back mutated state
    maybe_ctx = env.get("context")
    if isinstance(maybe_ctx, dict) and "content" in maybe_ctx:
        state["context"] = maybe_ctx
        ctx = maybe_ctx

    maybe_buffers = env.get("buffers")
    if isinstance(maybe_buffers, list):
        state["buffers"] = maybe_buffers

    # Persist new variables
    injected_keys = {"__builtins__", "context", "content", "papers", "buffers", *helpers.keys()}
    to_persist = {k: v for k, v in env.items() if k not in injected_keys}
    filtered, dropped = _filter_pickleable(to_persist)
    state["globals"] = filtered

    _save_state(state, state_path)

    out = stdout_buf.getvalue()
    err = stderr_buf.getvalue()

    if dropped and args.warn_unpickleable:
        msg = "Dropped unpickleable variables: " + ", ".join(dropped)
        err = (err + ("\n" if err else "") + msg + "\n")

    if out:
        sys.stdout.write(_truncate(out, args.max_output_chars))
    if err:
        sys.stderr.write(_truncate(err, args.max_output_chars))

    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="rlm_repl",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=textwrap.dedent(
            """\
            Persistent REPL for RLM-style scientific literature workflows.

            Examples:
              # Load a full corpus of PDFs:
              python rlm_repl.py load-corpus ./context/

              # Load a single PDF:
              python rlm_repl.py pdf ./context/paper.pdf

              # Explore:
              python rlm_repl.py status
              python rlm_repl.py exec -c "print(list_papers())"
              python rlm_repl.py exec -c "print(find_papers('climate'))"
              python rlm_repl.py exec -c "print(cite('Rockström 2009'))"
              python rlm_repl.py exec -c "print(stats())"

              # Chunk for subagent analysis:
              python rlm_repl.py exec -c "paths = write_chunks('.opencode/rlm_state/chunks'); print(paths[:3])"
            """
        ),
    )
    p.add_argument(
        "--state",
        default=str(DEFAULT_STATE_PATH),
        help=f"Path to state pickle (default: {DEFAULT_STATE_PATH})",
    )

    sub = p.add_subparsers(dest="cmd", required=True)

    # load-corpus
    p_corpus = sub.add_parser("load-corpus", help="Load all PDFs from a directory into the corpus")
    p_corpus.add_argument("corpus_dir", help="Path to directory containing PDF files")
    p_corpus.add_argument(
        "--write-chunks",
        dest="write_chunks",
        action="store_true",
        default=True,
        help="Write per-paper chunk files after each PDF is loaded (default: on)",
    )
    p_corpus.add_argument(
        "--no-write-chunks",
        dest="write_chunks",
        action="store_false",
        help="Disable per-paper chunk writing",
    )
    p_corpus.add_argument(
        "--chunks-dir",
        dest="chunks_dir",
        default=".opencode/rlm_state/chunks",
        help="Directory to write per-paper chunk files (default: .opencode/rlm_state/chunks)",
    )
    p_corpus.add_argument(
        "--chunk-size",
        dest="chunk_size",
        type=int,
        default=50_000,
        help="Maximum chars per chunk (default: 50000)",
    )
    p_corpus.add_argument(
        "--chunk-overlap",
        dest="chunk_overlap",
        type=int,
        default=2_000,
        help="Overlap chars between adjacent chunks of the same paper (default: 2000)",
    )
    p_corpus.set_defaults(func=cmd_load_corpus)

    # pdf
    p_pdf = sub.add_parser("pdf", help="Load a single PDF file")
    p_pdf.add_argument("pdf_path", help="Path to the PDF file")
    p_pdf.set_defaults(func=cmd_pdf)

    # init (legacy)
    p_init = sub.add_parser("init", help="Initialize state from a text or PDF file (legacy)")
    p_init.add_argument("context", help="Path to the context file")
    p_init.add_argument(
        "--max-bytes",
        type=int,
        default=None,
        help="Optional cap on bytes read",
    )
    p_init.set_defaults(func=cmd_init)

    # status
    p_status = sub.add_parser("status", help="Show current state summary")
    p_status.add_argument("--show-vars", action="store_true", help="List persisted variable names")
    p_status.set_defaults(func=cmd_status)

    # reset
    p_reset = sub.add_parser("reset", help="Delete the current state file and chunks")
    p_reset.set_defaults(func=cmd_reset)

    # export-buffers
    p_export = sub.add_parser("export-buffers", help="Export buffers to a text file")
    p_export.add_argument("out", help="Output file path")
    p_export.set_defaults(func=cmd_export_buffers)

    # exec
    p_exec = sub.add_parser("exec", help="Execute Python code with persisted state")
    p_exec.add_argument(
        "-c", "--code",
        default=None,
        help="Inline code string. If omitted, reads from stdin.",
    )
    p_exec.add_argument(
        "--max-output-chars",
        type=int,
        default=DEFAULT_MAX_OUTPUT_CHARS,
        help=f"Truncate output to this many chars (default: {DEFAULT_MAX_OUTPUT_CHARS})",
    )
    p_exec.add_argument(
        "--warn-unpickleable",
        action="store_true",
        help="Warn when variables could not be persisted",
    )
    p_exec.set_defaults(func=cmd_exec)

    return p


def main(argv: List[str]) -> int:
    # On Windows the console defaults to cp1252, which cannot encode Unicode
    # characters that appear in PDF text (e.g. special symbols, accented chars).
    # Reconfigure stdout/stderr to UTF-8 with replacement so we never crash.
    if hasattr(sys.stdout, "buffer") and (
        not sys.stdout.encoding
        or sys.stdout.encoding.lower() not in ("utf-8", "utf8")
    ):
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "buffer") and (
        not sys.stderr.encoding
        or sys.stderr.encoding.lower() not in ("utf-8", "utf8")
    ):
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        return int(args.func(args))
    except RlmReplError as e:
        sys.stderr.write(f"ERROR: {e}\n")
        return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
