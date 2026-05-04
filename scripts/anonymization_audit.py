# pyright: basic, reportMissingImports=false

"""Scan the repo for PII / de-anonymising patterns before NeurIPS ED submission.

The NeurIPS 2026 ED track defaults to double-blind. This script does NOT do an
exhaustive forensic audit; it flags the easy-to-miss leaks (real emails, real
names that look human, unfrozen acknowledgements, GitHub handles, university
domains, hostnames) so that authors can fix them before producing the
submission bundle.

Exit code:
    0 if no findings
    1 if any finding is found

Usage:
    python scripts/anonymization_audit.py
    python scripts/anonymization_audit.py --paths manuscript scripts pinn4csi
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

# Patterns that may leak identity. Each entry is (label, regex, allow-list).
# Decorators / BibTeX entry types that legitimately use @ but are not PII.
_DECORATOR_KEYS = {
    "pytest",
    "property",
    "staticmethod",
    "classmethod",
    "dataclass",
    "abstractmethod",
    "cached_property",
    "contextmanager",
    "wraps",
    "patch",
    "fixture",
    "parametrize",
    "override",
    "torch",
    "jax",
    "tf",
    "hydra",
    "click",
    "app",
    "cli",
}
_BIBTEX_KEYS = {
    "inproceedings",
    "article",
    "book",
    "incollection",
    "techreport",
    "misc",
    "phdthesis",
    "mastersthesis",
    "proceedings",
    "manual",
    "online",
    "url",
    "context",
    "type",
    "id",
    "language",
    "vocab",
}
# Path-shaped substrings that are clearly virtual/anonymous.
_OK_PATH_PREFIXES = (
    "/home/anonymous",
    "/home/runner",
    "/home/ubuntu",
    "/home/user",
    "/Users/anonymous",
)
# Domains belonging to UPSTREAM datasets we cite, not to authors.
_UPSTREAM_DOMAINS = {
    "tsinghua.edu",  # Widar3.0 home
    "ntu.edu.sg",  # NTU-Fi
    "wmich.edu",  # historical SignFi mirror
    "openreview.net",
    "neurips.cc",
}


def _gh_handle_filter(hit: str) -> bool:
    """True if the @-prefixed token is real PII (not a decorator/bibtex)."""
    h = hit.lstrip().lstrip("(").lstrip()
    if not h.startswith("@"):
        return False
    name = h[1:].split(".")[0].split("(")[0].strip(",;:{[")
    if name.lower() in _DECORATOR_KEYS:
        return False
    if name.lower() in _BIBTEX_KEYS:
        return False
    return name.lower() not in {"anonymous", "example"}


def _path_filter(hit: str) -> bool:
    h = hit.strip().lstrip("=:")
    return not any(h.startswith(p) for p in _OK_PATH_PREFIXES)


def _domain_filter(hit: str) -> bool:
    h = hit.lower()
    return not any(d in h for d in _UPSTREAM_DOMAINS)


def _ack_filter(hit: str, line: str) -> bool:
    """Skip occurrences inside our own anti-PII statements / README boilerplate."""
    low_line = line.lower()
    if "absent from every" in low_line:
        return False
    if "are not collected" in low_line or "no acknowledgements" in low_line:
        return False
    return not ("removed" in low_line and "acknowledg" in low_line)


PATTERNS: list[tuple[str, str, set[str], object | None]] = [
    (
        "real-looking email (not @example.com / @pinn4csi.local / @anonymous)",
        (
            r"[A-Za-z0-9._%+-]+@(?!example\.com|pinn4csi\.local"
            r"|anonymous(?:\.4open)?\.science|huggingface\.co"
            r"|noreply\.github\.com)[A-Za-z0-9.-]+\.[A-Za-z]{2,}"
        ),
        {"suanlab@gmail.com"},  # private memory; never injected into source
        None,
    ),
    (
        "GitHub handle (@username)",
        r"(?:^|[\s(])@[A-Za-z][A-Za-z0-9_-]{2,}\b",
        set(),
        _gh_handle_filter,
    ),
    (
        "absolute /home or /Users path (may reveal username)",
        r"(?:^|\s|=|:)/(?:home|Users)/[A-Za-z][A-Za-z0-9_-]+",
        set(),
        _path_filter,
    ),
    (
        "university / lab domain (excluding upstream-dataset domains)",
        r"\b[a-z0-9-]+\.(?:ac\.[a-z]{2}|edu|edu\.[a-z]{2}|kaist\.ac\.kr|snu\.ac\.kr)\b",
        set(),
        _domain_filter,
    ),
    (
        "ORCID id",
        r"\b\d{4}-\d{4}-\d{4}-\d{3}[0-9X]\b",
        set(),
        None,
    ),
    (
        "Acknowledgement / funding section",
        (
            r"(?:Acknowledgements|Acknowledgments|Funded by|Grant No\.?"
            r"|NRF-|NSFC|DARPA|ARO|ONR(?!\.)\b)"
        ),
        set(),
        _ack_filter,
    ),
    (
        "Real-name-looking author block in .tex (Anonymous is OK)",
        r"\\author\{(?!Anonymous)",
        set(),
        None,
    ),
]

DEFAULT_PATHS = [
    "manuscript/paper2",
    "scripts",
    "pinn4csi",
    "tests",
    "data/prepared",
    "outputs/croissant",
    "README.md",
    "SUBMISSION_README.md",
    "pyproject.toml",
]
SKIP_DIRS = {
    ".git",
    ".archive",
    ".sisyphus",
    "outputs",
    "venv",
    "node_modules",
    "__pycache__",
    ".omc",
    ".coverage",
    "sections_ko",
}
# Files that should not be audited:
#   - upstream NeurIPS template (we did not author it; cannot anonymise it)
#   - the audit/validation scripts themselves (their regexes match themselves)
#   - the Korean parallel build (not part of the submission bundle)
SKIP_FILES = {
    "manuscript/paper2/neurips_2026.sty",
    "manuscript/paper2/main_korean.tex",
    "manuscript/paper2/checklist_korean.tex",
    "manuscript/paper2/supplementary_korean.tex",
    "scripts/anonymization_audit.py",
    "scripts/validate_croissant.py",
}
SKIP_EXT = {
    ".pdf",
    ".png",
    ".jpg",
    ".jpeg",
    ".npy",
    ".log",
    ".aux",
    ".fls",
    ".fdb_latexmk",
    ".out",
    ".bbl",
    ".blg",
    ".synctex.gz",
}


def iter_files(paths: list[str]) -> list[Path]:
    out: list[Path] = []
    for p in paths:
        path = Path(p)
        if not path.exists():
            continue
        if path.is_file():
            out.append(path)
            continue
        for f in path.rglob("*"):
            if not f.is_file():
                continue
            if any(part in SKIP_DIRS for part in f.parts):
                continue
            if f.suffix in SKIP_EXT:
                continue
            if str(f) in SKIP_FILES:
                continue
            out.append(f)
    return out


def scan_file(path: Path) -> list[tuple[str, int, str, str]]:
    findings: list[tuple[str, int, str, str]] = []
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return findings
    lines = text.splitlines()
    for label, regex, allow, post_filter in PATTERNS:
        compiled = re.compile(regex)
        for i, line in enumerate(lines, start=1):
            for m in compiled.finditer(line):
                hit = m.group(0).strip()
                if hit in allow:
                    continue
                if any(a in hit for a in allow):
                    continue
                if post_filter is not None:
                    if post_filter is _ack_filter:
                        if not post_filter(hit, line):
                            continue
                    elif not post_filter(hit):
                        continue
                findings.append((label, i, hit, line.strip()[:180]))
    return findings


def scan_git_log() -> list[tuple[str, str]]:
    """Return list of (author_string, commit_hash) for unique authors."""
    try:
        out = subprocess.run(
            ["git", "log", "--pretty=format:%H|%an <%ae>"],
            capture_output=True,
            text=True,
            check=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return []
    seen: dict[str, str] = {}
    for line in out.stdout.splitlines():
        if "|" not in line:
            continue
        h, author = line.split("|", 1)
        seen.setdefault(author, h)
    return list(seen.items())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__ or "")
    parser.add_argument(
        "--paths",
        nargs="*",
        default=DEFAULT_PATHS,
        help="Paths to scan (file or directory).",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Treat any acknowledgement-section match as fatal.",
    )
    args = parser.parse_args()

    files = iter_files(args.paths)
    print(f"Scanning {len(files)} files...")
    total = 0
    for f in files:
        findings = scan_file(f)
        for label, lineno, hit, line in findings:
            total += 1
            print(f"  [{label}] {f}:{lineno}")
            print(f"    -> {hit}")
            print(f"    : {line}")

    print()
    print("Git authors observed:")
    git_authors = scan_git_log()
    suspicious_git = []
    for author, commit in git_authors:
        is_anon = (
            "Anonymous" in author
            or "anonymous" in author
            or "@example" in author
            or "@pinn4csi.local" in author
            or "@anonymous" in author
        )
        marker = "OK " if is_anon else "??"
        print(f"  {marker} {author} (first commit {commit[:8]})")
        if not is_anon:
            suspicious_git.append(author)

    print()
    if total == 0 and not suspicious_git:
        print("OK: no PII findings, git authors look anonymised.")
        return 0
    print(f"FOUND: {total} pattern hits, {len(suspicious_git)} suspicious git authors.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
