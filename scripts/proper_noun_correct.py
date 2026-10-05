#!/usr/bin/env python3
"""Post-transcription proper-noun correction pass (run on EVERY transcript).

Whisper garbles proper nouns systematically (place names, org names,
people's names). This script applies a list of regex substitutions to a
transcript .md, writes the corrected file in place, and appends a log of
what was changed to the output dir so the user can audit it.

Usage:
  proper_noun_correct.py TRANSCRIPT_MD [CORRECTIONS_YAML]

CORRECTIONS_YAML defaults to the `corrections.yaml` in the transcript's
parent tree (first one found walking up from the transcript dir), then to
~/.hermes/skills/audio-meeting-batch/corrections.yaml. A per-project
file (e.g. ~/Documents/<project>/corrections.yaml) is where project-
specific names live; the skill-level one holds cross-project fallbacks.

CORRECTIONS_YAML: list of {pattern, replacement, why} entries (case
sensitive). Keep patterns narrow (word boundaries, phrases) — they run on
verbatim transcripts; a bad pattern corrupts the record.

LOG: <TRANSCRIPT_MD> with .md -> _corrections.md.

Exit codes: 0 = ok (even 0 fixes), 1 = usage/read error, 2 = no
corrections file found anywhere.
"""
import re
import sys
from pathlib import Path

SKILL_DEFAULT = Path.home() / ".hermes/skills/audio-meeting-batch/corrections.yaml"


def load_corrections(path):
    """Minimal YAML-subset parser: list of {pattern, replacement, why}.

    Avoids a PyYAML dependency (runs on bare system python3). Supports
    one entry per 3-line block: '- pattern: "..."', 'replacement: "..."',
    'why: "..."'. Values may use single or double quotes. Escapes: \\ ->
    backslash, \" inside double-quoted, \\' inside single-quoted.
    """
    entries = []
    cur = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.rstrip()
        if not line or line.lstrip().startswith("#"):
            continue
        s = line.strip()
        if s.startswith("- pattern:"):
            if cur:
                entries.append(cur)
            cur = {}
            s = s[len("- pattern:"):]
        elif s.startswith("replacement:"):
            s = s[len("replacement:"):]
        elif s.startswith("why:"):
            s = s[len("why:"):]
        else:
            raise ValueError(f"unrecognized line: {raw!r}")
        s = s.strip()
        if s and s[0] in "\"'":
            q = s[0]
            end = len(s)
            i = 1
            while i < end:
                if s[i] == "\\" and i + 1 < len(s):
                    i += 2
                    continue
                if s[i] == q:
                    end = i
                    break
                i += 1
            val = s[1:end]
            if q == '"':
                val = val.replace('\\"', '"').replace("\\\\", "\\")
            else:
                val = val.replace("\\'", "'").replace("\\\\", "\\")
        else:
            val = s
        if raw.lstrip().startswith("- pattern:"):
            cur["pattern"] = val
        elif raw.lstrip().startswith("replacement:"):
            cur["replacement"] = val
        elif raw.lstrip().startswith("why:"):
            cur["why"] = val
    if cur:
        entries.append(cur)
    for e in entries:
        if "pattern" not in e or "replacement" not in e:
            raise ValueError(f"entry missing pattern/replacement: {e}")
    return entries


def find_corrections(explicit, transcript):
    if explicit:
        p = Path(explicit).expanduser()
        return p if p.is_file() else None
    for d in [transcript.parent, *transcript.parent.parents]:
        c = d / "corrections.yaml"
        if c.is_file():
            return c
    return SKILL_DEFAULT if SKILL_DEFAULT.is_file() else None


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    transcript = Path(sys.argv[1]).expanduser()
    corr_file = find_corrections(sys.argv[2] if len(sys.argv) > 2 else None,
                                 transcript)
    if corr_file is None:
        print(f"ERROR: no corrections file (walked up from {transcript.parent}; "
              f"tried {SKILL_DEFAULT})", file=sys.stderr)
        sys.exit(2)

    if not transcript.is_file():
        print(f"ERROR: transcript not found: {transcript}", file=sys.stderr)
        sys.exit(1)
    log = transcript.with_name(transcript.stem + "_corrections.md")

    data = load_corrections(corr_file)
    if not isinstance(data, list):
        print("ERROR: corrections file must be a YAML list of "
              "{pattern, replacement, why}", file=sys.stderr)
        sys.exit(1)

    text = transcript.read_text(encoding="utf-8")
    lines = [
        f"# Corrección de proper nouns — `{transcript.name}`",
        "",
        f"Corrections file: `{corr_file}`",
        "",
    ]
    total = 0
    for entry in data:
        pattern = entry["pattern"]
        repl = entry["replacement"]
        why = entry.get("why", "")
        out, n = re.subn(pattern, repl, text)
        total += n
        if n:
            lines.append(f"- `{pattern}` -> `{repl}` ({n}x) — {why}")
        text = out
    if not total:
        lines.append("- (ninguna coincidencia)")

    transcript.write_text(text, encoding="utf-8")
    log.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"FIXES: {total} in {transcript} (log: {log})")


if __name__ == "__main__":
    main()
