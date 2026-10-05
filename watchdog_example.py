#!/usr/bin/env python3
"""Example cron watchdog for a multi-hour batch run (see SKILL.md, "Long runs").

Run as a no_agent cron job every 5 minutes. It prints NOTHING while the job
is running and no output exists (no_agent + empty stdout = no delivery). The
first time the output file appears (or the PID dies with no output) it prints
ONE line and writes a .state file so every later tick stays silent (one-shot).

Usage (cron):
  cronjob_manage(action="create", name="batch-watch", script="watchdog_example.py",
                 no_agent=True, schedule="every 5m", repeat=120,
                 deliver="origin", failure_deliver="local", enabled_toolsets=[])
Edit the three constants below before creating the job.
"""
import os
import pathlib
import re

# --- EDIT THESE -------------------------------------------------------------
PID = "12345"                                   # the batch driver's PID
OUT = pathlib.Path("~/Documents/project/transcripts/01_...-a.md").expanduser()
STATE = OUT.with_name(OUT.stem + ".watch.state")
# ---------------------------------------------------------------------------


def alive(pid: str) -> bool:
    try:
        os.kill(int(pid), 0)
        return True
    except (ProcessLookupError, ValueError):
        return False


def main() -> None:
    if STATE.exists():
        return  # already reported once
    if OUT.exists() and OUT.stat().st_size > 1000:
        print(f"READY: {OUT} ({OUT.stat().st_size} bytes)")
        STATE.write_text(f"ready {OUT}\n")
        return
    if not alive(PID) and not OUT.exists():
        print(f"FAILED: batch PID {PID} exited with no output at {OUT}")
        STATE.write_text(f"failed pid={PID}\n")
        # (no output while running -> no delivery, as intended)


if __name__ == "__main__":
    main()
