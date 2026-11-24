"""Replace only the original token_oe 300-draw seed after the time-class fix."""
from datetime import datetime
import json
import os
from pathlib import Path
import signal
import subprocess
import time

P = Path(__file__).resolve().parents[1]
old = P / "scripts/results_100k/neue_punkte_online_token_oe_ue_suffixe300_seed0"
new = P / "scripts/results_100k/neue_punkte_online_token_oe_ue_suffixe300_seed0_v2"
runner = str(P / "scripts/run_token_oe_ue_seed.py")
if new.exists():
    raise FileExistsError(new)
targets = []
for entry in Path("/proc").iterdir():
    if not entry.name.isdigit():
        continue
    try:
        if entry.stat().st_uid != os.getuid():
            continue
        args = entry.joinpath("cmdline").read_bytes().decode().rstrip("\0").split("\0")
        if runner in args and "--out" in args and args[args.index("--out") + 1] == str(old):
            targets.append(int(entry.name))
    except (OSError, IndexError):
        continue
if len(targets) > 1:
    raise RuntimeError(f"Ambiguous original process: {targets}")
for pid in targets:
    os.kill(pid, signal.SIGTERM)
    for _ in range(50):
        try:
            state = Path(f"/proc/{pid}/stat").read_text().split(") ", 1)[1].split()[0]
        except FileNotFoundError:
            break
        if state == "Z":
            break
        time.sleep(0.2)
    else:
        raise RuntimeError(f"Original process {pid} did not stop")
progress_file = old / "progress.json"
progress = json.loads(progress_file.read_text())
progress.update(status="restarted", replacement_out=str(new),
                restart_reason="Transition source is now taken from raw activity/lifecycle, independently of an unknown time class",
                updated_at=datetime.now().astimezone().isoformat(timespec="seconds"))
progress_file.write_text(json.dumps(progress, indent=2, ensure_ascii=False) + "\n")
subprocess.run(["tmux", "new-session", "-d", "-s", "psp300_oe_ue_s0_v2", "-c",
                str(P.parent / "process-simulation-engine"), "bash",
                str(P / "data/bpic17_psp_token_oe_ue_seed0_v2.sh")], check=True)
print(json.dumps({"stopped_pids": targets, "replacement": str(new)}))
