from __future__ import annotations
from pathlib import Path
from sim_config import SimConfig
import logging
import re
import shlex
import time
import subprocess


log = logging.getLogger("slurm")

_SBATCH_ID_RE = re.compile(r"Submitted batch job (\d+)")


def _sbatch_script_text(cfg: SimConfig, run_dir: Path) -> str:
    s = cfg.slurm
    lines = ["#!/usr/bin/env bash"]
    # core SBATCH
    lines += [
        f"#SBATCH --partition={s.partition}",
        f"#SBATCH --ntasks={s.ntasks}",
        f"#SBATCH --cpus-per-task={s.cpus_per_task}",
        f"#SBATCH --time={s.time}",
        f"#SBATCH --output={run_dir}/slurm-%j.out",
        f"#SBATCH --error={run_dir}/slurm-%j.err",
    ]
    if s.account:
        lines.append(f"#SBATCH --account={s.account}")
    if s.mem:
        lines.append(f"#SBATCH --mem={s.mem}")
    if s.qos:
        lines.append(f"#SBATCH --qos={s.qos}")
    if s.nodes:
        lines.append(f"#SBATCH --nodes={s.nodes}")
    if s.constraint:
        lines.append(f"#SBATCH --constraint={shlex.quote(s.constraint)}")
    if s.gpus:
        lines.append(f"#SBATCH --gpus={s.gpus}")
    if s.chdir_to_run_dir:
        lines.append(f"#SBATCH --chdir={run_dir}")

    # extras verbatim
    for raw in s.extra_sbatch:
        lines.append(raw)

    # environment setup
    if s.env_setup:
        lines.append(s.env_setup)

    repo_root = Path(__file__).resolve().parent  # or make it a config field
    md_file = repo_root / "run_sim_MD.py"
    run_dir_abs = run_dir.resolve()
    print(run_dir_abs)

    # make matplotlib headless; reuse your logging/progress infra
    lines += [
        # "export MPLBACKEND=Agg",
        # IMPORTANT: run the same MD entrypoint your local path does
        # If run_sim_MD supports a CLI flag to point to the run dir, use it:
        f"{shlex.quote(s.python)} {shlex.quote(str(md_file))} --from-dir {shlex.quote(str(run_dir_abs))}",
    ]
    return "\n".join(lines) + "\n"


def _run(cmd: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, check=True, text=True, capture_output=True)


def run_md_slurm(cfg: SimConfig, run_dir: Path) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    script_path = run_dir / "md_job.sbatch"
    script_path.write_text(_sbatch_script_text(cfg, run_dir))
    log.info("Submitting SLURM job with %s", script_path.name)
    res = _run(["sbatch", str(script_path)])
    log.info(res.stdout.strip())

    m = _SBATCH_ID_RE.search(res.stdout)
    if not m:
        raise RuntimeError(f"Could not parse job id from sbatch output: {res.stdout}")
    jobid = m.group(1)
    (run_dir / "slurm_jobid.txt").write_text(jobid)

    if not cfg.slurm.wait:
        log.info("Submitted job %s (not waiting).", jobid)
        return

    # Poll until job leaves the queue; then check final state via sacct
    log.info("Waiting for job %s to finish...", jobid)
    while True:
        q = subprocess.run(
            ["squeue", "-j", jobid, "-h"], text=True, capture_output=True
        )
        if q.returncode == 0 and q.stdout.strip() == "":
            break  # not in queue anymore
        time.sleep(10)

    # Check final state
    sacct = subprocess.run(
        ["sacct", "-j", jobid, "--format=JobID,State,ExitCode", "-P", "-n"],
        text=True,
        capture_output=True,
    )
    state_line = sacct.stdout.strip().splitlines()[0] if sacct.stdout.strip() else ""
    log.info("sacct: %s", state_line or "(no sacct output)")
    if "COMPLETED" in state_line and (
        "0:0" in state_line or "0" in state_line.split("|")[-1]
    ):
        log.info("Job %s completed successfully.", jobid)
    else:
        log.error("Job %s finished with non-success state: %s", jobid, state_line)
        # surface a nonzero exit to upstream
        raise SystemExit(1)
