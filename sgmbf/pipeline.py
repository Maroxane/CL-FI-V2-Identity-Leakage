"""Runs several steps one after another in child processes (one process tree => one 'Stop' kills all).

    python -m sgmbf.pipeline --steps extract,main,fusion,ablation,robustness,fairness
"""
import argparse
import subprocess
import sys
import time

from .common import State, load_config, remove_pid, write_pid

STEP_CMD = {
    "download": ["-m", "sgmbf.download"],
    "mupta_extract": ["-m", "sgmbf.mupta"],
    "mupta_cross": ["-m", "sgmbf.cross"],
    "extract": ["-m", "sgmbf.extract"],
    "main": ["-m", "sgmbf.experiments", "--plan", "main"],
    "fusion": ["-m", "sgmbf.experiments", "--plan", "fusion"],
    "ablation": ["-m", "sgmbf.experiments", "--plan", "ablation"],
    "robustness": ["-m", "sgmbf.experiments", "--plan", "robustness"],
    "fairness": ["-m", "sgmbf.experiments", "--plan", "fairness"],
    "gate_select": ["-m", "sgmbf.experiments", "--plan", "gate_select"],
    "gate_final": ["-m", "sgmbf.experiments", "--plan", "gate_final"],
    "identity": ["-m", "sgmbf.identity"],
    "leakage": ["-m", "sgmbf.leakage"],
    "phase2": ["-m", "sgmbf.phase2"],
    "control": ["-m", "sgmbf.control"],
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", default="extract,main")
    args = ap.parse_args()
    cfg = load_config()
    state = State(cfg)
    write_pid(cfg, "pipeline")
    steps = [s.strip() for s in args.steps.split(",") if s.strip()]
    try:
        for i, s in enumerate(steps, 1):
            state.update(pipeline=steps, pipeline_step=i, pipeline_current=s)
            rc = subprocess.call([sys.executable] + STEP_CMD[s])
            if rc != 0:
                state.update(status="error", error=f"step '{s}' exited with code {rc}")
                sys.exit(rc)
        state.update(status="finished", stage="Pipeline finished: " + ", ".join(steps), finished=time.time())
    finally:
        remove_pid(cfg, "pipeline")


if __name__ == "__main__":
    main()
