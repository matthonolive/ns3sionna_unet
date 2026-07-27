#!/usr/bin/env python3
"""
run_campaign.py
==================================================================
Unattended measurement campaign over scenes x legs x traffic seeds.
Replaces the two-terminal ping-pong: for each server-backed leg it
starts the propagation server as a subprocess, WAITS until port 5555
actually accepts connections (no fixed sleeps), runs the ns-3 leg,
then waits for the server to exit (--single_run) before the next leg.

Resumable: progress is tracked in <campaign>/campaign_progress.json
keyed by (scene, leg, seed); completed runs are skipped, so a crashed
or interrupted campaign continues where it left off (delete the entry
or the output files to force a rerun).

Usage (from the ns-3 root):
  python contrib/sionna/model/ns3sionna/run_campaign.py \
      --scenes 'contrib/sionna/model/ns3sionna/worldbuilding/valsuite/v0*' \
               'contrib/sionna/model/ns3sionna/worldbuilding/heldout/seed*' \
      --unet_run /abs/path/runs/residual_cost \
      --seeds 1 2 3 --sim_time 10 --tx_power_dbm 10

  # synthetic-CFR A/B arm (same scenes, separate results subdir):
  ... --synthetic_cfr --results_subdir results_synthcfr

Per run it writes into <scene>/<results_subdir>/:
  {leg}_seed{K}_flow.csv / _pair.csv / _ts.csv / _timing.csv
  {leg}_seed{K}.server.log / .ns3.log

Legs: rt, unet, cost231, logdist, friis (friis needs no server).
Mobility: scenes containing mobility_trace.csv are replayed identically
across legs; other scenes run static. Traffic randomness is varied via
the ns-3 RngRun argument (--ns3_seed_arg, default 'rngRun').
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import shlex
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

LEGS_DEFAULT = ["rt", "unet", "cost231", "logdist", "friis"]
SERVER_LEGS = {"rt", "unet", "cost231", "logdist"}


def port_open(host: str, port: int, timeout=0.5) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def wait_port(host, port, timeout_s, proc=None, label=""):
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        if proc is not None and proc.poll() is not None:
            return False            # server died during startup
        if port_open(host, port):
            return True
        time.sleep(0.5)
    return False


def wait_port_free(host, port, timeout_s):
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        if not port_open(host, port):
            return True
        time.sleep(0.5)
    return False


def terminate(proc, grace_s=10):
    if proc is None or proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=grace_s)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()


def server_cmd(args, leg, env_root):
    cmd = [sys.executable, args.server_py, "--model_folder", str(env_root),
           "--single_run", "--est_csi"]
    if leg == "unet":
        cmd += ["--use_unet", "--unet_run", args.unet_run,
                "--unet_device", args.unet_device,
                "--unet_cov_thresh", str(args.unet_cov_thresh),
                "--unet_tx_cache", str(args.unet_tx_cache)]
    elif leg == "cost231":
        cmd += ["--use_cost231"]
    elif leg == "logdist":
        cmd += ["--use_logdist", "--logdist_exponent", str(args.logdist_exponent),
                "--logdist_ref_m", str(args.logdist_ref_m)]
    if args.synthetic_cfr and leg != "rt":
        cmd += ["--synthetic_cfr"]
    return cmd


def ns3_cmd(args, scene_dir: Path, leg: str, seed: int, res: Path):
    env_xml_rel = f"{scene_dir.name}/scene.xml"
    placements = scene_dir / "placements.csv"
    trace = scene_dir / "mobility_trace.csv"
    if trace.exists():
        mob = f"--enableMobility=0 --mobilityTraceIn={trace} --pokeSionnaPositions=1"
    else:
        mob = "--enableMobility=0 --pokeSionnaPositions=0"

    sim_time = args.sim_time
    exp = scene_dir / "expected.json"
    if args.auto_sim_time and exp.exists():
        hint = json.loads(exp.read_text()).get("sim_time_hint_s")
        if hint:
            sim_time = float(hint)

    prefix = res / f"{leg}_seed{seed}"
    inner = (f"{args.ns3_app} "
             f"--placements={placements} --txPowerDbm={args.tx_power_dbm} "
             f"--simTime={sim_time} --samplePeriod={args.sample_period} {mob} "
             f"--{args.ns3_seed_arg}={seed} "
             f"--timingCsv={prefix}_timing.csv "
             f"--outNodeTsCsv={prefix}_ts.csv "
             f"--outFlowCsv={prefix}_flow.csv --outPairCsv={prefix}_pair.csv")
    if leg == "friis":
        inner = f"{inner} --plModel=friis"
    else:
        inner = (f"{inner} --plModel=sionna --server=tcp://localhost:{args.port} "
                 f"--environment={env_xml_rel}")
    return ["./ns3", "run", inner], prefix


def run_one(args, scene_dir: Path, leg: str, seed: int, res: Path) -> bool:
    cmd, prefix = ns3_cmd(args, scene_dir, leg, seed, res)
    ns3_log = Path(f"{prefix}.ns3.log")
    srv_log = Path(f"{prefix}.server.log")

    if args.dry_run:
        print(f"    [dry] server: "
              f"{' '.join(shlex.quote(c) for c in server_cmd(args, leg, scene_dir.parent)) if leg in SERVER_LEGS else '(none)'}")
        print(f"    [dry] ns-3  : {' '.join(shlex.quote(c) for c in cmd)}")
        return True

    srv = None
    try:
        if leg in SERVER_LEGS:
            if not wait_port_free("localhost", args.port, args.server_timeout_s):
                print(f"    [err] port {args.port} still busy; aborting leg")
                return False
            srv = subprocess.Popen(server_cmd(args, leg, scene_dir.parent),
                                   stdout=open(srv_log, "w"),
                                   stderr=subprocess.STDOUT,
                                   cwd=args.ns3_root)
            if not wait_port("localhost", args.port, args.server_timeout_s,
                             proc=srv, label=leg):
                print(f"    [err] server for '{leg}' never opened port "
                      f"{args.port} (see {srv_log})")
                terminate(srv)
                return False

        t0 = time.time()
        rc = subprocess.run(cmd, stdout=open(ns3_log, "w"),
                            stderr=subprocess.STDOUT, cwd=args.ns3_root,
                            timeout=args.ns3_timeout_s).returncode
        dt = time.time() - t0
        if rc != 0:
            print(f"    [err] ns-3 leg '{leg}' rc={rc} (see {ns3_log})")
            terminate(srv)
            return False

        if srv is not None:
            try:
                srv.wait(timeout=args.server_timeout_s)
            except subprocess.TimeoutExpired:
                print(f"    [warn] server didn't exit after ns-3 finished; killing")
                terminate(srv)

        ok = Path(f"{prefix}_flow.csv").exists()
        if not ok:
            print(f"    [err] leg finished but {prefix}_flow.csv missing")
            return False
        # persist the measured ns-3 wall time (server startup excluded --
        # the clock starts after the port is up) for the cost-fidelity table
        Path(f"{prefix}_runmeta.json").write_text(json.dumps(
            {"wall_s": round(dt, 2), "leg": leg, "seed": seed}))
        print(f"    [ok] {leg} seed={seed} in {dt:.1f}s")
        return True

    except subprocess.TimeoutExpired:
        print(f"    [err] ns-3 leg '{leg}' timed out after {args.ns3_timeout_s}s")
        terminate(srv)
        return False
    except KeyboardInterrupt:
        terminate(srv)
        raise
    finally:
        terminate(srv)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", nargs="+", required=True,
                    help="scene dir globs (each dir must contain placements.csv)")
    ap.add_argument("--legs", nargs="+", default=LEGS_DEFAULT,
                    choices=LEGS_DEFAULT)
    ap.add_argument("--seeds", nargs="+", type=int, default=[1],
                    help="ns-3 RngRun values (traffic randomness)")
    ap.add_argument("--results_subdir", type=str, default="results")

    ap.add_argument("--ns3_root", type=str, default=".",
                    help="dir containing ./ns3 (run the script from there)")
    ap.add_argument("--ns3_app", type=str,
                    default="ns3sionna-example-sionna-traceflow")
    ap.add_argument("--ns3_seed_arg", type=str, default="rngRun",
                    help="ns-3 CLI arg used to vary traffic randomness")
    ap.add_argument("--server_py", type=str,
                    default="contrib/sionna/model/ns3sionna/ns3unet_spectrum.py")
    ap.add_argument("--port", type=int, default=5555)

    ap.add_argument("--unet_run", type=str, default="")
    ap.add_argument("--unet_device", type=str, default="cuda:0")
    ap.add_argument("--unet_cov_thresh", type=float, default=0.5)
    ap.add_argument("--unet_tx_cache", type=int, default=256)
    ap.add_argument("--logdist_exponent", type=float, default=3.0)
    ap.add_argument("--logdist_ref_m", type=float, default=1.0)
    ap.add_argument("--synthetic_cfr", action="store_true")

    ap.add_argument("--sim_time", type=float, default=10.0)
    ap.add_argument("--auto_sim_time", action="store_true", default=True,
                    help="use expected.json sim_time_hint_s when present")
    ap.add_argument("--tx_power_dbm", type=float, default=10.0)
    ap.add_argument("--sample_period", type=float, default=0.05)

    ap.add_argument("--server_timeout_s", type=float, default=180.0)
    ap.add_argument("--ns3_timeout_s", type=float, default=3600.0)
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()

    if "unet" in args.legs and not args.unet_run and not args.dry_run:
        ap.error("--unet_run is required when the unet leg is enabled")

    scene_dirs = []
    for pat in args.scenes:
        scene_dirs += [Path(p) for p in sorted(glob.glob(pat))]
    scene_dirs = [s for s in scene_dirs
                  if s.is_dir() and (s / "placements.csv").exists()]
    if not scene_dirs:
        raise SystemExit("no scene dirs matched (need placements.csv inside)")

    progress_path = Path(args.ns3_root) / "campaign_progress.json"
    progress = json.loads(progress_path.read_text()) if progress_path.exists() else {}

    total = len(scene_dirs) * len(args.legs) * len(args.seeds)
    done0 = sum(1 for v in progress.values() if v == "ok")
    print(f"[campaign] {len(scene_dirs)} scenes x {len(args.legs)} legs x "
          f"{len(args.seeds)} seeds = {total} runs "
          f"({done0} already done in progress file)")

    failures = []
    for si, scene in enumerate(scene_dirs, 1):
        res = scene / args.results_subdir
        res.mkdir(exist_ok=True)
        print(f"\n[{si}/{len(scene_dirs)}] {scene}")
        for seed in args.seeds:
            for leg in args.legs:
                key = f"{scene}::{args.results_subdir}::{leg}::seed{seed}"
                flow = res / f"{leg}_seed{seed}_flow.csv"
                if progress.get(key) == "ok" and flow.exists():
                    print(f"    [skip] {leg} seed={seed} (done)")
                    continue
                ok = run_one(args, scene, leg, seed, res)
                if not args.dry_run:
                    progress[key] = "ok" if ok else "failed"
                    progress_path.write_text(json.dumps(progress, indent=1))
                if not ok:
                    failures.append(key)

    print("\n[campaign] finished.")
    if failures:
        print(f"[campaign] {len(failures)} FAILED runs (rerun the script to retry):")
        for k in failures:
            print(f"    {k}")
        sys.exit(1)
    print("[campaign] aggregate with:")
    print(f"    python aggregate_campaign.py --scenes {' '.join(repr(s) for s in args.scenes)} "
          f"--results_subdir {args.results_subdir} --seeds "
          + " ".join(str(s) for s in args.seeds))


if __name__ == "__main__":
    main()