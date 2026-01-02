#!/usr/bin/env python3
import argparse, subprocess, sys, time, re
from pathlib import Path
import csv

RT_RE  = re.compile(r'^\[RT\]\s+tx=(\d+)\s+rx=(\d+).*?d=([\d.]+)m.*?delay=(\d+)ns.*?wb=([\d.]+)dB.*?tau_rms=([\d.]+)ns')
CFR_RE = re.compile(r'^\[CFR\]\s+tx=(\d+)\s+rx=(\d+).*?d=([\d.]+)\s+m.*?delay=(\d+)\s+ns.*?tau_rms=([\d.]+)\s+ns')
FRIIS_RE = re.compile(r'^\[FRIIS\]\s+tx=(\d+)\s+rx=(\d+).*?d=([\d.]+)\s+m\s+fspl=([\d.]+)\s+dB')

def run_cmd(cmd, cwd=None, timeout=None):
    p = subprocess.Popen(cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    lines = []
    try:
        for line in p.stdout:
            lines.append(line.rstrip("\n"))
        rc = p.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        p.kill()
        rc = -1
    return rc, lines

def parse_lines(lines):
    rows = []
    friis = []
    for ln in lines:
        m = RT_RE.match(ln)
        if m:
            tx, rx, d, delay, wb, tau = m.groups()
            rows.append(("rt", int(tx), int(rx), float(d), int(delay), float(wb), float(tau)))
            continue
        m = CFR_RE.match(ln)
        if m:
            tx, rx, d, delay, tau = m.groups()
            # UNet prints wb separately ("rx=... wb=..."), so we keep wb as NaN here unless you also print it in CFR line
            rows.append(("unet_cfr", int(tx), int(rx), float(d), int(delay), float("nan"), float(tau)))
            continue
        m = FRIIS_RE.match(ln)
        if m:
            tx, rx, d, fspl = m.groups()
            friis.append((int(tx), int(rx), float(d), float(fspl)))
    return rows, friis

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite_root", required=True, help="Root containing scene dirs (each has scene.xml + placements.csv)")
    ap.add_argument("--ns3", default="./ns3", help="Path to ns3 runner")
    ap.add_argument("--example", default="ns3sionna-example-sionna-montecarlo", help="Built example name")
    ap.add_argument("--python", default=sys.executable, help="Python executable for server")
    ap.add_argument("--server_py", required=True, help="Path to ns3unet_spectrum.py")
    ap.add_argument("--models", default="rt,unet,friis", help="Comma list: rt,unet,friis")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--channel", type=int, default=42)
    ap.add_argument("--channelWidth", type=int, default=80)
    ap.add_argument("--txPowerDbm", type=float, default=20.0)
    ap.add_argument("--simTime", type=float, default=3.0)
    ap.add_argument("--maxPackets", type=int, default=3)
    ap.add_argument("--interval", type=float, default=0.3)
    ap.add_argument("--packetSize", type=int, default=1024)
    ap.add_argument("--staIndex", type=int, default=0)

    # UNet options
    ap.add_argument("--unet_run", default="unet", help="Run dir containing model.pt/meta.json/norm_stats.npz")
    ap.add_argument("--unet_device", default="cuda")
    ap.add_argument("--est_csi", action="store_true", help="Pass --est_csi to server (needed for spectrum)")
    ap.add_argument("--out_csv", default="mc_results.csv")
    args = ap.parse_args()

    suite_root = Path(args.suite_root).resolve()
    scene_dirs = sorted([p for p in suite_root.iterdir() if p.is_dir() and (p/"scene.xml").exists() and (p/"placements.csv").exists()])
    if not scene_dirs:
        print("No scene dirs found under", suite_root)
        sys.exit(2)

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    out_csv = Path(args.out_csv).resolve()

    with out_csv.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["scene","rep","model","tx","rx","d_m","delay_ns","wb_db","tau_rms_ns","friis_fspl_db","raw_log"])

        for scene_dir in scene_dirs:
            rel_env = str(scene_dir.relative_to(suite_root) / "scene.xml")
            placements = str(scene_dir / "placements.csv")

            for rep in range(args.reps):
                for model in models:
                    # start server for rt/unet
                    server_lines = []
                    if model in ("rt","unet"):
                        srv_cmd = [
                            args.python, str(Path(args.server_py).resolve()),
                            "--model_folder", str(suite_root),
                            "--single_run",
                        ]
                        if args.est_csi:
                            srv_cmd.append("--est_csi")
                        if model == "unet":
                            srv_cmd += ["--use_unet", "--unet_run", args.unet_run, "--unet_device", args.unet_device]

                        # run server in background
                        srv = subprocess.Popen(srv_cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)

                        # give it a moment to bind
                        time.sleep(0.25)

                    ns3_cmd_str = (
                        f'{args.example} '
                        f'--propModel={("friis" if model=="friis" else "rt")} '
                        f'--environment={rel_env} '
                        f'--placementsCsv={placements} '
                        f'--staIndex={args.staIndex} '
                        f'--channel={args.channel} --channelWidth={args.channelWidth} '
                        f'--txPowerDbm={args.txPowerDbm} '
                        f'--simTime={args.simTime} '
                        f'--maxPackets={args.maxPackets} --interval={args.interval} --packetSize={args.packetSize} '
                        f'--caching=0 --verbose=0 --tracing=0 '
                    )

                    # IMPORTANT: the C++ propModel for rt/unet is the same on ns-3 side;
                    #            the difference is which server you started.
                    if model == "unet":
                        ns3_cmd_str = ns3_cmd_str.replace("--propModel=rt", "--propModel=unet")

                    ns3_cmd = [args.ns3, "run", ns3_cmd_str]
                    rc, ns3_lines = run_cmd(ns3_cmd, cwd=str(suite_root))

                    # collect server output if any
                    if model in ("rt","unet"):
                        # read until exit
                        for ln in srv.stdout:
                            server_lines.append(ln.rstrip("\n"))
                        srv.wait(timeout=10)

                    # parse combined logs (server is where [RT]/[CFR] lives)
                    combined = ns3_lines + server_lines
                    rows, friis = parse_lines(combined)

                    # map friis by (tx,rx)
                    friis_map = {(tx,rx): fspl for (tx,rx,_,fspl) in friis}

                    # write any RT/UNet rows found; if friis model, you may only have FRIIS prints (server not used)
                    if rows:
                        for (kind, tx, rx, d_m, delay_ns, wb_db, tau_rms_ns) in rows:
                            fspl = friis_map.get((tx,rx), float("nan"))
                            w.writerow([scene_dir.name, rep, model, tx, rx, d_m, delay_ns, wb_db, tau_rms_ns, fspl,
                                        " | ".join(combined[-50:])])  # last 50 lines as context
                    else:
                        # still record a stub row
                        w.writerow([scene_dir.name, rep, model, -1, -1, float("nan"), -1, float("nan"), float("nan"), float("nan"),
                                    " | ".join(combined[-50:])])

                    f.flush()

    print("Wrote:", out_csv)

if __name__ == "__main__":
    main()
