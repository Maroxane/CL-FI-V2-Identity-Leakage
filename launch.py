"""Start / stop the two web servers and any running job. Called by start.bat and stop.bat.

    python launch.py start | stop | status
"""
import os
import socket
import subprocess
import sys
import time
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from sgmbf.common import load_config  # noqa: E402

cfg = load_config()
PIDS = Path(cfg["paths"]["runs"]) / "pids"
LOGS = Path(cfg["paths"]["logs"])
PORTS = {"monitor": cfg["web"]["monitor_port"], "demo": cfg["web"]["demo_port"]}
APPS = {"monitor": "web.monitor_app:app", "demo": "web.demo_app:app"}
NO_WINDOW = 0x08000000 if os.name == "nt" else 0
NEW_GROUP = 0x00000200 if os.name == "nt" else 0


def port_open(p):
    with socket.socket() as s:
        s.settimeout(0.3)
        return s.connect_ex(("127.0.0.1", p)) == 0


def kill_tree(pid):
    import psutil
    try:
        parent = psutil.Process(pid)
    except psutil.NoSuchProcess:
        return 0
    procs = parent.children(recursive=True) + [parent]
    for p in procs:
        try:
            p.kill()
        except psutil.NoSuchProcess:
            pass
    psutil.wait_procs(procs, timeout=5)
    return len(procs)


def start():
    try:
        import torch
        if torch.cuda.is_available():
            p = torch.cuda.get_device_properties(0)
            print(f"  GPU : {p.name} ({p.total_memory / 2**30:.1f} Go)")
        else:
            print("  ATTENTION : CUDA indisponible, le calcul se fera sur CPU (tres lent).")
    except Exception as ex:
        print("  ATTENTION : PyTorch ne se charge pas :", ex)
    for name, port in PORTS.items():
        if port_open(port):
            print(f"  {name} deja actif sur le port {port}")
            continue
        log = open(LOGS / f"{name}.log", "ab")
        subprocess.Popen([sys.executable, "-m", "uvicorn", APPS[name], "--host", "127.0.0.1", "--port", str(port)],
                         cwd=str(ROOT), stdout=log, stderr=subprocess.STDOUT, creationflags=NO_WINDOW | NEW_GROUP)
        print(f"  {name} : demarrage sur http://127.0.0.1:{port}")
    t0 = time.time()
    while time.time() - t0 < 90 and not all(port_open(p) for p in PORTS.values()):
        time.sleep(1)
    if all(port_open(p) for p in PORTS.values()):
        print("\n  Suivi de l'entrainement : http://127.0.0.1:%d" % PORTS["monitor"])
        print("  Tester le modele        : http://127.0.0.1:%d" % PORTS["demo"])
        webbrowser.open(f"http://127.0.0.1:{PORTS['monitor']}")
    else:
        print("  ERREUR : les serveurs ne repondent pas. Voir logs\\monitor.log et logs\\demo.log")
        sys.exit(1)


def stop():
    n = 0
    for name in ("pipeline", "job", "monitor", "demo"):
        f = PIDS / f"{name}.pid"
        if f.exists():
            try:
                n += kill_tree(int(f.read_text()))
            except Exception:
                pass
            f.unlink(missing_ok=True)
    # servers whose pid file is missing: free the ports
    import psutil
    for p in PORTS.values():
        for c in psutil.net_connections(kind="inet"):
            if c.laddr and c.laddr.port == p and c.status == psutil.CONN_LISTEN and c.pid:
                n += kill_tree(c.pid)
    from sgmbf.common import State
    s = State(cfg)
    if s.read().get("status") in ("running", "starting"):
        s.update(status="stopped", stage="Stopped (stop.bat)")
    print(f"  {n} processus arretes." if n else "  Rien a arreter.")


def status():
    for name, port in PORTS.items():
        print(f"  {name:8s} port {port} : {'actif' if port_open(port) else 'arrete'}")


if __name__ == "__main__":
    {"start": start, "stop": stop, "status": status}[sys.argv[1] if len(sys.argv) > 1 else "status"]()
