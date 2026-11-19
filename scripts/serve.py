"""Unified launcher and process supervisor for PkmLens.

Starts Core API (8100), AI Service (8101), and Vite Web Frontend (5173).
Safely manages only PkmLens processes; never touches CardLens.
"""
import argparse
import os
import pathlib
import secrets
import signal
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
import webbrowser

try:
    import psutil
except ImportError:
    psutil = None

ROOT = pathlib.Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
PORTS = {"frontend": 5173, "core": 8100, "ai": 8101}


def get_ai_internal_key() -> str:
    key_file = DATA_DIR / ".ai_internal_key"
    if key_file.is_file():
        key = key_file.read_text(encoding="utf-8").strip()
        if len(key) >= 32:
            return key
    key = secrets.token_hex(32)
    key_file.parent.mkdir(parents=True, exist_ok=True)
    key_file.write_text(key, encoding="utf-8")
    return key


def setup_env():
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{ROOT / 'services/core-api'};{ROOT / 'services/ai'};{ROOT}"
    env["DATA_DIR"] = str(DATA_DIR)
    env["DATABASE_URL"] = f"sqlite:///{(DATA_DIR / 'pkmlens.db').as_posix()}"
    env["WEB_ORIGIN"] = "http://127.0.0.1:5173,http://localhost:5173"
    env["AI_URL"] = "http://127.0.0.1:8101"
    env["PKMLENS_WARMUP_ON_START"] = "1"
    env["PKMLENS_LIVE_PRICE_FETCH"] = "1"
    env["STORAGE_MODE"] = "local"
    env["YOLO_CONFIG_DIR"] = str(DATA_DIR / "yolo-config")
    env["HF_HOME"] = str(DATA_DIR / "huggingface")
    env["PKMLENS_GALLERY_BACKEND"] = "auto"
    env["AI_INTERNAL_KEY"] = get_ai_internal_key()
    return env


def find_pkmlens_processes():
    """Find listening processes on ports 5173, 8100, 8101 that belong ONLY to PkmLens."""
    if not psutil:
        return []
    pkm_dir = ROOT.resolve()
    found = []
    seen_pids = set()
    for port in PORTS.values():
        for conn in psutil.net_connections(kind="tcp"):
            if conn.status == "LISTEN" and conn.laddr.port == port:
                pid = conn.pid
                if not pid or pid in seen_pids:
                    continue
                try:
                    proc = psutil.Process(pid)
                    cwd = pathlib.Path(proc.cwd()).resolve()
                    is_pkm = cwd == pkm_dir or pkm_dir in cwd.parents
                    found.append({"port": port, "pid": pid, "proc": proc, "is_pkm": is_pkm, "name": proc.name()})
                    seen_pids.add(pid)
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
    return found


def stop_pkmlens_services():
    """Stop active PkmLens supervisor and service processes safely."""
    if psutil:
        pkm_dir = ROOT.resolve()
        me = os.getpid()
        for proc in psutil.process_iter(["pid", "name", "cmdline"]):
            if proc.info["pid"] == me:
                continue
            try:
                cmd = " ".join(proc.info.get("cmdline") or [])
                if "serve.py" in cmd:
                    cwd = pathlib.Path(proc.cwd()).resolve()
                    if cwd == pkm_dir or pkm_dir in cwd.parents:
                        proc.terminate()
                        psutil.wait_procs([proc], timeout=2)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
    procs = find_pkmlens_processes()
    if not procs:
        print("[INFO] Khong co tien trinh PkmLens nao dang chay tren cac cong 5173, 8100, 8101.")
        return True

    stopped_count = 0
    for item in procs:
        port, pid, proc, is_pkm, name = item["port"], item["pid"], item["proc"], item["is_pkm"], item["name"]
        if is_pkm:
            print(f"[STOP] Dang tat tien trinh PkmLens (Port {port}, PID {pid}, {name})...")
            try:
                # Terminate children first
                for child in proc.children(recursive=True):
                    try:
                        child.terminate()
                    except Exception:
                        pass
                proc.terminate()
                gone, alive = psutil.wait_procs([proc], timeout=3)
                for p in alive:
                    p.kill()
                stopped_count += 1
            except Exception as exc:
                print(f"[WARN] Khong the tat PID {pid}: {exc}")
        else:
            print(f"[ALERT] Cong {port} (PID {pid}) khong thuoc PkmLens! Bo qua khong dung den.")

    time.sleep(1)
    print(f"[DONE] Da dung {stopped_count} tien trinh PkmLens.")
    return True


def check_ports_available(auto_restart=True):
    procs = find_pkmlens_processes()
    if not procs:
        return True

    pkm_procs = [p for p in procs if p["is_pkm"]]
    foreign_procs = [p for p in procs if not p["is_pkm"]]

    if foreign_procs:
        print("\n[LOI] Phat hien cong da bi chiem boi ung dung khac khong phai PkmLens:")
        for p in foreign_procs:
            print(f"  - Cong {p['port']} (PID {p['pid']}: {p['name']})")
        print("Vui long dong cac ung dung nay truoc khi khoi dong PkmLens.")
        return False

    if pkm_procs:
        if auto_restart:
            print("\n[INFO] Phat hien cac tien trinh PkmLens cu dang chay tren cong 5173/8100/8101.")
            print("[INFO] Dang tu dong giai phong va khoi dong lai he thong moi nhat...")
            stop_pkmlens_services()
            return True
        else:
            print("\n[WARN] Cac cong 5173/8100/8101 dang duoc su dung boi PkmLens. Dung tham so --restart de khoi dong lai.")
            return False

    return True


def poll_health(url: str, timeout_sec: float = 20.0) -> bool:
    start = time.time()
    while time.time() - start < timeout_sec:
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "PkmLens-Supervisor"})
            with urllib.request.urlopen(req, timeout=1.5) as resp:
                if resp.status == 200:
                    return True
        except Exception:
            pass
        time.sleep(0.5)
    return False


def stream_output(process, prefix):
    try:
        for line in iter(process.stdout.readline, b""):
            text = line.decode("utf-8", errors="replace").rstrip()
            if text:
                print(f"[{prefix}] {text}", flush=True)
    except Exception:
        pass


def main():
    parser = argparse.ArgumentParser(description="PkmLens Unified System Launcher")
    parser.add_argument("--stop", action="store_true", help="Dung toan bo cac dich vu PkmLens dang chay")
    parser.add_argument("--status", action="store_true", help="Kiem tra trang thai cac cong va dich vu PkmLens")
    parser.add_argument("--no-browser", action="store_true", help="Khong tu dong mo trinh duyet")
    parser.add_argument("--no-restart", action="store_true", help="Khong tu dong khoi dong lai neu cong da mo")
    args = parser.parse_args()

    os.chdir(str(ROOT))

    if args.status:
        procs = find_pkmlens_processes()
        print("\n=== TRANG THAI DICH VU PKMLENS ===")
        if not procs:
            print("Cac cong 5173, 8100, 8101 deu trong. He thong dang OFF.")
        else:
            for p in procs:
                status = "PkmLens Process" if p["is_pkm"] else "EXTERNAL APP (Khong thuoc PkmLens)"
                print(f" - Cong {p['port']}: PID {p['pid']} ({p['name']}) -> {status}")
        return 0

    if args.stop:
        stop_pkmlens_services()
        return 0

    print("=" * 68)
    print("        PKMLENS - HE THONG MARKETPLACE & YOLO AI SCANNER")
    print("=" * 68)
    print(f"Thu muc lam viec: {ROOT}")

    # 1. Kiem tra cong
    if not check_ports_available(auto_restart=not args.no_restart):
        sys.exit(1)

    # 2. Chuan bi moi truong
    env = setup_env()
    python_exe = sys.executable

    print("\n[1/4] Chuan bi du lieu (scripts/prepare.py)...")
    res = subprocess.run([python_exe, "scripts/prepare.py"], env=env)
    if res.returncode != 0:
        print("[LOI] Chuan bi du lieu that bai!")
        sys.exit(1)

    # 3. Khoi dong cac tien trinh
    print("\n[2/4] Khoi dong cac dich vu nen...")
    children = []

    def cleanup():
        print("\n[STOP] Dang tat toan bo cac tien trinh PkmLens...")
        for name, p in children:
            if p.poll() is None:
                try:
                    if psutil:
                        parent = psutil.Process(p.pid)
                        for child in parent.children(recursive=True):
                            child.terminate()
                        parent.terminate()
                    else:
                        p.terminate()
                except Exception:
                    pass
        time.sleep(1)
        for name, p in children:
            if p.poll() is None:
                try:
                    p.kill()
                except Exception:
                    pass
        print("[STOP] Hoan tat!")

    # Core API (8100)
    core_proc = subprocess.Popen(
        [python_exe, "-m", "uvicorn", "cardlens_core.main:app", "--host", "127.0.0.1", "--port", "8100"],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    children.append(("CORE", core_proc))
    threading.Thread(target=stream_output, args=(core_proc, "CORE"), daemon=True).start()

    # AI Service (8101)
    ai_proc = subprocess.Popen(
        [python_exe, "-m", "uvicorn", "cardlens_ai.main:app", "--host", "127.0.0.1", "--port", "8101"],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    children.append(("AI", ai_proc))
    threading.Thread(target=stream_output, args=(ai_proc, "AI"), daemon=True).start()

    # Web Frontend (5173)
    frontend_proc = subprocess.Popen(
        ["npm.cmd", "run", "dev"],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    children.append(("WEB", frontend_proc))
    threading.Thread(target=stream_output, args=(frontend_proc, "WEB"), daemon=True).start()

    # 4. Kiem tra Health
    print("\n[3/4] Dang kiem tra ket noi den cac dich vu...")
    core_ok = poll_health("http://127.0.0.1:8100/healthz", timeout_sec=20)
    ai_ok = poll_health("http://127.0.0.1:8101/healthz", timeout_sec=35)
    web_ok = poll_health("http://127.0.0.1:5173", timeout_sec=20)
    if ai_ok:
        sample = ROOT / "data" / "evaluation" / "camera-scene.jpg"
        if sample.is_file():
            try:
                boundary = "----PkmLensWarmupBoundary"
                raw_img = sample.read_bytes()
                body = (
                    f"--{boundary}\r\n"
                    f'Content-Disposition: form-data; name="image"; filename="warmup.jpg"\r\n'
                    f"Content-Type: image/jpeg\r\n\r\n"
                ).encode("utf-8") + raw_img + f"\r\n--{boundary}--\r\n".encode("utf-8")
                req = urllib.request.Request(
                    "http://127.0.0.1:8101/v1/identify",
                    data=body,
                    headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
                    method="POST",
                )
                with urllib.request.urlopen(req, timeout=15):
                    pass
            except Exception:
                pass

    print("\n" + "=" * 68, flush=True)
    if core_ok and ai_ok and web_ok:
        print("          HE THONG PKMLENS DA KHOI DONG THANH CONG!", flush=True)
        print("=" * 68)
        print("   - Website Frontend : http://127.0.0.1:5173")
        print("   - Backend Core API : http://127.0.0.1:8100  (Docs: /docs)")
        print("   - AI Service (YOLO): http://127.0.0.1:8101  (Ready: /readyz)")
        print("=" * 68)
        print(" [Meo] Nhan Ctrl+C tai cua so nay de dung toan bo he thong.")
        print("=" * 68 + "\n")

        if not args.no_browser:
            print("[4/4] Dang mo trinh duyet tai http://127.0.0.1:5173...")
            webbrowser.open("http://127.0.0.1:5173")
    else:
        print("[CANH BAO] Mot so dich vu chua san sang sau thoi gian cho:")
        print(f"   - Core API: {'OK' if core_ok else 'CHUA PHAN HOI'}")
        print(f"   - AI Service: {'OK' if ai_ok else 'CHUA PHAN HOI'}")
        print(f"   - Web Frontend: {'OK' if web_ok else 'CHUA PHAN HOI'}")
        print("Vui long theo doi log ben duoi...")

    def handle_sig(sig, frame):
        cleanup()
        sys.exit(0)

    signal.signal(signal.SIGINT, handle_sig)
    signal.signal(signal.SIGTERM, handle_sig)

    try:
        while True:
            # Check if any child died
            for name, proc in children:
                ret = proc.poll()
                if ret is not None:
                    print(f"\n[CANH BAO] Dich vu {name} da dung voi ma loi {ret}!")
                    cleanup()
                    sys.exit(ret)
            time.sleep(1)
    except KeyboardInterrupt:
        cleanup()
        sys.exit(0)


if __name__ == "__main__":
    main()
