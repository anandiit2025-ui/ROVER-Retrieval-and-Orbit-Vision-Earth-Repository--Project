import os
import socket
import subprocess
import sys
import time
from pathlib import Path
from urllib.parse import urlparse

from dotenv import load_dotenv
from utils import ROOT

load_dotenv(ROOT / ".env")


class ServiceError(Exception):
    pass


def configured_path(name, base=None):
    value = os.getenv(name, "").strip()
    if not value:
        raise ServiceError(f"{name} is required")
    path = Path(value).expanduser()
    if not path.is_absolute() and base:
        path = base / path
    path = path.absolute()
    if not path.exists():
        raise ServiceError(f"{name} does not exist: {path}")
    return path


def address(name, default):
    url = os.getenv(name, default).strip().rstrip("/")
    parsed = urlparse(url)
    if parsed.scheme != "http" or not parsed.hostname or not parsed.port:
        raise ServiceError(f"{name} must be an HTTP URL with an explicit port")
    return url, parsed.hostname, parsed.port


def port_open(host, port):
    try:
        with socket.create_connection((host, port), timeout=1):
            return True
    except OSError:
        return False


def wait_for_port(process, host, port, timeout, label):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise ServiceError(f"{label} exited with status {process.returncode}")
        if port_open(host, port):
            return
        time.sleep(1)
    raise ServiceError(f"{label} was not ready within {timeout:g} seconds")


def stop(process, label):
    if not process or process.poll() is not None:
        return
    print(f"Stopping {label}...", flush=True)
    process.terminate()
    try:
        process.wait(timeout=15)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5)


def supported_option(source, current, legacy):
    if current in source:
        return current
    if legacy in source:
        return legacy
    raise ServiceError(f"Worker supports neither {current} nor {legacy}")


def main():
    repo = configured_path("EARTHDIAL_REPO")
    python = configured_path("EARTHDIAL_PYTHON")
    model = configured_path("EARTHDIAL_MODEL_PATH", repo)
    controller_script = repo / "demo" / "controller.py"
    worker_script = repo / "demo" / "model_worker.py"
    if not controller_script.is_file() or not worker_script.is_file():
        raise ServiceError(f"EarthDial demo scripts were not found under {repo}")

    controller_url, controller_host, controller_port = address(
        "EARTHDIAL_CONTROLLER_URL", "http://127.0.0.1:40000"
    )
    worker_url, worker_host, worker_port = address(
        "EARTHDIAL_WORKER_URL", "http://127.0.0.1:40001"
    )
    if port_open(worker_host, worker_port):
        print(f"EarthDial worker is already listening at {worker_url}")
        return

    logs = ROOT / "build" / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    controller = worker = controller_log = worker_log = None
    owns_controller = False
    try:
        if port_open(controller_host, controller_port):
            print(f"Reusing controller at {controller_url}", flush=True)
        else:
            controller_log = (logs / "earthdial_controller.log").open("w")
            controller = subprocess.Popen(
                [str(python), str(controller_script), "--host", controller_host,
                 "--port", str(controller_port)], cwd=repo,
                stdout=controller_log, stderr=subprocess.STDOUT,
            )
            owns_controller = True
            wait_for_port(
                controller, controller_host, controller_port, 30,
                "EarthDial controller",
            )

        source = worker_script.read_text(encoding="utf-8")
        controller_option = supported_option(
            source, "--controller-address", "--controller"
        )
        worker_option = supported_option(
            source, "--worker-address", "--worker"
        )
        command = [
            str(python), str(worker_script), "--host", worker_host,
            "--port", str(worker_port), controller_option, controller_url,
            worker_option, worker_url, "--model-path", str(model),
        ]
        device = os.getenv("EARTHDIAL_DEVICE", "cuda").strip()
        if "--device" in source and device:
            command.extend(["--device", device])
        if os.getenv("EARTHDIAL_LOAD_8BIT", "false").lower() in {
            "1", "true", "yes",
        }:
            command.append("--load-8bit")

        environment = os.environ.copy()
        environment["PYTHONPATH"] = os.pathsep.join(
            part for part in (str(repo / "src"), environment.get("PYTHONPATH", "")) if part
        )
        gpu = os.getenv("EARTHDIAL_CUDA_VISIBLE_DEVICES", "").strip()
        if gpu:
            environment["CUDA_VISIBLE_DEVICES"] = gpu
        worker_log_path = logs / "earthdial_worker.log"
        worker_log = worker_log_path.open("w")
        print(f"Loading EarthDial; log: {worker_log_path}", flush=True)
        worker = subprocess.Popen(
            command, cwd=repo, env=environment,
            stdout=worker_log, stderr=subprocess.STDOUT,
        )
        timeout = float(os.getenv("EARTHDIAL_STARTUP_TIMEOUT", "900"))
        wait_for_port(
            worker, worker_host, worker_port, timeout,
            "EarthDial model worker",
        )
        print(f"EarthDial worker is ready at {worker_url}", flush=True)
        print("Keep this terminal open. Press Ctrl+C to stop.", flush=True)
        return_code = worker.wait()
        if return_code:
            raise ServiceError(f"EarthDial worker exited with status {return_code}")
    except KeyboardInterrupt:
        print("\nShutdown requested.", flush=True)
    finally:
        stop(worker, "EarthDial worker")
        if owns_controller:
            stop(controller, "EarthDial controller")
        if worker_log:
            worker_log.close()
        if controller_log:
            controller_log.close()


if __name__ == "__main__":
    try:
        main()
    except (ServiceError, ValueError) as error:
        print(f"EarthDial service error: {error}", file=sys.stderr)
        raise SystemExit(1)
