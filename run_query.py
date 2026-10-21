import argparse
import os
import subprocess
from pathlib import Path


ROOT = Path(__file__).parent
ROVER = ROOT.parent
EARTHDIAL = ROVER.parent / "EarthDial"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("question", nargs="+")
    args = parser.parse_args()
    question = " ".join(args.question)

    environment = os.environ.copy()
    environment["EARTHDIAL_WORKER_URL"] = ""
    subprocess.run(
        [str(ROVER / ".venv" / "bin" / "python"), str(ROVER / "main.py"), question],
        cwd=ROVER, env=environment, check=True,
    )
    subprocess.run(
        [str(EARTHDIAL / ".venv" / "bin" / "python"), str(ROOT / "infer.py"),
         "--adapter", str(args.adapter.resolve())],
        cwd=ROVER, check=True,
    )


if __name__ == "__main__":
    main()
