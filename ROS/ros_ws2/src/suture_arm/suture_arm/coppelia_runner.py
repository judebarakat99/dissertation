#!/usr/bin/env python3
"""
Launch CoppeliaSim with a scene (GUI or headless).

Usage (standalone):
  python3 -m suture_arm.coppelia_runner --coppelia /path/to/coppeliaSim.sh --scene /path/to/scene.ttt
  # or via console script once installed:
  coppelia_run --coppelia /path/to/coppeliaSim.sh --scene /path/to/scene.ttt --headless

In ROS launch, we’ll call this as a normal process.
"""
import os
import sys
import time
import shlex
import signal
import argparse
import subprocess
from pathlib import Path

try:
    from ament_index_python.packages import get_package_share_directory
except Exception:
    get_package_share_directory = None


def _default_scene():
    if get_package_share_directory is None:
        return None
    try:
        share = get_package_share_directory('suture_arm')
        return str(Path(share) / 'scenes' / 'ur3_suture.ttt')
    except Exception:
        return None


def _guess_coppelia():
    # 1) env var
    root = os.environ.get('COPPELIASIM_ROOT', '')
    cand = Path(root) / 'coppeliaSim.sh'
    if cand.is_file():
        return str(cand)

    # 2) common download locations
    guesses = [
        '~/CoppeliaSim/coppeliaSim.sh',
        '~/Downloads/CoppeliaSim/coppeliaSim.sh',
        '~/Downloads/CoppeliaSim_Edu/coppeliaSim.sh',
    ]
    for g in guesses:
        p = Path(os.path.expanduser(g))
        if p.is_file():
            return str(p)
    return None


def main():
    parser = argparse.ArgumentParser(description='Start CoppeliaSim with a scene.')
    parser.add_argument('--coppelia', '-c', default=_guess_coppelia(),
                        help='Path to coppeliaSim.sh (or .exe). Can also set COPPELIASIM_ROOT.')
    parser.add_argument('--scene', '-s', default=_default_scene(),
                        help='Path to .ttt scene file (defaults to package scene if installed).')
    parser.add_argument('--headless', action='store_true', help='Run with -h (no GUI).')
    parser.add_argument('--extra', default='', help='Extra args passed to CoppeliaSim (quoted string).')
    parser.add_argument('--wait', action='store_true', default=True,
                        help='(default) Wait and forward SIGINT/SIGTERM to CoppeliaSim.')
    parser.add_argument('--nowait', dest='wait', action='store_false', help='Detach after starting.')
    args = parser.parse_args()

    if not args.coppelia:
        print('[coppelia_runner] ERROR: could not locate coppeliaSim.sh. Use --coppelia or set COPPELIASIM_ROOT.', file=sys.stderr)
        return 2

    if not args.scene:
        print('[coppelia_runner] WARNING: no scene given; starting the simulator without a scene.', file=sys.stderr)

    coppelia = os.path.abspath(os.path.expanduser(args.coppelia))
    if not os.path.isfile(coppelia):
        print(f'[coppelia_runner] ERROR: not a file: {coppelia}', file=sys.stderr)
        return 2

    # Working directory = CoppeliaSim root (important for add-ons)
    workdir = str(Path(coppelia).resolve().parent)

    cmd = [coppelia]
    if args.headless:
        cmd.append('-h')
    if args.scene:
        cmd.append(os.path.abspath(os.path.expanduser(args.scene)))
    if args.extra:
        cmd += shlex.split(args.extra)

    print('[coppelia_runner] launching:', ' '.join(shlex.quote(x) for x in cmd))
    print('[coppelia_runner] cwd:', workdir)

    # Launch
    proc = subprocess.Popen(cmd, cwd=workdir)

    if not args.wait:
        print('[coppelia_runner] launched (detached). pid =', proc.pid)
        return 0

    # Forward signals, wait
    def _forward(signum, _frame):
        try:
            proc.send_signal(signum)
        except Exception:
            pass
    signal.signal(signal.SIGINT, _forward)
    signal.signal(signal.SIGTERM, _forward)

    try:
        rc = proc.wait()
        print('[coppelia_runner] exited with code', rc)
        return rc
    except KeyboardInterrupt:
        try:
            proc.send_signal(signal.SIGINT)
            return proc.wait(timeout=5.0)
        except Exception:
            proc.kill()
            return 130


if __name__ == '__main__':
    sys.exit(main())
