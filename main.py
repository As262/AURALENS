import argparse
import multiprocessing as mp
import signal
import sys

from eye_tracking import EyeTracker
from aura_overlay import run as run_aura


def _run_eye(stop_event):
    EyeTracker().run(stop_event)


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--eye", action="store_true")
    parser.add_argument("--overlay", action="store_true")
    args, _ = parser.parse_known_args()

    run_eye = args.eye or (not args.eye and not args.overlay)
    run_overlay = args.overlay or (not args.eye and not args.overlay)

    stop_event = mp.Event()
    processes = []

    if run_eye:
        p_eye = mp.Process(target=_run_eye, args=(stop_event,), name="eye-tracking")
        p_eye.start()
        processes.append(p_eye)

    if run_overlay:
        p_overlay = mp.Process(target=run_aura, args=(stop_event,), name="aura-overlay")
        p_overlay.start()
        processes.append(p_overlay)

    def _handle_exit(_sig, _frame):
        stop_event.set()

    signal.signal(signal.SIGINT, _handle_exit)
    signal.signal(signal.SIGTERM, _handle_exit)

    try:
        for p in processes:
            p.join()
    except KeyboardInterrupt:
        stop_event.set()
    finally:
        for p in processes:
            if p.is_alive():
                p.terminate()
        for p in processes:
            p.join(timeout=1.0)


if __name__ == "__main__":
    mp.freeze_support()
    mp.set_start_method("spawn", force=True)
    main()
