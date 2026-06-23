"""Entry point. Launches the system-tray app by default.

Debug flags (run a single component standalone, no tray):
    python main.py --eye        # eye tracker only
    python main.py --overlay    # overlay only
    python main.py --calibrate  # run calibration only
"""
import argparse
import multiprocessing as mp


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--eye", action="store_true")
    parser.add_argument("--overlay", action="store_true")
    parser.add_argument("--calibrate", action="store_true")
    args, _ = parser.parse_known_args()

    if args.eye or args.overlay or args.calibrate:
        from config import SharedState, load, push_to_shared
        shared = SharedState()
        push_to_shared(load(), shared)
        stop_event = mp.Event()
        try:
            if args.calibrate:
                from calibration import run_calibration
                run_calibration(stop_event, shared)
            elif args.eye:
                from eye_tracking import EyeTracker
                EyeTracker(shared, load().get("calibration")).run(stop_event)
            elif args.overlay:
                from aura_overlay import run as run_overlay
                run_overlay(stop_event, shared)
        except KeyboardInterrupt:
            stop_event.set()
        return

    from tray_app import main as tray_main
    tray_main()


if __name__ == "__main__":
    mp.freeze_support()
    mp.set_start_method("spawn", force=True)
    main()
