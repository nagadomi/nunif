import argparse

from .display_mode import guard_main


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent-pid", required=True, type=int)
    parser.add_argument("--parent-create-time", required=True, type=float)
    parser.add_argument("--monitor-name", required=True)
    parser.add_argument("--monitor-center-x", required=True, type=int)
    parser.add_argument("--monitor-center-y", required=True, type=int)
    parser.add_argument("--width", required=True, type=int)
    parser.add_argument("--height", required=True, type=int)
    parser.add_argument("--frequency", required=True, type=float)
    args = parser.parse_args()
    state = {
        "parent_pid": args.parent_pid,
        "parent_create_time": None if args.parent_create_time < 0 else args.parent_create_time,
        "monitor_name": args.monitor_name,
        "monitor_center": {
            "x": args.monitor_center_x,
            "y": args.monitor_center_y,
        },
        "original_mode": {
            "width": args.width,
            "height": args.height,
            "frequency": args.frequency,
        },
    }
    return guard_main(state)


if __name__ == "__main__":
    raise SystemExit(main())
