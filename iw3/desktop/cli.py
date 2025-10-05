import sys
import os
from .utils import (
    init_win32,
    create_parser, set_state_args,
    iw3_desktop_main,
)


def cli_main():
    init_win32()
    if sys.platform == "win32":
        # Update the command prompt title to avoid accidental matches by --window-name option
        os.system("title iw3.desktop")

    parser = create_parser()
    args = parser.parse_args()
    set_state_args(args)
    
    # When using --local-viewer, use platform-aware entry point
    if args.local_viewer:
        # Import the platform-aware runner
        try:
            from .local_viewer import run_local_viewer_cli
        except ImportError as e:
            print(f"Error: Local Viewer is not available: {e}", file=sys.stderr)
            raise RuntimeError("Local Viewer is not available")
        
        # Define the worker function
        def worker():
            try:
                # On Windows, wx.App is already created by run_local_viewer_cli on main thread
                # On Linux, we need to create it here in the worker
                init_wxapp = (sys.platform != "win32")
                return iw3_desktop_main(args, init_wxapp=init_wxapp)
            except Exception as e:
                print(f"Error in worker: {e}", file=sys.stderr)
                import traceback
                traceback.print_exc()
                raise
        
        try:
            # This function handles platform differences internally:
            # - Linux: Calls worker directly (worker creates wx.App, uses wx.Yield)
            # - Windows: Creates wx.App on main thread, runs worker in background
            run_local_viewer_cli(worker)
        except Exception as e:
            print(f"Error in run_local_viewer_cli: {e}", file=sys.stderr)
            import traceback
            traceback.print_exc()
            raise
    else:
        # Web streaming - original behavior
        iw3_desktop_main(args, init_wxapp=True)


if __name__ == "__main__":
    cli_main()