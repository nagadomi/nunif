from . import video as VU
from .transnetv2 import TransNetV2
import torch


def _fps_config(max_fps):
    def callback(stream):
        if max_fps is None:
            # keep original frame rate, use raw pts
            return VU.VideoOutputConfig(fps=None)
        else:
            fps = VU.get_fps(stream)
            if float(fps) > max_fps:
                fps = max_fps
            return VU.VideoOutputConfig(fps=fps)

    return callback


def detect_boundary(
        video_file,
        device="cuda",
        window_size=100, padding_size=25, threshold=0.5,
        max_fps=None,
        start_time=None,
        end_time=None,
        stop_event=None,
        suspend_event=None,
        tqdm_fn=None,
        tqdm_title=None,
):
    assert (window_size % padding_size == 0 and
            window_size // padding_size >= 3)  # pad1 + frames + pad2

    model = TransNetV2().load().eval().to(device)
    frames = []
    results = []
    first_frame = [True]
    frame_count = [0]

    def push_predict(x, pts):
        # NOTE: No need autocast. Nothing improves.
        with torch.inference_mode():
            single_frame_pred, all_frame_pred = model(x)
            single_frame_pred = torch.sigmoid(single_frame_pred).flatten()
        results.append((
            single_frame_pred[padding_size:-padding_size].cpu(),
            pts[padding_size:-padding_size].cpu()
        ))
        for _ in range((window_size - padding_size * 2) // padding_size):
            frames.pop(0)

    def batch_callback(x, pts):
        frame_count[0] += x.shape[0]
        pts = torch.tensor(pts, dtype=torch.long)
        if x.shape[0] < padding_size:
            n = padding_size - x.shape[0]
            pad_x = torch.cat((x[-1:],) * n, dim=0)
            pad_pts = torch.cat((pts[-1:],) * n, dim=0)
            x = torch.cat((x, pad_x), dim=0)
            pts = torch.cat((pts, pad_pts), dim=0)

        if first_frame[0]:
            first_frame[0] = False
            pad_x = torch.cat((x[0:1],) * padding_size, dim=0)
            pad_pts = torch.cat((pts[0:1],) * padding_size, dim=0)
            frames.append((pad_x, pad_pts))
            frames.append((x, pts))
        else:
            frames.append((x, pts))

        if len(frames) == window_size // padding_size:
            push_predict(torch.cat([x_ for x_, _ in frames], dim=0),
                         torch.cat([pts_ for _, pts_ in frames], dim=0))

    callback_pool = VU.FrameCallbackPool(
        batch_callback,
        require_pts=True,
        batch_size=padding_size,
        device=device,
        max_workers=0,  # must be sequential
    )
    VU.hook_frame(
        video_file, callback_pool,
        config_callback=_fps_config(max_fps),
        title=tqdm_title or "Shot Boundary Detection",
        vf="scale=48:27:flags=bilinear",  # input size for TransNetV2
        start_time=start_time, end_time=end_time,
        stop_event=stop_event,
        suspend_event=suspend_event,
        tqdm_fn=tqdm_fn
    )
    if stop_event is not None and stop_event.is_set():
        return set()

    last_x = frames[-1][0][-1:]
    last_pts = frames[-1][1][-1:]
    pad_x = torch.cat((last_x,) * padding_size, dim=0)
    pad_pts = torch.cat((last_pts,) * padding_size, dim=0)
    while (not results or results[-1][1][-1] != last_pts[0]):
        frames.append((pad_x, pad_pts))
        if len(frames) == window_size // padding_size:
            push_predict(torch.cat([x_ for x_, _ in frames], dim=0),
                         torch.cat([pts_ for _, pts_ in frames], dim=0))

    frame_preds = torch.cat([pred for pred, pts in results], dim=0)[:frame_count[0]]
    frame_pts = torch.cat([pts for pred, pts in results], dim=0)[:frame_count[0]]
    segment_pts = set(frame_pts[frame_preds > threshold].tolist())

    # NOTE: pts is the end point of the segment. It is not the starting point.
    return segment_pts


def _hevc_deadlock_test():
    import argparse

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--input", "-i", type=str, required=True,
                        help="hevc input video file")
    args = parser.parse_args()

    for i in range(1, 60):
        for j in [2, 3, 4, 5, 6, 8, 16]:
            pts = detect_boundary(
                args.input,
                device="cuda",
                window_size=100, padding_size=25, threshold=0.5,
                max_fps=30,
                start_time=str(i),
                end_time=str(i + j),
                stop_event=None,
                suspend_event=None,
                tqdm_fn=None,
                tqdm_title=None,
            )
            print(i, j, pts)


if __name__ == "__main__":
    # _hevc_deadlock_test()
    pass
