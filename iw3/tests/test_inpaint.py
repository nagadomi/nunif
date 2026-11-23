import torch

from iw3.forward_inpaint import ForwardInpaintImage, ForwardInpaintVideo
from iw3.mlbw_inpaint import MLBWInpaintImage
from iw3.monobw_inpaint import MonoBWInpaintImage


def test_video_inpaint_alpha_memory_corruption() -> None:
    vid_model = ForwardInpaintVideo(device_id=-1, pre_padding=1, post_padding=1)
    vid_model.model_seq = 4

    outputs = []
    for i in range(5):
        x = torch.full((1, 3, 16, 16), float(i))
        depth = torch.ones((1, 1, 16, 16)) * 0.5
        alpha = torch.full((1, 1, 16, 16), float(i + 1) / 10.0)
        ret = vid_model.infer(x, depth, divergence=0.0, convergence=0.5, synthetic_view="both", alpha=alpha)
        if ret[0] is not None:
            assert len(ret) == 4
            assert ret[0] is not None and ret[2] is not None
            outputs.append((ret[0], ret[2]))  # left_eye, left_alpha

    ret = vid_model.flush()
    if ret[0] is not None:
        assert len(ret) == 4
        assert ret[0] is not None and ret[2] is not None
        outputs.append((ret[0], ret[2]))

    # Flatten all emitted frames
    all_left_eye = torch.cat([o[0] for o in outputs], dim=0)
    all_left_alpha = torch.cat([o[1] for o in outputs], dim=0)
    assert all_left_eye.shape[0] == 5
    assert all_left_alpha.shape[0] == 5

    # Check each frame's alpha value is not corrupted by subsequent frame additions
    for i in range(5):
        expected_alpha = float(i + 1) / 10.0
        assert torch.allclose(all_left_alpha[i], torch.tensor(expected_alpha), atol=1e-4)


def test_inpaint_warp_alpha_no_foreground_holes() -> None:
    h, w = 256, 256
    alpha = torch.zeros(1, 1, h, w)
    alpha[:, :, 40:216, 40:216] = 1.0  # foreground object (clock face)

    depth = torch.ones(1, 1, h, w) * 0.1
    depth[:, :, 40:216, 40:216] = 0.5  # object base depth
    depth[:, :, 60:196, 120:136] = 0.9  # depth step inside object (clock hand)

    models = [
        ForwardInpaintImage(device_id=-1),
        MLBWInpaintImage(device_id=-1),
        MonoBWInpaintImage(device_id=-1),
    ]

    for model in models:
        left_alpha, right_alpha = model.warp_alpha(alpha, depth, divergence=5.0, convergence=0.5, synthetic_view="both")
        # The region [80:180, 80:160] is completely inside the foreground face (Alpha == 1)
        # Even across depth steps, holes (Alpha == 0) must not appear inside the foreground
        assert (left_alpha[:, :, 80:180, 80:160] == 0).sum().item() == 0
        assert (right_alpha[:, :, 80:180, 80:160] == 0).sum().item() == 0
