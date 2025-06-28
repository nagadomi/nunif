from nunif.models import load_model
from nunif.utils.ui import TorchHubDir
from os import path


HUB_MODEL_DIR = path.join(path.dirname(__file__), "pretrained_models", "hub")


def pth_url(filename):
    return "https://github.com/nagadomi/nunif/releases/download/0.0.0/" + filename


ROW_FLOW_V2_URL = pth_url("iw3_row_flow_v2_20240130.pth")
ROW_FLOW_V3_URL = pth_url("iw3_row_flow_v3_20250627.pth")
ROW_FLOW_V3_SYM_URL = pth_url("iw3_row_flow_v3_sym_20250628.pth")

MLBW_L2_D1_URL = pth_url("iw3_mlbw_l2_d1_20250627.pth")
MLBW_L2_D2_URL = pth_url("iw3_mlbw_l2_d2_20250627.pth")
MLBW_L2_D3_URL = pth_url("iw3_mlbw_l2_d3_20250627.pth")

MLBW_L4_D1_URL = pth_url("iw3_mlbw_l4_d1_20250627.pth")
MLBW_L4_D2_URL = pth_url("iw3_mlbw_l4_d2_20250627.pth")
MLBW_L4_D3_URL = pth_url("iw3_mlbw_l4_d3_20250627.pth")

# small model
MLBW_L2S_D1_URL = pth_url("iw3_mlbw_l2s_d1_20250627.pth")
MLBW_L4S_D1_URL = pth_url("iw3_mlbw_l4s_d1_20250627.pth")

# weak convergence_model (convergence 0.3-0.7 only)
MLBW_L2_D2_WEAK_URL = pth_url("iw3_mlbw_l2_d2_weak_20250627.pth")
MLBW_L2_D3_WEAK_URL = pth_url("iw3_mlbw_l2_d3_weak_20250627.pth")
MLBW_L4_D2_WEAK_URL = pth_url("iw3_mlbw_l4_d2_weak_20250627.pth")
MLBW_L4_D3_WEAK_URL = pth_url("iw3_mlbw_l4_d3_weak_20250627.pth")


def load_mlbw_model(
        method, divergence, device_id,
        use_weak_convergence_model=False,
):
    if method in {"mlbw_l2", "mlbw_l2s"}:
        if divergence <= 4:
            if method == "mlbw_l2s":
                url = MLBW_L2S_D1_URL
            else:
                url = MLBW_L2_D1_URL
        elif divergence <= 7:
            if use_weak_convergence_model:
                url = MLBW_L2_D2_WEAK_URL
            else:
                url = MLBW_L2_D2_URL
        else:
            if use_weak_convergence_model:
                url = MLBW_L2_D3_WEAK_URL
            else:
                url = MLBW_L2_D3_URL
    elif method in {"mlbw_l4", "mlbw_l4s"}:
        if divergence <= 4:
            if method == "mlbw_l4s":
                url = MLBW_L4S_D1_URL
            else:
                url = MLBW_L4_D1_URL
        elif divergence <= 7:
            if use_weak_convergence_model:
                url = MLBW_L4_D2_WEAK_URL
            else:
                url = MLBW_L4_D2_URL
        else:
            if use_weak_convergence_model:
                url = MLBW_L4_D3_WEAK_URL
            else:
                url = MLBW_L4_D3_URL
    else:
        raise ValueError(method)

    model = load_model(url, weights_only=True, device_ids=[device_id])[0].eval()
    model.delta_output = True

    return model


def load_row_flow_model(method, device_id):
    if method in {"row_flow_v3", "row_flow"}:
        model = load_model(ROW_FLOW_V3_URL, weights_only=True, device_ids=[device_id])[0].eval()
        model.symmetric = False
        model.delta_output = True
    elif method in {"row_flow_v3_sym", "row_flow_sym"}:
        model = load_model(ROW_FLOW_V3_SYM_URL, weights_only=True, device_ids=[device_id])[0].eval()
        model.symmetric = True
        model.delta_output = True
    elif method == "row_flow_v2":
        model = load_model(ROW_FLOW_V2_URL, weights_only=True, device_ids=[device_id])[0].eval()
        model.delta_output = True
    else:
        raise ValueError(method)

    return model


def create_stereo_model(
        method, divergence, device_id,
        use_weak_convergence_model=False,
):
    with TorchHubDir(HUB_MODEL_DIR):
        if method.startswith("row_flow"):
            return load_row_flow_model(method, device_id=device_id)
        elif method.startswith("mlbw_"):
            return load_mlbw_model(
                method,
                divergence=divergence,
                device_id=device_id,
                use_weak_convergence_model=use_weak_convergence_model,
            )
        elif method in {"forward", "forward_fill", "backward"}:
            return None
        else:
            raise ValueError(method)
