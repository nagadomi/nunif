import argparse
from tqdm import tqdm
import torch
import torch.nn as nn
from nunif.models import load_model, save_model, get_model_config


def calibrate_output():
    class RGBCalibration(nn.Module):
        def __init__(self):
            super().__init__()
            self.rgb = nn.Parameter(torch.zeros((1, 3, 1, 1), dtype=torch.float32))

        def forward(self, x):
            return torch.clamp(x.detach() + self.rgb, 0., 1.)

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--input", "-i", type=str, required=True, help="input 4x model")
    parser.add_argument("--gpu", type=int, default=0, help="gpu id. -1 for cpu")
    args = parser.parse_args()
    device = f"cuda:{args.gpu}" if args.gpu >= 0 else "cpu"
    model, meta = load_model(args.input)
    assert meta["name"] == "waifu2x.swin_unet_4x"
    input_size = 64
    batch_size = 16
    epoch = 100
    steps = 2048
    amp = False if device == "cpu" else True
    amp_device_type = "cuda" if "cuda" in device else "cpu"
    amp_dtype = torch.bfloat16 if amp_device_type == "cpu" else torch.float16
    offset = get_model_config(model, "i2i_offset")
    scale = get_model_config(model, "i2i_scale")
    acc = 8
    model = model.to(device).eval()
    criterion = nn.MSELoss().to(device)
    cal = RGBCalibration().to(device)
    cal.train()
    grad_scaler = torch.cuda.amp.GradScaler(enabled=amp)
    optimizer = torch.optim.Adam(cal.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.95)

    for epoch in range(epoch):
        losses = []
        c = 0
        for step in tqdm(range(steps // batch_size), ncols=80):
            optimizer.zero_grad()
            rgb = (torch.rand((batch_size, 3, 1, 1)) * 255).round() / 255.0
            x = rgb.expand((batch_size, 3, input_size, input_size)).clone().to(device)
            y = rgb.expand((batch_size, 3,
                            input_size * scale - offset * 2,
                            input_size * scale - offset * 2)).clone().to(device)
            with torch.autocast(device_type=amp_device_type, dtype=amp_dtype, enabled=amp):
                with torch.no_grad():
                    z = model.unet(x)
                z = cal(z)
                loss = criterion(z, y)
            losses.append(loss.item())
            grad_scaler.scale(loss).backward()
            c += 1
            if c % acc == 0:
                grad_scaler.step(optimizer)
                grad_scaler.update()
        scheduler.step()
        print(f"epoch {epoch}: loss={sum(losses) / len(losses)}, lr={scheduler.get_lr()}, RGB={cal.rgb.data.flatten().tolist()}")

    print(f"RGBCalibration: {cal.rgb.data.flatten().tolist()}")


if __name__ == "__main__":
    calibrate_output()
