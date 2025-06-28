# Original work is perlin-numpy: https://github.com/pvigier/perlin-numpy)
# Pierre Vigier / MIT License
# Vadim Kantorov ported to pytorch: https://gist.github.com/vadimkantorov/ac1b097753f217c5c11bc2ff396e0a57
# some minor changes by nagdaomi

import torch
import math


def interpolant(t):
    return 6 * t**5 - 15 * t**4 + 10 * t**3


def generate_perlin_noise_2d(shape, res, tileable=(False, False), fade=interpolant):
    delta = (res[0] / shape[0], res[1] / shape[1])
    d = (shape[0] // res[0], shape[1] // res[1])

    grid = torch.stack(torch.meshgrid(
        torch.arange(0, res[0], delta[0]),
        torch.arange(0, res[1], delta[1]), indexing="ij"), dim=-1) % 1
    angles = 2. * math.pi * torch.rand(res[0] + 1, res[1] + 1)
    gradients = torch.stack((torch.cos(angles), torch.sin(angles)), dim=-1)

    if tileable[0]:
        gradients[-1, :] = gradients[0, :]
    if tileable[1]:
        gradients[:, -1] = gradients[:, 0]

    tile_grads = lambda slice1, slice2: gradients[slice1[0]:slice1[1], slice2[0]:slice2[1]].repeat_interleave(d[0], 0).repeat_interleave(d[1], 1)
    dot = lambda grad, shift: (torch.stack((grid[:shape[0], :shape[1], 0] + shift[0], grid[:shape[0], :shape[1], 1] + shift[1]), dim=-1) * grad[:shape[0], :shape[1]]).sum(dim=-1)

    n00 = dot(tile_grads([0, -1], [0, -1]), [0, 0])
    n10 = dot(tile_grads([1, None], [0, -1]), [-1, 0])
    n01 = dot(tile_grads([0, -1], [1, None]), [0, -1])
    n11 = dot(tile_grads([1, None], [1, None]), [-1, -1])
    t = fade(grid[:shape[0], :shape[1]])

    return math.sqrt(2) * torch.lerp(torch.lerp(n00, n10, t[..., 0]), torch.lerp(n01, n11, t[..., 0]), t[..., 1])


def generate_perlin_noise_2d_octaves(shape, res, tileable=(False, False), octaves=1, persistence=0.5):
    noise = torch.zeros(shape)
    frequency = 1
    amplitude = 1
    for _ in range(octaves):
        noise += amplitude * generate_perlin_noise_2d(shape, (frequency * res[0], frequency * res[1]), tileable=tileable)
        frequency *= 2
        amplitude *= persistence
    return noise


def generate_perlin_noise_3d(shape, res, tileable=(False, False, False), fade=interpolant, device=None):
    if device is None:
        device = "cpu"

    delta = (res[0] / shape[0], res[1] / shape[1], res[2] / shape[2])
    d = (shape[0] // res[0], shape[1] // res[1], shape[2] // res[2])

    grid = torch.stack(
        torch.meshgrid(
            torch.arange(0, res[0], delta[0], device=device),
            torch.arange(0, res[1], delta[1], device=device),
            torch.arange(0, res[2], delta[2], device=device),
            indexing="ij"
        ),
        dim=-1
    ) % 1

    # Gradients
    theta = 2 * torch.pi * torch.rand((res[0] + 1, res[1] + 1, res[2] + 1), device=device)
    phi = 2 * torch.pi * torch.rand((res[0] + 1, res[1] + 1, res[2] + 1), device=device)
    gradients = torch.stack((torch.sin(phi) * torch.cos(theta), torch.sin(phi) * torch.sin(theta), torch.cos(phi),), dim=3)

    if tileable[0]:
        gradients[-1, :, :] = gradients[0, :, :]
    if tileable[1]:
        gradients[:, -1, :] = gradients[:, 0, :]
    if tileable[2]:
        gradients[:, :, -1] = gradients[:, :, 0]

    gradients = gradients.repeat_interleave(d[0], dim=0).repeat_interleave(d[1], dim=1).repeat_interleave(d[2], dim=2)

    g000 = gradients[:-d[0], :-d[1], :-d[2]]
    g100 = gradients[d[0]:, :-d[1], :-d[2]]
    g010 = gradients[:-d[0], d[1]:, :-d[2]]
    g110 = gradients[d[0]:, d[1]:, :-d[2]]
    g001 = gradients[:-d[0], :-d[1], d[2]:]
    g101 = gradients[d[0]:, :-d[1], d[2]:]
    g011 = gradients[:-d[0], d[1]:, d[2]:]
    g111 = gradients[d[0]:, d[1]:, d[2]:]

    # Ramps
    ramp = lambda shift: (grid - torch.tensor(shift, device=device))
    print(ramp((0, 0, 0)).shape, g000.shape)
    n000 = (ramp((0, 0, 0)) * g000).sum(dim=3)
    n100 = (ramp((1, 0, 0)) * g100).sum(dim=3)
    n010 = (ramp((0, 1, 0)) * g010).sum(dim=3)
    n110 = (ramp((1, 1, 0)) * g110).sum(dim=3)
    n001 = (ramp((0, 0, 1)) * g001).sum(dim=3)
    n101 = (ramp((1, 0, 1)) * g101).sum(dim=3)
    n011 = (ramp((0, 1, 1)) * g011).sum(dim=3)
    n111 = (ramp((1, 1, 1)) * g111).sum(dim=3)

    # Interpolation
    t = interpolant(grid)
    n00 = n000 * (1 - t[:, :, :, 0]) + t[:, :, :, 0] * n100
    n10 = n010 * (1 - t[:, :, :, 0]) + t[:, :, :, 0] * n110
    n01 = n001 * (1 - t[:, :, :, 0]) + t[:, :, :, 0] * n101
    n11 = n011 * (1 - t[:, :, :, 0]) + t[:, :, :, 0] * n111

    n0 = (1 - t[:, :, :, 1]) * n00 + t[:, :, :, 1] * n10
    n1 = (1 - t[:, :, :, 1]) * n01 + t[:, :, :, 1] * n11

    return ((1 - t[:, :, :, 2]) * n0 + t[:, :, :, 2] * n1)


def _test_tile():
    import torchvision.transforms.functional as TF
    import time

    torch.manual_seed(1)
    S = 128
    RES = 4
    shape = (S, S)
    res = (S // RES // 2, S // RES // 2)
    noise = generate_perlin_noise_2d(shape, res, tileable=(True, True)).unsqueeze(0)
    noise = (noise - noise.min()) / (noise.max() - noise.min())
    noise = torch.cat([noise, noise], dim=1)
    noise = torch.cat([noise, noise], dim=2)

    TF.to_pil_image(noise).show()
    time.sleep(2)

    torch.manual_seed(1)

    noise = generate_perlin_noise_2d_octaves(shape, res, tileable=(True, True), octaves=2).unsqueeze(0)
    noise = (noise - noise.min()) / (noise.max() - noise.min())
    noise = torch.cat([noise, noise], dim=1)
    noise = torch.cat([noise, noise], dim=2)

    TF.to_pil_image(noise).show()


def _test_3d():
    import torchvision.transforms.functional as TF
    import torchvision.io as IO
    from os import path
    import time

    im = IO.read_image("cc0/320/dog.png") / 255.0
    print(im.shape)

    torch.manual_seed(1)
    S = max(im.shape[1], im.shape[2])
    RES = 1
    T = 30 * 4
    shape = (T, S, S)
    res = (T // 2, S // RES // 2, S // RES // 2)

    def gen_noise(scale):
        noise = generate_perlin_noise_3d(shape, (res[0] // scale, res[1] * scale, res[2] * scale),
                                         tileable=(True, False, False), device="cuda").unsqueeze(1).cpu()
        return noise

    noise = gen_noise(2) + gen_noise(2)
    noise = (noise - noise.min()) / (noise.max() - noise.min())
    noise = noise.expand(T, 3, S, S)

    image_list = [
        TF.to_pil_image(torch.clamp(im + noise[i] * 0.1, 0, 1))
        for i in range(noise.shape[0])]
    image_list[0].save(
        path.join("tmp", "perlin3d.gif"), format="gif",
        append_images=image_list, save_all=True,
        duration=1000/30, loop=1)


if __name__ == "__main__":
    #_test_tile()
    _test_3d()
