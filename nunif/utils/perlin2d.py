# Original work is perlin-numpy: https://github.com/pvigier/perlin-numpy)
# Pierre Vigier / MIT License
# Vadim Kantorov ported to pytorch: https://gist.github.com/vadimkantorov/ac1b097753f217c5c11bc2ff396e0a57
# some minor changes by nagdaomi

import torch
import math


def generate_perlin_noise_2d(shape, res, tileable=(False, False), fade=lambda t: 6 * t**5 - 15 * t**4 + 10 * t**3):
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


def _test_tile():
    import torchvision.transforms.functional as TF
    import time

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

    noise = generate_perlin_noise_2d_octaves(shape, res, tileable=(True, True), octaves=2).unsqueeze(0)
    noise = (noise - noise.min()) / (noise.max() - noise.min())
    noise = torch.cat([noise, noise], dim=1)
    noise = torch.cat([noise, noise], dim=2)

    TF.to_pil_image(noise).show()


if __name__ == "__main__":
    _test_tile()
