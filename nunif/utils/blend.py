# layer composite / blend mode
# TODO: optimize
import torch


def blend(a, b, alpha):
    return a * alpha + b * (1. - alpha)


def multiply(a, b):
    return a * b


def screen(a, b):
    return 1. - (2. * (1. - a) * (1. - b))


def overlay(a, b):
    out = torch.empty(a.shape, dtype=a.dtype, device=a.device)
    th = a < 0.5
    th_not = torch.logical_not(th)
    out[th] = 2. * multiply(a[th], b[th])
    out[th_not] = screen(a[th_not], b[th_not])

    return out


def hardlight(a, b):
    return overlay(b, a)


def softlight(a, b):
    # this formula adopts W3C spec, not photoshop spec.
    out = torch.empty(a.shape, dtype=a.dtype, device=a.device)

    th1 = b <= 0.5
    th1_not = torch.logical_not(th1)
    out[th1] = a[th1] - (1. - 2. * b[th1]) * a[th1] * (1. - a[th1])

    th2 = torch.logical_and(th1_not, a <= 0.25)
    th2_not = torch.logical_and(th1_not, torch.logical_not(th2))

    out[th2] = a[th2] + (2. * b[th2] - 1) * ((((16. * a[th2] - 12.) * a[th2] + 4.) * a[th2]) - a[th2])
    out[th2_not] = a[th2_not] + (2. * b[th2_not] - 1) * (torch.sqrt(a[th2_not]) - a[th2_not])
    return out


def lighten(a, b):
    return torch.maximum(a, b)


def darken(a, b):
    return torch.minimum(a, b)


def _test():
    """
    I have visually checked that
    the result of this code is roughly the same as
    the result of the layer composition mode in GIMP
    """
    from PIL import Image, ImageDraw
    from . import pil_io
    import cv2

    def show(name, im):
        cv2.imshow(name, pil_io.to_cv2(im))

    def show_op(func, a, b):
        show(func.__name__, pil_io.to_image(func(pil_io.to_tensor(a), pil_io.to_tensor(b))))

    a = Image.open("waifu2x/docs/images/miku_128.png")
    b = Image.new("RGB", (128, 128), (200, 200, 200))
    ImageDraw.Draw(b).rectangle([0, 0, 64, 128], fill=(50, 50, 50))
    show("a", a)
    show("b", b)
    show_op(multiply, a, b)
    show_op(overlay, a, b)
    show_op(screen, a, b)
    show_op(hardlight, a, b)
    show_op(softlight, a, b)
    cv2.waitKey(0)


if __name__ == "__main__":
    _test()
