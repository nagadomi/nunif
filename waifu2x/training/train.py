import argparse
from nunif.models import get_model_names
from .trainer import Waifu2xTrainer


def train(args):
    ARCH_SWIN_UNET = {"waifu2x.swin_unet_1x",
                      "waifu2x.swin_unet_2x",
                      "waifu2x.swin_unet_4x"}
    assert args.discriminator_stop_criteria < args.generator_start_criteria
    if args.size % 4 != 0:
        raise ValueError("--size must be a multiple of 4")
    if args.arch in ARCH_SWIN_UNET and ((args.size - 16) % 12 != 0 or (args.size - 16) % 16 != 0):
        raise ValueError("--size must be `(SIZE - 16) % 12 == 0 and (SIZE - 16) % 16 == 0` for SwinUNet models")
    if args.method in {"noise", "noise_scale", "noise_scale4x"} and args.noise_level is None:
        raise ValueError("--noise-level is required for noise/noise_scale")
    if args.pre_antialias and args.arch != "waifu2x.swin_unet_4x":
        raise ValueError("--pre-antialias is only supported for waifu2x.swin_unet_4x")

    if args.method in {"scale", "scale4x", "scale8x"}:
        # disable
        args.noise_level = -1

    if args.loss is None:
        if args.arch in {"waifu2x.vgg_7", "waifu2x.upconv_7"}:
            args.loss = "y_charbonnier"
        elif args.arch in {"waifu2x.cunet", "waifu2x.upcunet"}:
            args.loss = "aux_lbp"
        elif args.arch in {"waifu2x.swin_unet_1x", "waifu2x.swin_unet_2x"}:
            args.loss = "lbp"
        elif args.arch in {"waifu2x.swin_unet_4x"}:
            args.loss = "lbp5"
        elif args.arch in {"waifu2x.swin_unet_8x"}:
            args.loss = "y_charbonnier"
        else:
            args.loss = "y_charbonnier"

    trainer = Waifu2xTrainer(args)
    trainer.fit()


def register(subparsers, default_parser):
    parser = subparsers.add_parser(
        "waifu2x",
        parents=[default_parser],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    waifu2x_models = sorted([name for name in get_model_names() if name.startswith("waifu2x.")])

    parser.add_argument("--method", type=str,
                        choices=["noise", "scale", "noise_scale",
                                 "scale4x", "noise_scale4x",
                                 "scale8x", "noise_scale8x"],
                        required=True,
                        help="waifu2x method")
    parser.add_argument("--arch", type=str,
                        choices=waifu2x_models,
                        required=True,
                        help="network arch")
    parser.add_argument("--style", type=str,
                        choices=["art", "photo"],
                        default="art",
                        help="image style used for jpeg noise level")
    parser.add_argument("--noise-level", type=int,
                        choices=[0, 1, 2, 3],
                        help="jpeg noise level for noise/noise_scale")
    parser.add_argument("--size", type=int, default=112,
                        help="input size")
    parser.add_argument("--num-samples", type=int, default=50000,
                        help="number of samples for each epoch")
    parser.add_argument("--loss", type=str,
                        choices=["lbp", "lbp5", "lbpm", "lbp5m", "y_charbonnier", "charbonnier",
                                 "aux_lbp", "aux_y_charbonnier", "aux_charbonnier",
                                 "alex11", "aux_alex11", "l1", "y_l1", "l1lpips"],
                        help="loss function")
    parser.add_argument("--da-jpeg-p", type=float, default=0.0,
                        help="HQ JPEG(quality=92-99) data augmentation for gt image")
    parser.add_argument("--da-scale-p", type=float, default=0.25,
                        help="random downscale data augmentation for gt image")
    parser.add_argument("--da-chshuf-p", type=float, default=0.0,
                        help="random channel shuffle data augmentation for gt image")
    parser.add_argument("--da-unsharpmask-p", type=float, default=0.0,
                        help="random unsharp mask data augmentation for gt image")
    parser.add_argument("--da-grayscale-p", type=float, default=0.0,
                        help="random grayscale data augmentation for gt image")
    parser.add_argument("--da-color-p", type=float, default=0.0,
                        help="random color jitter data augmentation for gt image")
    parser.add_argument("--deblur", type=float, default=0.0,
                        help=("shift parameter of resize blur."
                              " 0.0-0.1 is a reasonable value."
                              " blur = uniform(0.95 + deblur, 1.05 + deblur)."
                              " blur >= 1 is blur, blur <= 1 is sharpen. mean 1 by default"))
    parser.add_argument("--resize-blur-p", type=float, default=0.1,
                        help=("probability that resize blur should be used"))
    parser.add_argument("--hard-example", type=str, default="linear",
                        choices=["none", "linear", "top10", "top20"],
                        help="hard example mining for training data sampleing")
    parser.add_argument("--hard-example-scale", type=float, default=4.,
                        help="max weight scaling factor of hard example sampler")
    parser.add_argument("--b4b", action="store_true",
                        help="use only bicubic downsampling for bicubic downsampling restoration")
    parser.add_argument("--freeze", action="store_true",
                        help="call model.freeze() if avaliable")
    # GAN related options
    parser.add_argument("--discriminator", type=str,
                        help="discriminator name or .pth or [`l3`, `l3c`, `l3v1`, `l3v1`].")
    parser.add_argument("--discriminator-weight", type=float, default=1.0,
                        help="discriminator loss weight")
    parser.add_argument("--update-criterion", type=str, choices=["psnr", "loss", "all"], default="psnr",
                        help=("criterion for updating the best model file. "
                              "`all` forced to saves the best model each epoch."))
    parser.add_argument("--discriminator-only", action="store_true",
                        help="training discriminator only")
    parser.add_argument("--discriminator-stop-criteria", type=float, default=0.5,
                        help=("When the loss of the discriminator is less than the specified value,"
                              " stops training of the discriminator."
                              " This is the limit to prevent too strong discriminator."
                              " Also, the discriminator skip probability is interpolated between --generator-start-criteria and --discriminator-stop-criteria."))
    parser.add_argument("--generator-start-criteria", type=float, default=0.9,
                        help=("When the loss of the discriminator is greater than the specified value,"
                              " stops training of the generator."
                              " This is the limit to prevent too strong generator."
                              " Also do not hit the newbie discriminator."))
    parser.add_argument("--discriminator-learning-rate", type=float,
                        help=("learning-rate for discriminator. --learning-rate by default."))
    parser.add_argument("--pre-antialias", action="store_true",
                        help=("Set `pre_antialias=True` for SwinUNet4x."))

    parser.set_defaults(
        batch_size=16,
        optimizer="adamw",
        learning_rate=0.0002,
        scheduler="cosine",
        learning_rate_cycles=5,
        learning_rate_decay=0.995,
        learning_rate_decay_step=[1],
        # for adamw
        weight_decay=0.001,
    )
    parser.set_defaults(handler=train)

    return parser
