# Colorspace

`Colorspace`(`--colorspace`) option allows you to specify the YUV `colorspace` and `color_range` (Dynamic Range) of the output video 

| Colorspace  | |
|-------------|-|
| unspecified | Do Nothing. It works the same as the older iw3 version. 
| auto     | Use the same `colorspace` and `color_range` as the input video.
| bt709    | Use bt709 for `colorspace`. `color_range` will be the same as the input video.
| bt709-pc | BT.709 Full Range
| bt709-tv | BT.709 Limited Range
| bt601\*  | BT.601 version of `bt709*`

## Difference between `unspecified` and other options

### `unspecified`

When `unspecified` is specified, the stereo generation process (expecting full range input) is performed without converting the dynamic range of the frame images, even if the video is limited range.

This can cause color changes and banding artifact.

### option other than `unspecified`

When an option other than `unspecified` is specified, the frame image is converted to full range before stereo generation processing is performed.
And if the output is limited range, it is re-converted from full range to limited range when video encoding.

This is a more correct process than `unspecified`.
However, if the `colorspace` and `color_range` of the input video are unknown or have undefined values, the color conversion may not be correct or may result in an error.

## Relationship between Colorspace and Pixel Format

When `rgb24` is specified for `Pixel Format`, the behavior is the same as the `unspecified` case, no matter what `Colorspace` is specified.

When `yuv420p` or `yuv444p` is specified for `Pixel Format` and the output `color_range` is full range(tv), `yuvj420p` or `yuvj444p` will be used instead.

## Case where `colorspace` of the input video is unknown

If the input video is smaller than 720p, BT.601 is used, otherwise BT.709 is used.

## Cases where `color_range` of the input video is unknown

Determines `color_range` from `pix_fmt`. `yuv4*` is limited range(tv), `yuvj4*`, `rgb*`, `gbr*` is full range(pc).

If none of the above, leave unknown (unspecified).

## Cases where `colorspace` or `color_range` of the input video is still unknown

If only one of the two is unknown, the conversion is performed using the (forced)guessed value.
If both are unknown, it works the same as if `unspecified` is specified.

## tv -> pc, pc -> tv conversion

The dynamic range changes between the input video and the output video, so the colors will change.

## Export / Export Disparity

### `unspecified`

Outputs frame images without converting dynamic range.
The dynamic range of the frame image also differs depending on whether the input video is limited range(tv) or full range(pc).

### option other than `unspecified`

Outputs frame images converted to **full range**.
If the video is limited range, the color will change on the frame image. If the video is encoded to limited range on import, the color of the final video will not change.

## Import (yml input)

### Case exported with options other than `unspecified`

If exported specifying other than `unspecified`, `source_color_range` and`output_colorspace` fileds are added to the yml file.

`source_color_range` is the dynamic range of the input video (1: MPEG Limited Range, 2: JPEG Full Range).
`output_colorspace` is the color space (1: BT.709, 5: BT.601) in YUV of the frame image. 
Note that `source_color_range` is the dynamic range of the source video, not the dynamic range of the frame image. The frame image is always full range in this case.
This fields are used in `auto` or `colorspace` and `color_range` conversions during import.

### Case where `source_color_range`,`output_colorspace` does not exist in yml file

The reason for the missing `source_color_range`,`output_colorspace` in the yml file is one of the following.

- Exported with an older iw3 version
- Exported with Colorspace=unspecified
- Both `colorspace` and `color_range` of the input video were unknown

If `source_color_range` and `output_colorspace` are missing in the yml file, the frame image is assumed to be the same as `colorspace` and `color_range` specified by `Colorspace`.

For example,

- When `bt709-tv` is specified, the frame image is processed as BT.709 Limited Range to rgb24 conversion result.
- When `bt709-pc` is specified, the frame image is processed as BT.709 Full Range to rgb24 conversion result.
