# Font Resource

Font related resources and their metadata for generating synthetic text image data.

Currently only Japanese Font is supported.

# Download Fonts

```
python3 -m font_resource.download_google_fonts
```
The font files are stored in `font_resource/fonts`.
If a font is already exists, it will be skipped.

# Generate Font List

[Font List](docs/font_list.md)

Show simple font list.
```
python3 -m font_resource.list
```

Show detailed font list.
```
python3 -m font_resource.list --metadata
```

Output in markdown format.
```
python3 -m font_resource.list --markdown
python3 -m font_resource.list --metadata --markdown
```

# File Index

TODO: write this
