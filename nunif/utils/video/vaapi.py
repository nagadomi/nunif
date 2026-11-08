def is_vaapi_codec(codec) -> bool:
    return isinstance(codec, str) and codec.endswith("_vaapi")
