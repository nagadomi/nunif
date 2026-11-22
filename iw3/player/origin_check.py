from urllib.parse import urlsplit


def is_same_origin_request(
    origin: str | None,
    sec_fetch_site: str | None,
    host: str | None,
    scheme: str,
) -> bool:
    """Validate Origin and Sec-Fetch-Site headers against the request host."""
    if sec_fetch_site is not None:
        if sec_fetch_site.strip().lower() not in {"same-origin", "none"}:
            return False

    if origin is None:
        return True

    if not host:
        return False

    expected_scheme = "https" if scheme.lower() in {"https", "wss"} else "http"
    default_port = 443 if expected_scheme == "https" else 80

    try:
        parsed_origin = urlsplit(origin.strip())
        parsed_host = urlsplit(f"{expected_scheme}://{host.strip()}")
        raw_origin_port = parsed_origin.port
        raw_host_port = parsed_host.port
    except ValueError:
        return False

    if (
        not parsed_origin.scheme
        or not parsed_origin.hostname
        or parsed_origin.username is not None
        or parsed_origin.password is not None
        or parsed_origin.path not in ("", "/")
        or parsed_origin.query
        or parsed_origin.fragment
    ):
        return False

    if (
        not parsed_host.hostname
        or parsed_host.username is not None
        or parsed_host.password is not None
        or parsed_host.path
        or parsed_host.query
        or parsed_host.fragment
    ):
        return False

    origin_port = raw_origin_port if raw_origin_port is not None else default_port
    host_port = raw_host_port if raw_host_port is not None else default_port

    return (
        parsed_origin.scheme.lower() == expected_scheme
        and parsed_origin.hostname.lower() == parsed_host.hostname.lower()
        and origin_port == host_port
    )
