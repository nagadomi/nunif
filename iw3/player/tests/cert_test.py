import datetime
import ipaddress
import os
import ssl
import stat
import sys
from pathlib import Path

import pytest
from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.x509.oid import NameOID

from ..server import create_hypercorn_config, ensure_server_credentials, generate_self_signed_cert


@pytest.mark.parametrize(
    ("bind_addr", "expected_dns", "expected_ips"),
    [
        ("127.0.0.1", ["localhost"], [ipaddress.ip_address("127.0.0.1")]),
        (
            "192.168.1.10",
            ["localhost"],
            [ipaddress.ip_address("127.0.0.1"), ipaddress.ip_address("192.168.1.10")],
        ),
        ("example.local", ["localhost", "example.local"], [ipaddress.ip_address("127.0.0.1")]),
    ],
)
def test_generate_self_signed_cert_properties(
    tmp_path: Path,
    bind_addr: str,
    expected_dns: list[str],
    expected_ips: list[ipaddress.IPv4Address | ipaddress.IPv6Address],
) -> None:
    cert_file = tmp_path / "server.crt"
    key_file = tmp_path / "server.key"

    generate_self_signed_cert(str(tmp_path), str(cert_file), str(key_file), bind_addr)

    assert "OpenSSL" not in sys.modules
    assert cert_file.exists()
    assert key_file.exists()

    cert = x509.load_pem_x509_certificate(cert_file.read_bytes())
    private_key = serialization.load_pem_private_key(key_file.read_bytes(), password=None)

    assert isinstance(private_key, rsa.RSAPrivateKey)
    assert private_key.key_size == 2048
    assert cert.version == x509.Version.v3
    assert cert.subject == cert.issuer
    assert cert.subject.get_attributes_for_oid(NameOID.STATE_OR_PROVINCE_NAME)[0].value == "Tokyo"
    assert cert.subject.get_attributes_for_oid(NameOID.LOCALITY_NAME)[0].value == "Local"
    assert cert.subject.get_attributes_for_oid(NameOID.ORGANIZATION_NAME)[0].value == "My App"
    assert cert.subject.get_attributes_for_oid(NameOID.COMMON_NAME)[0].value == bind_addr
    assert isinstance(cert.signature_hash_algorithm, hashes.SHA256)

    validity = cert.not_valid_after_utc - cert.not_valid_before_utc
    assert validity == datetime.timedelta(days=10 * 365)

    san = cert.extensions.get_extension_for_class(x509.SubjectAlternativeName).value
    assert san.get_values_for_type(x509.DNSName) == expected_dns
    assert san.get_values_for_type(x509.IPAddress) == expected_ips

    ssl_ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    ssl_ctx.load_cert_chain(certfile=str(cert_file), keyfile=str(key_file))

    config = create_hypercorn_config("127.0.0.1", 1304, cert_file=str(cert_file), key_file=str(key_file))
    assert config.create_ssl_context() is not None


def test_existing_cert_preserved_and_compatible(tmp_path: Path) -> None:
    cert_file = tmp_path / "server.crt"
    key_file = tmp_path / "server.key"

    # Simulate a legacy certificate using TraditionalOpenSSL (PKCS#1) RSA key format.
    legacy_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    now = datetime.datetime.now(datetime.timezone.utc)
    subject = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "192.168.0.50")])
    legacy_cert = (
        x509.CertificateBuilder()
        .subject_name(subject)
        .issuer_name(subject)
        .public_key(legacy_key.public_key())
        .serial_number(12345)
        .not_valid_before(now)
        .not_valid_after(now + datetime.timedelta(days=365))
        .sign(private_key=legacy_key, algorithm=hashes.SHA256())
    )
    legacy_cert_bytes = legacy_cert.public_bytes(serialization.Encoding.PEM)
    legacy_key_bytes = legacy_key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.TraditionalOpenSSL,
        encryption_algorithm=serialization.NoEncryption(),
    )
    cert_file.write_bytes(legacy_cert_bytes)
    key_file.write_bytes(legacy_key_bytes)

    generate_self_signed_cert(str(tmp_path), str(cert_file), str(key_file), "127.0.0.1")

    assert cert_file.read_bytes() == legacy_cert_bytes
    assert key_file.read_bytes() == legacy_key_bytes

    config = create_hypercorn_config("127.0.0.1", 1304, cert_file=str(cert_file), key_file=str(key_file))
    assert config.create_ssl_context() is not None


@pytest.mark.skipif(os.name != "posix", reason="POSIX file mode checks apply on POSIX platforms")
def test_private_key_and_secret_permissions(tmp_path: Path) -> None:
    cert_dir = tmp_path / "certs"
    cert_file = cert_dir / "server.crt"
    key_file = cert_dir / "server.key"
    secret_key_file = cert_dir / "secret.key"

    ensure_server_credentials(
        bind_addr="127.0.0.1",
        cert_dir=str(cert_dir),
        cert_file=str(cert_file),
        key_file=str(key_file),
        secret_key_file=str(secret_key_file),
    )

    assert stat.S_IMODE(cert_dir.stat().st_mode) == 0o700
    assert stat.S_IMODE(key_file.stat().st_mode) == 0o600
    assert stat.S_IMODE(secret_key_file.stat().st_mode) == 0o600
