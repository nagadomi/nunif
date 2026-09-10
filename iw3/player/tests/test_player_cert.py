"""Test the real generator without importing GPU/server dependencies.

Run: python iw3/player/tests/test_player_cert.py
Requires cryptography (also a dependency of the existing pyOpenSSL requirement).
Only the trusted local function is extracted; no network or user code is run.
"""
import ast
import ipaddress
import os
from pathlib import Path
import ssl
import tempfile
from datetime import datetime, timezone
from cryptography import x509
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import padding

path = Path(__file__).resolve().parents[1] / 'server.py'
tree = ast.parse(path.read_text(encoding='utf-8'))
fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'generate_self_signed_cert')
namespace = {'os': os, 'ipaddress': ipaddress}
exec(compile(ast.Module(body=[fn], type_ignores=[]), str(path), 'exec'), namespace)
generate = namespace['generate_self_signed_cert']
for address in ('192.0.2.10', '127.0.0.1', '::1'):
    with tempfile.TemporaryDirectory(prefix='iw3-cert-test-') as temp:
        directory = Path(temp) / 'certs'
        cert_path, key_path = directory / 'server.crt', directory / 'server.key'
        generate(str(directory), str(cert_path), str(key_path), address)
        cert_bytes, key_bytes = cert_path.read_bytes(), key_path.read_bytes()
        cert = x509.load_pem_x509_certificate(cert_bytes)
        key = serialization.load_pem_private_key(key_bytes, password=None)
        assert cert.public_key().public_numbers() == key.public_key().public_numbers()
        cert.public_key().verify(cert.signature, cert.tbs_certificate_bytes, padding.PKCS1v15(), cert.signature_hash_algorithm)
        san = cert.extensions.get_extension_for_class(x509.SubjectAlternativeName).value
        assert 'localhost' in san.get_values_for_type(x509.DNSName)
        assert ipaddress.ip_address(address) in san.get_values_for_type(x509.IPAddress)
        assert ipaddress.ip_address('127.0.0.1') in san.get_values_for_type(x509.IPAddress)
        assert len(list(san)) == len(set(san))
        assert cert.not_valid_before_utc <= datetime.now(timezone.utc) < cert.not_valid_after_utc
        context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        context.load_cert_chain(cert_path, key_path)
        generate(str(directory), str(cert_path), str(key_path), address)
        assert cert_path.read_bytes() == cert_bytes and key_path.read_bytes() == key_bytes
        print('PASS: certificate signature, key, SANs, TLS loading and preservation for', address)
print('PASS: all certificate regression tests')
