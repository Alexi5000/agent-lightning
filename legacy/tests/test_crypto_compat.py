# Copyright (c) Microsoft. All rights reserved.

import base64
import json
from datetime import datetime, timedelta, timezone
from typing import Any

import jwt
import pytest
from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding, rsa
from cryptography.x509.oid import NameOID
from jwt.algorithms import RSAAlgorithm
from litellm.proxy.auth.handle_jwt import JWTHandler
from litellm.proxy.auth.litellm_license import LicenseCheck


@pytest.mark.asyncio
@pytest.mark.parametrize("certificate", [False, True])
async def test_litellm_jwt_crypto_compatibility(certificate: bool, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("JWT_AUDIENCE", raising=False)
    private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    public_key = private_key.public_key()
    if certificate:
        name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "offline.test")])
        now = datetime.now(timezone.utc)
        cert = (
            x509.CertificateBuilder()
            .subject_name(name)
            .issuer_name(name)
            .public_key(public_key)
            .serial_number(x509.random_serial_number())
            .not_valid_before(now - timedelta(minutes=1))
            .not_valid_after(now + timedelta(days=1))
            .sign(private_key, hashes.SHA256())
        )
        key: Any = cert.public_bytes(serialization.Encoding.PEM).decode()
    else:
        key = json.loads(RSAAlgorithm.to_jwk(public_key))

    async def local_key(kid: str | None) -> Any:
        assert kid is None
        return key

    handler: Any = JWTHandler()
    monkeypatch.setattr(handler, "get_public_key", local_key)
    try:
        token = jwt.encode({"sub": "offline-user"}, private_key, algorithm="RS256")
        assert (await handler.auth_jwt(token))["sub"] == "offline-user"
        other_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
        invalid = jwt.encode({"sub": "offline-user"}, other_key, algorithm="RS256")
        with pytest.raises(Exception, match="Validation fails"):
            await handler.auth_jwt(invalid)
    finally:
        await handler.close()


def test_litellm_offline_license_signature(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LITELLM_LICENSE", raising=False)
    private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    message = json.dumps({"expiration_date": "2099-01-01", "max_users": 10}).encode()
    signature = private_key.sign(
        message,
        padding.PSS(mgf=padding.MGF1(hashes.SHA256()), salt_length=padding.PSS.MAX_LENGTH),
        hashes.SHA256(),
    )
    checker: Any = LicenseCheck()
    valid = base64.b64encode(message + b"." + signature).decode()
    assert checker.verify_license_without_api_request(private_key.public_key(), valid) is True
    invalid = base64.b64encode(message.replace(b"10", b"11") + b"." + signature).decode()
    assert checker.verify_license_without_api_request(private_key.public_key(), invalid) is False
