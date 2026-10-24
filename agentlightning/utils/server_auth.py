# Copyright (c) Microsoft. All rights reserved.

"""Optional shared bearer-token authentication for Agent Lightning HTTP servers.

Trusted workers intentionally share one store, so the token is a single shared secret
rather than a per-worker identity. It is disabled unless a token is passed explicitly
or the `AGL_SERVER_TOKEN` environment variable is set.
"""

from __future__ import annotations

import hmac
from typing import Dict, Optional

from agentlightning.env_var import LightningEnvVar, resolve_str_env_var

__all__ = ["resolve_server_token", "bearer_headers", "is_authorized_header"]


def resolve_server_token(override: Optional[str] = None) -> Optional[str]:
    """Return the shared token from `override` or `AGL_SERVER_TOKEN`, or `None` when unset."""
    token = resolve_str_env_var(LightningEnvVar.AGL_SERVER_TOKEN, override=override)
    if token is None:
        return None
    token = token.strip()
    return token or None


def bearer_headers(token: Optional[str]) -> Dict[str, str]:
    """Return the `Authorization` header for `token`, or an empty dict when no token is configured."""
    if not token:
        return {}
    return {"Authorization": f"Bearer {token}"}


def is_authorized_header(authorization: Optional[str], token: str) -> bool:
    """Check an `Authorization` header value against the shared token in constant time."""
    if not authorization:
        return False
    scheme, _, credentials = authorization.partition(" ")
    if scheme.lower() != "bearer":
        return False
    return hmac.compare_digest(credentials.strip().encode("utf-8"), token.encode("utf-8"))
