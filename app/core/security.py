"""Shared secret authentication for trusted server-side integrations."""
import secrets
from fastapi import HTTPException, Request
from app.core import config

def _authenticate(request: Request):
    # Fail closed if no server-side token has been configured.
    if not config.API_TOKEN:
        raise HTTPException(status_code=503, detail="API authentication is not configured")
    auth = request.headers.get("authorization", "")
    scheme, _, token = auth.partition(" ")
    if scheme.lower() != "bearer" or not secrets.compare_digest(token, config.API_TOKEN):
        raise HTTPException(status_code=401, detail="Unauthorized", headers={"WWW-Authenticate": "Bearer"})


