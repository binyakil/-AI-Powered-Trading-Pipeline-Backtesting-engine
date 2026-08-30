"""
Julaba Hardware Identity Lock (Stub)
=====================================
Placeholder module — all checks pass.
"""

import logging

logger = logging.getLogger(__name__)

PERM_OWNER = "owner"
PERM_RUN_ONLY = "run-only"


def verify_hardware_lock() -> bool:
    """Always returns True — no hardware lock enforced."""
    return True


def enforce_hardware_lock():
    """No-op — hardware lock disabled."""
    pass


def enforce_file_integrity():
    """No-op — file integrity check disabled."""
    pass


def _get_permission_level() -> str:
    """Always returns run-only."""
    return PERM_RUN_ONLY


def _compute_fingerprint() -> str:
    """Return empty fingerprint."""
    return "stub"


def _is_master_password_set() -> bool:
    """No master password."""
    return False


def _verify_master_password(pw: str) -> bool:
    """Always returns False."""
    return False


def _set_master_password(pw: str) -> bool:
    """No-op."""
    return False


def _set_permission_level(level: str) -> bool:
    """No-op."""
    return False
