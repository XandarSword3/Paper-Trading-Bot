"""Tests for the V4 live recorder boot registration script."""

from __future__ import annotations

from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "register_live_tasks.ps1"


def test_registration_script_exists():
    assert SCRIPT.exists(), "scripts/register_live_tasks.ps1 is missing"


def test_registration_script_properties():
    text = SCRIPT.read_text(encoding="utf-8")
    # Own account, not SYSTEM
    assert "SYSTEM" not in text.upper().replace("NOT SYSTEM", "")
    assert "$env:USERDOMAIN\\$env:USERNAME" in text
    # Without highest privileges
    assert 'RunLevel    = "Limited"' in text or "RunLevel" in text and "Limited" in text
    # At startup
    assert "New-ScheduledTaskTrigger -AtStartup" in text
    # No execution time limit
    assert "ExecutionTimeLimit" in text and "[TimeSpan]::Zero" in text
    # Allowed on battery, keeps running on battery
    assert "AllowStartIfOnBatteries" in text
    assert "DontStopIfGoingOnBatteries" in text
    # Restart on failure
    assert "RestartCount" in text and "RestartInterval" in text
    # Password typed by the user, not hardcoded
    assert "Get-Credential" in text
    assert "Password" in text and "$password" in text
