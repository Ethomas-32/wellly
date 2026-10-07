"""
Persistent per-WhatsApp-number user profiles (currently just a name),
backed by a simple JSON file so it survives server restarts.

For an MVP with a handful of users this is plenty. If you outgrow it,
swap the _load()/_save() functions for a real database (SQLite is the
natural next step) — everything else calling into this module stays
the same.
"""
import json
import os

PROFILES_FILE = "user_profiles.json"


def _load() -> dict:
    if not os.path.exists(PROFILES_FILE):
        return {}
    with open(PROFILES_FILE, "r", encoding="utf-8") as f:
        try:
            return json.load(f)
        except json.JSONDecodeError:
            return {}


def _save(profiles: dict) -> None:
    with open(PROFILES_FILE, "w", encoding="utf-8") as f:
        json.dump(profiles, f, indent=2, ensure_ascii=False)


def get_profile(sender: str) -> dict | None:
    """Return this sender's profile dict, or None if we've never seen them."""
    return _load().get(sender)


def get_name(sender: str) -> str | None:
    profile = get_profile(sender)
    return profile.get("name") if profile else None


def is_awaiting_name(sender: str) -> bool:
    profile = get_profile(sender)
    return bool(profile and profile.get("awaiting_name"))


def start_onboarding(sender: str) -> None:
    """Mark this sender as new — we've asked for their name and are waiting for it."""
    profiles = _load()
    profiles[sender] = {"name": None, "awaiting_name": True}
    _save(profiles)


def save_name(sender: str, name: str) -> None:
    """Store the name they gave us and mark onboarding as complete."""
    profiles = _load()
    profiles[sender] = {"name": name.strip(), "awaiting_name": False}
    _save(profiles)


def clear_profile(sender: str) -> None:
    """Remove a single sender's profile — useful when re-testing onboarding."""
    profiles = _load()
    profiles.pop(sender, None)
    _save(profiles)


def clear_all() -> None:
    """Wipe every stored profile."""
    _save({})