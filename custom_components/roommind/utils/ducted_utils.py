"""Helpers for ducted (zoned, shared-unit) systems and continuous internal heat gains.

Ducted: one central unit feeds several rooms through dampers. A room's device reports "cooling" while the unit runs and
its damper is open, but how much it actually gets depends on its share of the airflow, which moves whenever another
zone opens or closes. A device (or an adapter entity in front of a zone) can publish that share as the attribute
``delivered_power_fraction`` (0..1, 1 = this room gets the unit's full output). RoomMind then uses it as the EKF
training label instead of a flat 1.0, so beta_c is learned as "cooling rate at full share" and stays valid whatever
the other zones are doing.

Internal gains: a room can list power sensors (W or kW) in ``heat_gain_sensors`` (PCs, a gym machine, a fridge...).
Their sum in kW becomes the occupancy heat input, so beta_o is learned per kW instead of per "someone is in".
"""

from __future__ import annotations

import math
from typing import Any

from homeassistant.core import HomeAssistant

DELIVERED_PF_ATTR = "delivered_power_fraction"
# Long-run share while the unit runs, published by the device/adapter; used to plan (MPCController ducted_share)
EXPECTED_SHARE_ATTR = "expected_share"
# Occupied rooms with gain sensors add this much (kW) for the people in them (~1 person)
OCCUPANT_GAIN_KW = 0.1


def delivered_fraction(state: Any) -> float | None:
    """The share of the central unit's output this device's room gets, or None if the device doesn't say."""
    if state is None:
        return None
    value = state.attributes.get(DELIVERED_PF_ATTR)
    try:
        frac = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(frac):
        return None
    return min(1.0, max(0.0, frac))


def expected_share(hass: HomeAssistant, entity_ids) -> float | None:
    """Largest typical share among the room's devices (a room on two zones would sum; we have one each)."""
    best: float | None = None
    for eid in entity_ids:
        state = hass.states.get(eid)
        if state is None:
            continue
        try:
            v = float(state.attributes.get(EXPECTED_SHARE_ATTR))
        except (TypeError, ValueError):
            continue
        if math.isfinite(v) and 0.0 < v <= 1.0:
            best = v if best is None else max(best, v)
    return best


def internal_gain_kw(hass: HomeAssistant, room: dict) -> float | None:
    """Sum of the room's heat-gain power sensors in kW; None when the room has none configured."""
    eids = room.get("heat_gain_sensors") or []
    if not eids:
        return None
    total = 0.0
    for eid in eids:
        state = hass.states.get(eid)
        if state is None or state.state in ("unavailable", "unknown", ""):
            continue
        try:
            value = float(state.state)
        except ValueError:
            continue
        if not math.isfinite(value) or value < 0:
            continue
        unit = (state.attributes.get("unit_of_measurement") or "W").strip()
        total += value if unit.lower() == "kw" else value / 1000.0
    return total
