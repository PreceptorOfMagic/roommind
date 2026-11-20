"""Tests for the ducted-zone helpers (fork: shared central unit feeding zoned rooms)."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from custom_components.roommind.utils.ducted_utils import (
    coupled_ambient,
    delivered_fraction,
    expected_share,
    internal_gain_kw,
    is_ducted,
    neighbour_temp,
    passively_flowing,
    planning_share,
)


def _state(state="cool", **attrs):
    return SimpleNamespace(state=state, attributes=attrs)


def _hass(states: dict):
    hass = MagicMock()
    hass.states.get.side_effect = states.get
    return hass


class TestDeliveredFraction:
    def test_reads_and_clamps(self):
        assert delivered_fraction(_state(delivered_power_fraction=0.4)) == 0.4
        assert delivered_fraction(_state(delivered_power_fraction=1.7)) == 1.0
        assert delivered_fraction(_state(delivered_power_fraction=-1)) == 0.0

    def test_missing_or_bad(self):
        assert delivered_fraction(None) is None
        assert delivered_fraction(_state()) is None
        assert delivered_fraction(_state(delivered_power_fraction="x")) is None
        assert delivered_fraction(_state(delivered_power_fraction=float("nan"))) is None


class TestShares:
    def test_projected_preferred_over_expected(self):
        hass = _hass({"climate.z": _state(expected_share=0.5, projected_share=0.25)})
        assert planning_share(hass, ["climate.z"]) == 0.25

    def test_falls_back_to_expected(self):
        hass = _hass({"climate.z": _state(expected_share=0.5)})
        assert planning_share(hass, ["climate.z"]) == 0.5
        assert expected_share(hass, ["climate.z"]) == 0.5

    def test_none_without_attributes(self):
        hass = _hass({"climate.z": _state()})
        assert planning_share(hass, ["climate.z"]) is None
        assert planning_share(hass, ["climate.missing"]) is None

    def test_invalid_projection_ignored(self):
        hass = _hass({"climate.z": _state(expected_share=0.3, projected_share=0)})
        assert planning_share(hass, ["climate.z"]) == 0.3


class TestPassiveFlow:
    def test_constant_zone_off_but_cooled(self):
        st = _state("off", hvac_action="cooling", delivered_power_fraction=0.2)
        assert is_ducted(st)
        assert passively_flowing(st)

    def test_off_and_idle(self):
        assert not passively_flowing(_state("off", hvac_action="off", delivered_power_fraction=0.0))
        assert not passively_flowing(_state("off", hvac_action="cooling", delivered_power_fraction=0.0))

    def test_plain_ac_is_not_ducted(self):
        st = _state("cool", hvac_action="cooling")
        assert not is_ducted(st)
        assert not passively_flowing(st)


class TestCoupling:
    def test_blend(self):
        assert coupled_ambient(30.0, 24.0, 0.0) == 30.0
        assert coupled_ambient(30.0, 24.0, 0.5) == pytest.approx(27.0)
        # clamped to 0.95: a room never loses all contact with outdoors
        assert coupled_ambient(30.0, 24.0, 2.0) == pytest.approx(30.0 - 0.95 * 6.0)
        assert coupled_ambient(30.0, 24.0, -1.0) == 30.0

    def test_neighbours_default_all_others(self):
        rooms = {"a": {"current_temp": 20.0}, "b": {"current_temp": 22.0}, "c": {"current_temp": 26.0}}
        assert neighbour_temp(rooms, "a", {}) == pytest.approx(24.0)

    def test_neighbours_listed(self):
        rooms = {"a": {"current_temp": 20.0}, "b": {"current_temp": 22.0}, "c": {"current_temp": 26.0}}
        assert neighbour_temp(rooms, "a", {"coupling_rooms": ["b", "a"]}) == 22.0

    def test_neighbours_missing_temps(self):
        rooms = {"a": {"current_temp": 20.0}, "b": {"current_temp": None}, "c": {}}
        assert neighbour_temp(rooms, "a", {}) is None
        assert neighbour_temp({}, "a", {}) is None


class TestInternalGain:
    def test_sums_w_and_kw(self):
        hass = _hass({
            "sensor.pc": _state("300", unit_of_measurement="W"),
            "sensor.gym": _state("0.5", unit_of_measurement="kW"),
            "sensor.off": _state("unavailable"),
        })
        room = {"heat_gain_sensors": ["sensor.pc", "sensor.gym", "sensor.off", "sensor.missing"]}
        assert internal_gain_kw(hass, room) == pytest.approx(0.8)

    def test_none_without_sensors(self):
        assert internal_gain_kw(_hass({}), {}) is None
