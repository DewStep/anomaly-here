"""Sensor platform for anomaly_here."""

from __future__ import annotations

from datetime import timedelta
from typing import TYPE_CHECKING

from homeassistant.components.persistent_notification import create
from homeassistant.components.sensor import SensorEntity, SensorEntityDescription
from homeassistant.core import Event, EventStateChangedData
from homeassistant.helpers.event import (
    async_call_later,
    async_track_state_change_event,
)

from .entity import AnomalyHereEntity

if TYPE_CHECKING:
    from homeassistant.core import HomeAssistant
    from homeassistant.helpers.entity_platform import AddEntitiesCallback

    from .coordinator import AnomalyHereDataUpdateCoordinator
    from .data import AnomalyHereConfigEntry

ENTITY_DESCRIPTIONS = (
    SensorEntityDescription(
        key="anomaly_here",
        name="Integration Sensor",
        icon="mdi:format-quote-close",
    ),
)

CONF_TARGET_SENSOR = "target_sensor"


async def async_setup_entry(
    hass: HomeAssistant,  # noqa: ARG001 Unused function argument: `hass`
    entry: AnomalyHereConfigEntry,
    async_add_entities: AddEntitiesCallback,
) -> None:
    """Set up the sensor platform."""
    async_add_entities(
        AnomalyHereSensor(
            coordinator=entry.runtime_data.coordinator,
            entity_description=entity_description,
        )
        for entity_description in ENTITY_DESCRIPTIONS
    )


def setup_platform(hass, config, discovery_info=None):
    sensor = AnomalyDetector(hass, target_sensor=config.get(CONF_TARGET_SENSOR, []))


class AnomalyDetector:
    def __init__(self, hass, target_sensor) -> None:
        self.hass = hass
        self.sensors = target_sensor

    async def activity_noticed(self, event: Event[EventStateChangedData]) -> None:
        self.restart_check()
        # change this once the code to figure it out is written
        self.restart_check = async_call_later(
            self.hass, timedelta(minutes=5), self.alert_call
        )

    async def async_added_to_hass(self) -> None:
        async_track_state_change_event(self.hass, self.sensors, self.activity_noticed)
        self.restart_check = async_call_later(
            self.hass, timedelta(minutes=5), self.alert_call
        )

    async def alert_call(self, _now) -> None:
        create(self.hass, "Inactivity detected")


class AnomalyHereSensor(AnomalyHereEntity, SensorEntity):
    """anomaly_here Sensor class."""

    def __init__(
        self,
        coordinator: AnomalyHereDataUpdateCoordinator,
        entity_description: SensorEntityDescription,
    ) -> None:
        """Initialize the sensor class."""
        super().__init__(coordinator)
        self.entity_description = entity_description

    @property
    def native_value(self) -> str | None:
        """Return the native value of the sensor."""
        return self.coordinator.data.get("body")
