"""
Custom integration to integrate anomaly_here with Home Assistant.

For more details about this integration, please refer to
https://github.com/DewStep/anomaly-here
"""

from __future__ import annotations

from datetime import timedelta
from typing import TYPE_CHECKING

from homeassistant.components.persistent_notification import create
from homeassistant.const import CONF_PASSWORD, CONF_USERNAME, Platform
from homeassistant.core import Event, EventStateChangedData
from homeassistant.helpers import entity_registry as er
from homeassistant.helpers.aiohttp_client import async_get_clientsession
from homeassistant.helpers.event import (
    async_call_later,
    async_track_state_change_event,
)
from homeassistant.loader import async_get_loaded_integration

from .api import AnomalyHereApiClient
from .const import DOMAIN, LOGGER
from .coordinator import AnomalyHereDataUpdateCoordinator
from .data import AnomalyHereData

if TYPE_CHECKING:
    from homeassistant.core import HomeAssistant

    from .data import AnomalyHereConfigEntry

PLATFORMS: list[Platform] = [
    Platform.SENSOR,
    Platform.BINARY_SENSOR,
    Platform.SWITCH,
]


# https://developers.home-assistant.io/docs/config_entries_index/#setting-up-an-entry
async def async_setup_entry(
    hass: HomeAssistant,
    entry: AnomalyHereConfigEntry,
) -> bool:
    """Set up this integration using UI."""
    coordinator = AnomalyHereDataUpdateCoordinator(
        hass=hass,
        logger=LOGGER,
        name=DOMAIN,
        update_interval=timedelta(hours=1),
    )
    entry.runtime_data = AnomalyHereData(
        client=AnomalyHereApiClient(
            username=entry.data[CONF_USERNAME],
            password=entry.data[CONF_PASSWORD],
            session=async_get_clientsession(hass),
        ),
        integration=async_get_loaded_integration(hass, entry.domain),
        coordinator=coordinator,
    )

    # Apparently ZHA creates sensors with unique IDs,
    # which is needed for the entity registry to find it.
    entity_registry = er.async_get(hass)
    # creates a dictionary with device classes as the keys and
    # entity IDs as the values.
    entity_list = {}
    for entity in entity_registry.entities.values():
        if entity.domain == "binary_sensor":
            device_class = entity.device_class
            if device_class in entity_list:
                entity_list[device_class].append(entity.entity_id)
            else:
                entity_list[device_class] = entity.entity_id
    # Filters through only certain sensor types,
    # adds the delay of how long they stay active.
    # It's ones where I know how long they stay active after detection
    # ADD MORE FUTURE ME
    known_sensors = {"test": [0, ["test_door_sensor"]], "occupancy": [50], "door": [0]}
    for entity in entity_list:
        if entity == "occupancy":
            known_sensors["occupancy"].append(entity_list[entity])
        elif entity == "door":
            known_sensors["door"].append(entity_list[entity])

    Detector = AnomalyDetector(hass, known_sensors)
    await Detector.async_setup()
    hass.data.setdefault(DOMAIN, {})
    hass.data[DOMAIN][entry.entry_id] = Detector

    # https://developers.home-assistant.io/docs/integration_fetching_data#coordinated-single-api-poll-for-data-for-all-entities
    await coordinator.async_config_entry_first_refresh()

    await hass.config_entries.async_forward_entry_setups(entry, PLATFORMS)
    entry.async_on_unload(entry.add_update_listener(async_reload_entry))

    return True


async def async_unload_entry(
    hass: HomeAssistant,
    entry: AnomalyHereConfigEntry,
) -> bool:
    """Handle removal of an entry."""
    Detector = hass.data[DOMAIN][entry.entry_id]
    await Detector.async_shutdown()
    return await hass.config_entries.async_unload_platforms(entry, PLATFORMS)


async def async_reload_entry(
    hass: HomeAssistant,
    entry: AnomalyHereConfigEntry,
) -> None:
    """Reload config entry."""
    await hass.config_entries.async_reload(entry.entry_id)


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

    async def async_setup(self) -> None:
        self.EndList = []
        for type in self.sensors:
            for entity in self.sensors[type][1]:
                EndListener = async_track_state_change_event(
                    self.hass, entity, self.activity_noticed
                )
                create(self.hass, ("Test call. Listener Created for " + entity))
                self.EndList.append(EndListener)
        self.restart_check = async_call_later(
            self.hass, timedelta(minutes=5), self.alert_call
        )

    async def alert_call(self, _now) -> None:
        create(self.hass, "Inactivity detected")

    async def async_shutdown(self) -> None:
        for listener in self.EndList:
            listener()  # Call the listener to remove it
        self.restart_check()  # Cancel the scheduled alert call
