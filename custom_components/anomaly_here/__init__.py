"""
Custom integration to integrate anomaly_here with Home Assistant.

For more details about this integration, please refer to
https://github.com/DewStep/anomaly-here
"""

from __future__ import annotations

import datetime
import sqlite3
from typing import TYPE_CHECKING

from homeassistant.components.persistent_notification import create
from homeassistant.const import CONF_PASSWORD, CONF_USERNAME, Platform
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
    from homeassistant.core import Event, EventStateChangedData, HomeAssistant

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
        update_interval=datetime.timedelta(hours=1),
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
                entity_list[device_class] = [entity.entity_id]
    # Filters through only certain sensor types,
    # adds the delay of how long they stay active.
    # It's ones where I know how long they stay active after detection
    # ADD MORE FUTURE ME
    known_sensors = {
        "occupancy": [50],
        "door": [0],
    }
    for entity_type, entities in entity_list.items():
        if entity_type in known_sensors:
            known_sensors[entity_type].append(entities)
    detector = AnomalyDetector(hass, known_sensors)
    await detector.async_setup()
    hass.data.setdefault(DOMAIN, {})
    hass.data[DOMAIN][entry.entry_id] = detector

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
    detector = hass.data[DOMAIN][entry.entry_id]
    await detector.async_shutdown()
    return await hass.config_entries.async_unload_platforms(entry, PLATFORMS)


async def async_reload_entry(
    hass: HomeAssistant,
    entry: AnomalyHereConfigEntry,
) -> None:
    """Reload config entry."""
    await hass.config_entries.async_reload(entry.entry_id)


class AnomalyDetector:
    """
    A collection of listeners for binary sensors.

    Attributes
    ----------
    hass : HomeAssistant
        The Home Assistant instance.
    sensors : dict
        A dictionary of lists of binary sensors of a given type.

    Methods
    -------
    async_setup() -> None
        Setup the AnomalyDetector and creates a database of events
    activity_noticed(event: Event[EventStateChangedData]) -> None
        Called when a binary sensor changes state. Restarts the inactivity timer.
    create_listeners() -> None
        Creates an async event state change listener
        for all binary sensors in self.sensors.
    alert_call(_now) -> None
        Called when the inactivity delay passes without being restarted.
        Creates a persistent notification in Home Assistant.
    async_shutdown() -> None
        removes all listeners and cancels the timer

    """

    def __init__(self, hass: HomeAssistant, target_sensor: dict) -> None:
        """Initialise AnomalyDetector with Home Assistant and binary sensors."""
        self.hass = hass
        self.sensors = target_sensor

    async def async_setup(self) -> None:
        """Set up the AnomalyDetector and database."""
        await self.create_listeners()
        self.activity_log = sqlite3.connect("activity_log.db", isolation_level=None)
        self.activity_log.execute(
            "CREATE TABLE IF NOT EXISTS activity (timestamp TEXT,event TEXT)"
        )
        self.activity_log.execute("DELETE FROM activity WHERE 1")
        create(self.hass, ("Test call. Database created."))

    async def activity_noticed(self, _event: Event[EventStateChangedData]) -> None:
        """
        Restart inactivity timer.

        Parameters
        ----------
        event : Event[EventStateChangedData]
            The event that triggered the callback.

        """
        self.restart_check()
        create(
            self.hass,
            ("Test call. Event noticed, entity_id: " + str(_event.data["entity_id"])),
        )
        create(
            self.hass,
            (
                "Test call. Event noticed, entity_id p2: "
                + str(type(_event.data["entity_id"]))
            ),
        )
        create(
            self.hass,
            ("Test call. Event noticed, old_state: " + str(_event.data["old_state"])),
        )
        create(
            self.hass,
            (
                "Test call. Event noticed, old_state p2: "
                + str(type(_event.data["old_state"]))
            ),
        )
        create(
            self.hass,
            ("Test call. Event noticed, new_state: " + str(_event.data["new_state"])),
        )
        create(
            self.hass,
            (
                "Test call. Event noticed, new_state p2: "
                + str(type(_event.data["new_state"]))
            ),
        )
        create(
            self.hass,
            ("Test call. Current time: " + str(datetime.datetime.now(tz=datetime.UTC))),
        )
        # self.activity_log.execute(
        #    "INSERT INTO activity VALUES (?, ?)",
        #    [str(datetime.datetime.now(tz=datetime.UTC)), str(cleaned_event_data)],
        # )
        create(self.hass, ("Test call. Event logged to database."))
        current_db = self.activity_log.execute("SELECT * FROM activity").fetchall()
        create(self.hass, (str(current_db)))
        # change this once the code to figure it out is written
        self.restart_check = async_call_later(
            self.hass, datetime.timedelta(minutes=5), self.alert_call
        )

    async def create_listeners(self) -> None:
        """Create listeners for all binary sensors in self.sensors."""
        self.EndList = []
        create(self.hass, ("Test call. Setup started."))
        for entity_class in self.sensors:
            try:
                for entity in self.sensors[entity_class][1]:
                    end_listener = async_track_state_change_event(
                        self.hass, entity, self.activity_noticed
                    )
                    create(self.hass, ("Test call. Listener Created for " + entity))
                    self.EndList.append(end_listener)
            except IndexError:
                create(
                    self.hass, ("Test call. No listeners created for " + entity_class)
                )
        self.restart_check = async_call_later(
            self.hass, datetime.timedelta(minutes=5), self.alert_call
        )
        create(self.hass, ("Test call. All listeners created"))

    async def alert_call(self, _now: datetime.datetime) -> None:
        """Send a persistent notification to Home Assistant."""
        create(self.hass, "Inactivity detected")

    async def async_shutdown(self) -> None:
        """Remove all listeners and timer so that the integration can be unloaded."""
        for listener in self.EndList:
            listener()  # Call the listener to remove it
        self.restart_check()  # Cancel the scheduled alert call
