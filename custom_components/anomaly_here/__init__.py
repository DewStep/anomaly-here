"""
Custom integration to integrate anomaly_here with Home Assistant.

For more details about this integration, please refer to
https://github.com/DewStep/anomaly-here
"""

from __future__ import annotations

import datetime
import sqlite3
import time
from typing import TYPE_CHECKING

import pandas as pd
import sqlalchemy
from homeassistant.components.persistent_notification import create
from homeassistant.const import CONF_PASSWORD, CONF_USERNAME, Platform
from homeassistant.helpers import entity_registry as er
from homeassistant.helpers.aiohttp_client import async_get_clientsession
from homeassistant.helpers.event import (
    async_call_later,
    async_track_point_in_time,
    async_track_state_change_event,
    async_track_time_interval,
)
from homeassistant.loader import async_get_loaded_integration
from sqlalchemy import Column, Integer, Nullable, String, orm

from .analysis import build_activations, estimate_hold_times, merge_episodes, run_merge
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

Base = orm.declarative_base()


class Episodes_db(Base):
    __tablename__ = "episodes"  # Table name

    id = Column(Integer, primary_key=True, autoincrement=True)
    scope_type = Column(String, nullable=False)
    scope_value = Column(String, nullable=False)
    start_ts = Column(Integer, nullable=False)
    end_ts = Column(Integer, nullable=False)
    event_count = Column(Integer, nullable=False)
    active_duration_s = Column(Integer, nullable=False)
    created_at = Column(Integer, nullable=False)


class EventsDB(Base):
    __tablename__ = "events"

    id = Column(Integer, primary_key=True, autoincrement=True)
    unix_time = Column(Integer, nullable=False)
    sensor = Column(String, nullable=False)
    value = Column(String, nullable=False)


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

        engine = sqlalchemy.create_engine("sqlite:///activity_log.db")
        Base.metadata.create_all(engine)
        self.session = orm.Session(bind=engine)

        # self.activity_log = sqlite3.connect("activity_log.db", isolation_level=None)
        # self.activity_log.execute(
        #    "CREATE TABLE IF NOT EXISTS activity (timestamp TEXT,event TEXT)"
        # )
        # self.activity_log.execute("DELETE FROM activity WHERE 1")
        # create(self.hass, ("Test call. Database created."))
        current_date = str(datetime.datetime.now(tz=datetime.UTC).date())
        evening_datetime = datetime.datetime(
            int(current_date[:4]),
            int(current_date[5:7]),
            int(current_date[8:]),
            11,
            40,
            00,
            tzinfo=datetime.UTC,
        )
        create(
            self.hass,
            ("Test call. current time." + str(datetime.datetime.now(tz=datetime.UTC))),
        )
        create(self.hass, ("Test call. analyse time." + str(evening_datetime)))
        self.analyse_start = async_track_point_in_time(
            self.hass,
            self.analysis_start,
            evening_datetime,
        )

    async def activity_noticed(self, _event: Event[EventStateChangedData]) -> None:
        """
        Restart inactivity timer.

        Parameters
        ----------
        event : Event[EventStateChangedData]
            The event that triggered the callback.

        """
        self.restart_check()
        new_state = _event.data["new_state"]
        if new_state is not None:
            write_time = int(new_state.last_changed.timestamp())
            new_event = EventsDB(
                unix_time=write_time,
                sensor=_event.data["entity_id"],
                value=new_state.state,
            )
            self.session.add(new_event)
            self.session.commit()
        # change this once the code to figure it out is written
        self.restart_check = async_call_later(
            self.hass, datetime.timedelta(minutes=5), self.alert_call
        )

    async def create_listeners(self) -> None:
        """Create listeners for all binary sensors in self.sensors."""
        self.EndList = []
        for entity_class in self.sensors:
            try:
                for entity in self.sensors[entity_class][1]:
                    end_listener = async_track_state_change_event(
                        self.hass, entity, self.activity_noticed
                    )
                    self.EndList.append(end_listener)
            except IndexError:
                create(
                    self.hass, ("Test call. No listeners created for " + entity_class)
                )
        self.restart_check = async_call_later(
            self.hass, datetime.timedelta(minutes=5), self.alert_call
        )

    async def analysis_start(self, _now: datetime.datetime):
        create(self.hass, ("Test call. Analysis cycle started"))
        event_data = self.session.query(EventsDB).all()
        events_list = []
        for event in event_data:
            if type(event.unix_time) is int:
                row_data = {
                    "time": datetime.datetime.fromtimestamp(
                        event.unix_time, tz=datetime.UTC
                    ),
                    "sensor": event.sensor,
                    "value": event.value,
                }
                events_list.append(row_data)
            else:
                create(self.hass, ("Can't use event like that"))
        self.events_full = pd.DataFrame(events_list)
        create(self.hass, "self.events_full: " + str(type(self.events_full)))
        try:
            thresholds = await self.hass.async_add_executor_job(
                run_merge, self.events_full, self.hass
            )
            create(self.hass, ("Test call. episodes of type " + str(type(thresholds))))
        except Exception as e:
            create(self.hass, ("Test call. Error in analysis_start: " + str(e)))
            return

        event_data = self.session.query(EventsDB).all()
        for event in event_data:
            self.session.delete(event)
        self.session.commit()
        # Test run
        form = "%Y-%m-%d %H:%M:%S"
        merge_thresholds = thresholds.loc[:, ["entity", "final_G_s"]]
        rows, _cols = merge_thresholds.shape
        holds = estimate_hold_times(self.events_full)
        for entity_info in range(rows):
            ind_thresh = merge_thresholds.iloc[entity_info]
            if ind_thresh.iloc[0] != "HOUSE":
                entity = ind_thresh.iloc[0]
                s_type = "entity"
                s_value = entity
            else:
                entity = None
                s_type = "house"
                s_value = "house"
            activations = build_activations(self.events_full, holds, entity)
            episodes = merge_episodes(activations, ind_thresh.iloc[1])
            ep_rows, _ep_cols = episodes.shape
            for i in range(ep_rows):
                episode_date = str(episodes.at[i, "start"])[:19]
                epoch_start = int(
                    datetime.datetime.strptime(episode_date, form).timestamp()
                )
                episode_date = str(episodes.at[i, "end"])[:19]
                epoch_end = int(
                    datetime.datetime.strptime(episode_date, form).timestamp()
                )
                new_episode = Episodes_db(
                    scope_type=s_type,
                    scope_value=s_value,
                    start_ts=epoch_start,
                    end_ts=epoch_end,
                    event_count=2,
                    active_duration_s=2,
                    created_at=int(time.time()),
                )
                self.session.add(new_episode)
        self.session.commit()
        data = {"time": [], "sensor": [], "value": []}
        self.events_full = pd.DataFrame(data)
        create(self.hass, ("Test call. Analysis cycle completed"))
        episode_data = self.session.query(Episodes_db).all()
        run_number = 0
        for episode in episode_data:
            run_number += 1
            create(
                self.hass,
                (
                    "Test call. Episode: "
                    + str(episode.scope_type)
                    + " "
                    + str(episode.scope_value)
                    + " "
                    + str(episode.start_ts)
                    + " "
                    + str(episode.end_ts)
                ),
            )
            if run_number > 10:
                break

        create(self.hass, ("Test call. "))
        self.daily_analysis = async_track_time_interval(
            self.hass, self.event_analysis, datetime.timedelta(days=1)
        )

    async def event_analysis(self, _now: datetime.datetime):
        create(self.hass, ("Test call. Daily analysis started"))

    async def alert_call(self, _now: datetime.datetime) -> None:
        """Send a persistent notification to Home Assistant."""
        create(self.hass, "Inactivity detected")

    async def async_shutdown(self) -> None:
        """Remove all listeners and timer so that the integration can be unloaded."""
        for listener in self.EndList:
            listener()  # Call the listener to remove it
        self.restart_check()  # Cancel the scheduled alert call
        self.daily_analysis()  # Cancel the scheduled daily analysis
        self.analyse_start()  # Cancel the scheduled analysis start
        episode_data = self.session.query(Episodes_db).all()
        for episode in episode_data:
            self.session.delete(episode)
        self.session.commit()
