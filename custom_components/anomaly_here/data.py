"""Custom types for anomaly_here."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from homeassistant.config_entries import ConfigEntry
    from homeassistant.loader import Integration

    from .api import AnomalyHereApiClient
    from .coordinator import AnomalyHereDataUpdateCoordinator


type AnomalyHereConfigEntry = ConfigEntry[AnomalyHereData]


@dataclass
class AnomalyHereData:
    """Data for the Anomaly Here integration."""

    client: AnomalyHereApiClient
    coordinator: AnomalyHereDataUpdateCoordinator
    integration: Integration
