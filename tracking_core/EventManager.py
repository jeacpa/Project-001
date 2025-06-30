from abc import ABC, abstractmethod
from typing import Any, List

from Clients.sql import SqlClient
from tracking_core.SimpleCounter import Dict
from constants import INTERSECTION_DIRECTION, INTERSECTION_ID
from tracking_core.structures import TrackingEvent
from util import utc_now


class EventManager(ABC):

    @abstractmethod
    def write_batch_events(self, events: List[TrackingEvent]) -> None:
        pass

    def write_event(self, name: str, attributes: Dict[str, Any]) -> None:
        self.write_batch_events([TrackingEvent(name=name, attributes=attributes)])

    def default_event_attributes(self):
        return {
            "intersection_id": INTERSECTION_ID,
            "direction": INTERSECTION_DIRECTION.value,
        }


class NullEventManager(EventManager):
    """
    An event manager that does nothing.
    """

    def write_batch_events(self, events: List[TrackingEvent]) -> None:
        pass


class SqlEventManager(EventManager):
    """
    An event manager that writes events to a SQL database.
    """

    _sql_client: SqlClient

    def __init__(self, sql: SqlClient):
        self._sql_client = sql

    def write_batch_events(self, events: List[TrackingEvent]) -> None:

        rows = []
        for te in events:
            attributes = self._default_event_attributes()
            attributes.update(te.attributes)

            rows.append((utc_now(), te.name, attributes))

        self._sql_client.insert_batch(
            "tbl_event",
            ["occurred_at", "name", "attributes"],
            rows,
        )

        for row in rows:
            print(f"[Event] {row[0]} - {row[1]}: {row[2]}")
