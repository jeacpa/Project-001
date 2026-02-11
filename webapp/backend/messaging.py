
from dataclasses import dataclass
from multiprocessing.connection import Client
from telnetlib import IP
from typing import List, Optional, Tuple

from pydantic import BaseModel, Field

from constants import COUNT_ZONE, IPC_ADDRESS

class ControlRequest(BaseModel):
    action: str
    x: Optional[int] = None
    y: Optional[int] = None
    count_zone: Optional[List[List[int]]] = Field(alias="countZone", default=None)

    class Config:
        allow_population_by_field_name = True
        populate_by_name = True
        by_alias = True


class ControlState(BaseModel):
    show_zones: bool = Field(alias="showZones", default=False)
    show_light: bool = Field(alias="showLight", default=False)
    show_text: bool = Field(alias="showText", default=False)
    show_boxes: bool = Field(alias="showBoxes", default=False)
    paused: bool = Field(default=False)
    count_zone: List[Tuple[int, int]] = Field(alias="countZone", default=COUNT_ZONE)

    class Config:
        allow_population_by_field_name = True
        populate_by_name = True
        by_alias = True


class ControlResponse(BaseModel):
    state: ControlState

    class Config:
        allow_population_by_field_name = True


control_state = ControlState(
    show_zones=True,
    show_boxes=True,
    show_light=True,
    show_text=True,
    count_zone=COUNT_ZONE,
)


class MessageingClient:
    _address: str

    def __init__(self, address: str = IPC_ADDRESS):
        self._address = address
        
    def send_message(self, msg: ControlRequest) -> ControlResponse:
        try:
            with Client(self._address) as conn:
                conn.send(msg)
                return conn.recv()
        except Exception as e:
            print("Messaging client error:", repr(e), "type:", type(e))

            # For now, any error just returns the current state
            return ControlResponse(state=control_state)