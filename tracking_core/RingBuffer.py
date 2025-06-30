from dataclasses import dataclass
import os
import pickle
from typing import IO, Any, Dict, List, Optional


@dataclass
class RingBufferEntry:
    id: int
    position: int
    length: int


class RingBuffer:

    _size: int
    _file: IO[bytes]
    _id_map: Dict[int, int]
    _entries: List[Optional[RingBufferEntry]]
    _next: int

    def __init__(self, file_name: str, size: int):
        self._size = size
        self._file = open(file_name, "w+b")
        self._entries = [None] * self._size
        self._id_map = {}
        self._next = 0

    def close(self):
        self._file.close()

    def store(self, id: int, data: Any):
        serialized = pickle.dumps(data)
        length = len(serialized)

        position = -1  # Put at end of file by default

        if self._next >= self._size:
            self._next = 0

        existing = self._entries[self._next]

        # See if we can re-use existing space
        if existing is not None:
            # Remove old from the map
            del self._id_map[existing.id]
            if existing.length >= length:
                position = existing.position

        if position == -1:
            self._file.seek(0, os.SEEK_END)
            position = self._file.tell()

        self._file.seek(position, os.SEEK_SET)
        self._file.write(serialized)
        self._file.flush()

        self._entries[self._next] = RingBufferEntry(id, position, length)
        self._id_map[id] = self._next
        self._next += 1

    def retrieve(self, id: int) -> Any:

        index = self._id_map.get(id)
        if index is None:
            return None

        entry = self._entries[index]
        self._file.seek(entry.position, os.SEEK_SET)
        serialized = self._file.read(entry.length)
        return pickle.loads(serialized)

    def item_exists(self, id: int):
        return id in self._id_map
