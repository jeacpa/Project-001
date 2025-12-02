from operator import is_
import struct
import time
from typing import Optional, Tuple
import numpy as np
from multiprocessing import shared_memory
from multiprocessing.resource_tracker import unregister, _CLEANUP_FUNCS

# Original (pre-JPEG) frame shape
HEIGHT = 1080
WIDTH = 1920
CHANNELS = 3
DTYPE = np.uint8

FRAME_SHAPE = (HEIGHT, WIDTH, CHANNELS)

# Maximum allowed JPEG size (bytes)
# Adjust upward if your real frames + quality are larger.
MAX_JPEG_BYTES = 1024 * 1024 * 2 # 2 MB

# Shared memory segment names (must match across producer/consumers)
DATA_NAME = "latest_jpeg_data"    # raw JPEG bytes buffer
LENGTH_NAME = "latest_jpeg_len"   # int32: current JPEG length
SEQ_NAME = "latest_jpeg_seq"      # int64: seqlock sequence
CONSUMER_TS_NAME = "latest_jpeg_consumer_ts"  # int64: last consumed timestamp

class MemoryStreamer:
    _frame_counter: int
    _data_shm: shared_memory.SharedMemory
    _length_shm: shared_memory.SharedMemory
    _seq_shm: shared_memory.SharedMemory
    _consumer_ts_shm: shared_memory.SharedMemory
    _created: bool
    _length_view: np.ndarray
    _seq_view: np.ndarray
    _last_seen_seq: int
    _is_producer: bool

    def __init__(self, is_producer: Optional[bool] = False):
        """
        Create shared memory segments if they don't exist, otherwise attach to them.

        Returns:
            data_shm, length_shm, seq_shm, created (bool)
        """
        self._frame_counter = 0
        self._last_seen_seq = 0
        self._is_producer = is_producer

        self._created = False
        if is_producer:
            self._init_producer()
        else:
            self._init_consumer()

    def _init_producer(self) -> None:
        if self._created:
            return

        int32_size = np.dtype(np.int32).itemsize
        int64_size = np.dtype(np.int64).itemsize

        self._data_shm = shared_memory.SharedMemory(
            name=DATA_NAME,
            create=True,
            size=MAX_JPEG_BYTES,
        )
        self._length_shm = shared_memory.SharedMemory(
            name=LENGTH_NAME,
            create=True,
            size=int32_size,
        )
        self._seq_shm = shared_memory.SharedMemory(
            name=SEQ_NAME,
            create=True,
            size=int64_size,
        )
        self._consumer_ts_shm = shared_memory.SharedMemory(
            name=CONSUMER_TS_NAME,
            create=True,
            size=8, # float as 64
        )

        self._length_view = np.ndarray((), dtype=np.int32, buffer=self._length_shm.buf)
        self._seq_view = np.ndarray((), dtype=np.int64, buffer=self._seq_shm.buf)

        self._length_view[...] = 0   # no JPEG yet
        self._seq_view[...] = 0      # even => no write in progress

        self._created = True

    def _init_consumer(self) -> bool:
        if self._created:
            return True
        
        try:
            self._data_shm = shared_memory.SharedMemory(name=DATA_NAME, create=False)
            self._length_shm = shared_memory.SharedMemory(name=LENGTH_NAME, create=False)
            self._seq_shm = shared_memory.SharedMemory(name=SEQ_NAME, create=False)
            self._consumer_ts_shm = shared_memory.SharedMemory(name=CONSUMER_TS_NAME, create=False)
        except FileNotFoundError:
            return False

        # Prevent resource tracker from trying to unlink shared memory segments for consumers
        unregister(self._data_shm._name, "shared_memory")
        unregister(self._length_shm._name, "shared_memory")
        unregister(self._seq_shm._name, "shared_memory")
        unregister(self._consumer_ts_shm._name, "shared_memory")

        self._length_view = np.ndarray((), dtype=np.int32, buffer=self._length_shm.buf)
        self._seq_view = np.ndarray((), dtype=np.int64, buffer=self._seq_shm.buf)

        self._created = True

    def close(self) -> None:

        if not self._created:
            return
        
        # If we producer mark the seq as -1 to signal offline to consumers
        if self._is_producer:
            self._seq_view[...] = -1

        del self._length_view
        del self._seq_view

        self._data_shm.close()
        self._length_shm.close()
        self._seq_shm.close()
        self._consumer_ts_shm.close()

        if self._is_producer:
            self._data_shm.unlink()
            self._length_shm.unlink()
            self._seq_shm.unlink()
            self._consumer_ts_shm.unlink()

        self._data_shm = None
        self._length_shm = None
        self._seq_shm = None
        self._consumer_ts_shm = None

        self._created = False

    @property
    def is_first_producer(self) -> bool:
        return self._created
    
    def consumer_timestamp(self) -> float:
        if not self._created:
            return 0.0
        raw: bytes = bytes(self._consumer_ts_shm.buf[:8])  # copy 8 bytes
        ts: float = struct.unpack("d", raw)[0]

        return ts

    def _write_consumer_timestamp(self) -> None:
        if not self._created:
            return
        raw: bytes = struct.pack("d", time.time())
        self._consumer_ts_shm.buf[:8] = raw

    def produce_frame(self, jpeg_bytes: bytes) -> None:
        
        if not self._is_producer:
            raise RuntimeError("Cannot produce frame from a consumer MemoryStreamer")
        
        jpeg_len = len(jpeg_bytes)

        if jpeg_len > MAX_JPEG_BYTES:
            print(
                f"[PRODUCER] JPEG too large ({jpeg_len} > {MAX_JPEG_BYTES}), dropping."
            )
            return

        buf = self._data_shm.buf

        # ---- Seqlock write: mark write in progress (odd seq) ----
        # seq is always even when no write in progress.
        # We increment by 1 to make it odd (writer active),
        # write data, then increment again to make it even (publish).
        self._seq_view[...] += 1     # now odd => readers will wait/retry

        # ---- Write JPEG bytes + length atomically from reader's POV ----
        buf[:jpeg_len] = jpeg_bytes
        self._length_view[...] = jpeg_len

        # ---- Publish: even seq means a complete frame is visible ----
        self._seq_view[...] += 1     # now even => readers can safely read

    @property
    def is_connected(self) -> bool:
        if not self._created:
            return False
        
        seq0 = int(self._seq_view[...])
        return seq0 != -1
    
    def consume_frame(self) -> Optional[bytes]:

        if not self._init_consumer():
            return None
        
        self._write_consumer_timestamp()
        
        buf = self._data_shm.buf

        # ---- Cheap early check: is there a NEW frame at all? ----
        seq0 = int(self._seq_view[...])

        # Check if producer has gone offline
        if seq0 == -1:
            # Producer offline, close down and retry to connect in future
            self.close()
            return None
        
        if seq0 == 0:
            # No frame yet
            return None

        # If seq is odd, writer is in progress
        if seq0 & 1:
            return None

        # If seq hasn't changed since last time, don't bother reading/decoding again
        if seq0 <= self._last_seen_seq:
            return None


        jpeg_len = int(self._length_view[...])
        if jpeg_len <= 0 or jpeg_len > MAX_JPEG_BYTES:
            print(
                f"[CONSUMER] JPEG too large ({jpeg_len} > {MAX_JPEG_BYTES}), dropping."
            )
            return None

        # Copy JPEG bytes into a local immutable buffer
        jpeg_bytes = bytes(buf[:jpeg_len])

        seq1 = int(self._seq_view[...])

        if seq0 == seq1 and not (seq1 & 1):
            # Consistent snapshot of a NEW frame
            self._last_seen_seq = seq1
            return jpeg_bytes
