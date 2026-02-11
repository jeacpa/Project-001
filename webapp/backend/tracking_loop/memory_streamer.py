from operator import is_
import struct
import time
from typing import Optional, Tuple, List
import numpy as np
from multiprocessing import shared_memory
from multiprocessing.resource_tracker import unregister, _CLEANUP_FUNCS


# Maximum allowed JPEG size (bytes)
# Adjust upward if your real frames + quality are larger.
MAX_JPEG_BYTES = 1024 * 1024 * 2 # 2 MB

# Number of buffers to cycle through.
# Use NUM_BUFFERS=2 for double buffering, 3+ for more buffering.
# Higher values reduce frame loss but use more memory.
NUM_BUFFERS = 3

def _make_shm_names(buffer_idx: int) -> Tuple[str, str]:
    """Generate shared memory names for a given buffer index."""
    return (
        f"latest_jpeg_data_{buffer_idx}",
        f"latest_jpeg_len_{buffer_idx}",
    )

# Shared memory segment names (must match across producer/consumers)
SEQ_NAME = "latest_jpeg_seq"              # int64: seqlock sequence
WRITE_BUF_IDX_NAME = "write_buf_idx"      # int32: which buffer the producer is currently writing to
CONSUMER_TS_NAME = "latest_jpeg_consumer_ts"  # float64: last consumed timestamp

class MemoryStreamer:
    _frame_counter: int
    _data_shms: List[shared_memory.SharedMemory]
    _length_shms: List[shared_memory.SharedMemory]
    _seq_shm: shared_memory.SharedMemory
    _write_buf_idx_shm: shared_memory.SharedMemory
    _consumer_ts_shm: shared_memory.SharedMemory
    _created: bool
    _length_views: List[np.ndarray]
    _seq_view: np.ndarray
    _write_buf_idx_view: np.ndarray
    _last_seen_seq: int
    _is_producer: bool
    _last_read_buf_idx: int  # Track which buffer we last read to detect consumer lag
    _producer_frame_count: int  # For FPS calculation on producer side
    _producer_last_report_ts: float  # Last time producer reported stats

    def __init__(self, is_producer: Optional[bool] = False):
        """
        Create shared memory segments if they don't exist, otherwise attach to them.

        Args:
            is_producer: True if this instance will produce frames, False if consuming.
        """
        self._frame_counter = 0
        self._last_seen_seq = 0
        self._is_producer = is_producer
        self._last_read_buf_idx = -1  # Track last buffer read by consumer
        self._producer_frame_count = 0
        self._producer_last_report_ts = time.time()

        self._data_shms = []
        self._length_shms = []
        self._length_views = []

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

        # Create NUM_BUFFERS JPEG data buffers
        for i in range(NUM_BUFFERS):
            data_name, len_name = _make_shm_names(i)
            
            data_shm = shared_memory.SharedMemory(
                name=data_name,
                create=True,
                size=MAX_JPEG_BYTES,
            )
            length_shm = shared_memory.SharedMemory(
                name=len_name,
                create=True,
                size=int32_size,
            )
            
            self._data_shms.append(data_shm)
            self._length_shms.append(length_shm)
            
            length_view = np.ndarray((), dtype=np.int32, buffer=length_shm.buf)
            length_view[...] = 0
            self._length_views.append(length_view)

        self._seq_shm = shared_memory.SharedMemory(
            name=SEQ_NAME,
            create=True,
            size=int64_size,
        )
        self._write_buf_idx_shm = shared_memory.SharedMemory(
            name=WRITE_BUF_IDX_NAME,
            create=True,
            size=int32_size,
        )
        self._consumer_ts_shm = shared_memory.SharedMemory(
            name=CONSUMER_TS_NAME,
            create=True,
            size=8,  # float64
        )

        self._seq_view = np.ndarray((), dtype=np.int64, buffer=self._seq_shm.buf)
        self._write_buf_idx_view = np.ndarray((), dtype=np.int32, buffer=self._write_buf_idx_shm.buf)

        self._seq_view[...] = 0      # even => no write in progress
        self._write_buf_idx_view[...] = 0  # start writing to buffer 0

        self._created = True

    def _init_consumer(self) -> bool:
        if self._created:
            return True
        
        try:
            # Attach to all buffer segments
            for i in range(NUM_BUFFERS):
                data_name, len_name = _make_shm_names(i)
                
                data_shm = shared_memory.SharedMemory(name=data_name, create=False)
                length_shm = shared_memory.SharedMemory(name=len_name, create=False)
                
                self._data_shms.append(data_shm)
                self._length_shms.append(length_shm)
                
                unregister(data_shm._name, "shared_memory")
                unregister(length_shm._name, "shared_memory")
                
                length_view = np.ndarray((), dtype=np.int32, buffer=length_shm.buf)
                self._length_views.append(length_view)

            self._seq_shm = shared_memory.SharedMemory(name=SEQ_NAME, create=False)
            self._write_buf_idx_shm = shared_memory.SharedMemory(name=WRITE_BUF_IDX_NAME, create=False)
            self._consumer_ts_shm = shared_memory.SharedMemory(name=CONSUMER_TS_NAME, create=False)
        except FileNotFoundError:
            return False

        # Prevent resource tracker from trying to unlink shared memory segments for consumers
        unregister(self._seq_shm._name, "shared_memory")
        unregister(self._write_buf_idx_shm._name, "shared_memory")
        unregister(self._consumer_ts_shm._name, "shared_memory")

        self._seq_view = np.ndarray((), dtype=np.int64, buffer=self._seq_shm.buf)
        self._write_buf_idx_view = np.ndarray((), dtype=np.int32, buffer=self._write_buf_idx_shm.buf)

        self._created = True
        return True

    def close(self) -> None:

        if not self._created:
            return
        
        # If we are producer, mark the seq as -1 to signal offline to consumers
        if self._is_producer:
            self._seq_view[...] = -1

        del self._length_views
        del self._seq_view
        del self._write_buf_idx_view

        for data_shm in self._data_shms:
            data_shm.close()
        for length_shm in self._length_shms:
            length_shm.close()
        self._seq_shm.close()
        self._write_buf_idx_shm.close()
        self._consumer_ts_shm.close()

        if self._is_producer:
            for data_shm in self._data_shms:
                data_shm.unlink()
            for length_shm in self._length_shms:
                length_shm.unlink()
            self._seq_shm.unlink()
            self._write_buf_idx_shm.unlink()
            self._consumer_ts_shm.unlink()

        self._data_shms = []
        self._length_shms = []
        self._seq_shm = None
        self._write_buf_idx_shm = None
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
        """
        Produce a frame using ring buffering:
        1. Get the current write buffer index
        2. Mark write in progress (odd seq)
        3. Write to that buffer
        4. Rotate the buffer index to the next buffer
        5. Publish the frame (even seq)
        
        This allows consumers to read from a stable (non-writing) buffer.
        With NUM_BUFFERS > 2, there's more headroom for consumers to catch up.
        """
        
        if not self._is_producer:
            raise RuntimeError("Cannot produce frame from a consumer MemoryStreamer")
        
        jpeg_len = len(jpeg_bytes)

        if jpeg_len > MAX_JPEG_BYTES:
            print(
                f"[PRODUCER] JPEG too large ({jpeg_len} > {MAX_JPEG_BYTES}), dropping."
            )
            return

        # Get the buffer we'll write to
        write_idx = int(self._write_buf_idx_view[...])
        buf = self._data_shms[write_idx].buf
        length_view = self._length_views[write_idx]

        # ---- Seqlock write: mark write in progress (odd seq) ----
        self._seq_view[...] += 1     # now odd => readers will wait/retry

        # ---- Write JPEG bytes + length ----
        buf[:jpeg_len] = jpeg_bytes
        length_view[...] = jpeg_len

        # ---- Rotate buffer index to next buffer ----
        # After this, consumers will read from the buffer we just wrote to,
        # and the producer will write to the next buffer on the next call.
        self._write_buf_idx_view[...] = (write_idx + 1) % NUM_BUFFERS

        # ---- Publish: even seq means a complete frame is visible ----
        self._seq_view[...] += 1     # now even => readers can safely read

        # ---- Producer accounting (no debug output) ----
        # Keep a short-lived counter to avoid unbounded growth.
        self._producer_frame_count += 1
        now = time.time()
        elapsed_since_report = now - self._producer_last_report_ts
        if elapsed_since_report >= 2.0:
            # reset counters periodically (silently)
            self._producer_frame_count = 0
            self._producer_last_report_ts = now

    @property
    def is_connected(self) -> bool:
        if not self._created:
            return False
        
        seq0 = int(self._seq_view[...])
        return seq0 != -1
    
    def consume_frame(self) -> Optional[bytes]:
        """
        Consume a frame using ring buffering with detailed debug tracking.
        """

        if not self._init_consumer():
            return None
        
        self._write_consumer_timestamp()
        
        # ---- Cheap early check: is there a NEW frame at all? ----
        seq0 = int(self._seq_view[...])

        # Check if producer has gone offline
        if seq0 == -1:
            self.close()
            return None
        
        if seq0 == 0:
            # No frame yet - this is a common case
            return None

        # If seq is odd, writer is in progress
        if seq0 & 1:
            for _ in range(100):
                seq0 = int(self._seq_view[...])
                if not (seq0 & 1):
                    break
            else:
                # Spin-wait timed out - producer still writing
                return None

        # Get buffer indices
        write_idx = int(self._write_buf_idx_view[...])
        read_idx = (write_idx - 1) % NUM_BUFFERS
        
        buf = self._data_shms[read_idx].buf
        length_view = self._length_views[read_idx]

        # Check for producer advance during our read
        seq1_check = int(self._seq_view[...])
        write_idx_check = int(self._write_buf_idx_view[...])
        if write_idx_check != write_idx:
            # Producer cycled - recursively re-sync
            return self.consume_frame()

        # Check if this is a new frame (seq > last_seen_seq)
        if seq1_check <= self._last_seen_seq:
            # Same frame as before - no new data
            return None

        jpeg_len = int(length_view[...])
        if jpeg_len <= 0 or jpeg_len > MAX_JPEG_BYTES:
            return None

        # Copy JPEG bytes
        jpeg_bytes = bytes(buf[:jpeg_len])

        # Final consistency check
        seq_final = int(self._seq_view[...])
        if seq1_check == seq_final and not (seq_final & 1):
            # Valid frame
            self._last_seen_seq = seq_final
            self._last_read_buf_idx = read_idx
            return jpeg_bytes

        # Seq changed or became odd during copy - discard
        return None
