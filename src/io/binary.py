import numpy as np
from typing import BinaryIO, Literal

class RawByteStreamReader:
    """
    A low-level reader for extracting structured numerical data from binary streams.

    This reader is specifically designed to handle the 'BINARY' data blocks in 
    VTK 2.0 files used in the Solenoidal Turbulence DNS Velocity Snapshots (DS1), 
    supporting the extraction of large scale field data (vorticity, density, velocity) 
    from the raw bypass streams.
    """

    def __init__(self, stream: BinaryIO) -> None:
        """
        Initializes the reader with an open binary stream.

        Args:
            stream: A binary file-like object positioned at the start of a data segment.
        """
        self.stream = stream

    def read_numeric_block(self, count: int, dtype: np.dtype, byte_order: Literal['>', '<', '=']='>') -> np.ndarray:
        """
        Reads a block of fixed-size numeric elements into a numpy array.

        Args:
            count: The number of elements to read from the current stream position.
            dtype: The target numpy data type for the array elements (e.g., float32).
            byte_order: The endianness of the source data. VTK files typically use 
                big-endian ('>'). Use '<' for little-endian or '=' for system native.

        Returns:
            A 1D numpy array containing the parsed numerical data.

        Raises:
            EOFError: If the stream ends before the requested number of elements are read.
            ValueError: If the provided count, dtype or byte_order is invalid.
            IOError: If an error occurs during stream reading.
        """
        if count < 0:
            raise ValueError(f"count must be non-negative, got {count}")
        if byte_order not in ('>', '<', '='):
            raise ValueError(f"Invalid byte_order: {byte_order}")

        # Ensure dtype is a numpy dtype and apply byte order
        dt = np.dtype(dtype).newbyteorder(byte_order)
        bytes_to_read = count * dt.itemsize
        
        try:
            raw_bytes = self.stream.read(bytes_to_read)
        except Exception as e:
            raise IOError(f"Error reading from stream: {e}")
        
        if len(raw_bytes) < bytes_to_read:
            raise EOFError(f"Requested {bytes_to_read} bytes, but only {len(raw_bytes)} bytes were read.")
        
        return np.frombuffer(raw_bytes, dtype=dt)

    def skip_bytes(self, num_bytes: int) -> None:
        """
        Advances the stream pointer by a specific number of bytes.

        Used to navigate over VTK file headers or sub-block identifiers to reach 
        actual grid point data.

        Args:
            num_bytes: Number of bytes to skip forward.

        Raises:
            IOError: If seeking fails or exceeds stream bounds.
        """
        try:
            # Get current position
            current_pos = self.stream.tell()
            
            # Seek relative to current position
            # Note: stream.seek(num_bytes, 1) is not supported by all BinaryIO (e.g. some wrappers)
            # but is standard for files.
            self.stream.seek(num_bytes, 1)
            
            # Check if we moved as expected. Some systems allow seeking beyond EOF
            # but others might truncate.
            new_pos = self.stream.tell()
            if new_pos < 0:
                 raise IOError(f"Seek resulted in negative position: {new_pos}")
            
            # If num_bytes > 0 and new_pos didn't move enough, we likely hit EOF
            if num_bytes > 0 and new_pos < current_pos + num_bytes:
                 raise IOError(f"Seek exceeded stream bounds (EOF): requested {num_bytes}, only moved {new_pos - current_pos}")
                 
        except (IOError, ValueError) as e:
            if isinstance(e, IOError):
                raise
            raise IOError(f"Failed to skip {num_bytes} bytes: {e}")


