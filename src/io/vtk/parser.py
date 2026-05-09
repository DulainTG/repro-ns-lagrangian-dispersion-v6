import numpy as np
import re
from typing import BinaryIO, Dict, Union, Tuple
from src.fields.grid import GridMetadata

class AthenaHeaderParser:
    """Parses Athena++ specific comment lines within VTK headers.

    Athena++ VTK files include a comment line (usually the second line) containing
    simulation time, cycle number, and variable descriptions.
    """

    def parse_header_metadata(self, stream: BinaryIO) -> Dict[str, Union[float, int, str]]:
        """Extracts simulation state from the VTK header lines.

        Args:
            stream: The open VTK file stream in binary mode.

        Returns:
            A dictionary containing parsed metadata (e.g., 'time', 'cycle', 'variables').

        Raises:
            ValueError: If the header does not match the expected Athena++ format or session version.
        """
        stream.seek(0)
        try:
            line1 = stream.readline().decode('ascii', errors='ignore')
            line2 = stream.readline().decode('ascii', errors='ignore')
        except Exception as e:
            raise ValueError(f"Failed to read VTK header: {e}")

        if not line1.startswith('# vtk DataFile Version'):
            raise ValueError("Not a valid VTK file")

        metadata = {}
        time_match = re.search(r'time=\s*([\d.e+-]+)', line2)
        cycle_match = re.search(r'cycle=\s*(\d+)', line2)
        vars_match = re.search(r'variables=\s*([\w,_]+)', line2)

        if time_match:
            metadata['time'] = float(time_match.group(1))
        if cycle_match:
            metadata['cycle'] = int(cycle_match.group(1))
        if vars_match:
            metadata['variables'] = vars_match.group(1)

        if 'time' not in metadata and 'cycle' not in metadata:
            raise ValueError("Header does not match the expected Athena++ format")

        return metadata

class VtkGridExtractor:
    """Extracts the computational domain layout from VTK STRUCTURED_POINTS sections.

    Handles the extraction of DIMENSIONS, ORIGIN, and SPACING parameters to configure
    the simulation mesh for Lagrangian integration.
    """

    def extract_grid(self, stream: BinaryIO) -> GridMetadata:
        """Parses grid dimensions and physical scaling from the VTK stream.

        Args:
            stream: The open VTK file stream positioned after the version header.

        Returns:
            A configured GridMetadata object for the 3D domain.

        Raises:
            RuntimeError: If mandatory grid parameters (DIMENSIONS, SPACING) are missing.
        """
        stream.seek(0)
        # Read the beginning of the file to find grid parameters. 
        # 20KB should be plenty for the header.
        content = stream.read(20000).decode('ascii', errors='ignore')

        dims_match = re.search(r'DIMENSIONS\s+(\d+)\s+(\d+)\s+(\d+)', content)
        origin_match = re.search(r'ORIGIN\s+([\d.e+-]+)\s+([\d.e+-]+)\s+([\d.e+-]+)', content)
        spacing_match = re.search(r'SPACING\s+([\d.e+-]+)\s+([\d.e+-]+)\s+([\d.e+-]+)', content)

        if not dims_match or not spacing_match:
            raise RuntimeError("Mandatory grid parameters (DIMENSIONS, SPACING) are missing.")

        dims = tuple(map(int, dims_match.groups()))
        origin = tuple(map(float, origin_match.groups())) if origin_match else (0.0, 0.0, 0.0)
        spacing = tuple(map(float, spacing_match.groups()))

        # Determine if it's cell data (Athena++ VTK files usually are)
        is_cell_data = 'CELL_DATA' in content
        
        if is_cell_data:
            # VTK STRUCTURED_POINTS DIMENSIONS are for points. 
            # In CELL_DATA mode, there are (NX-1)*(NY-1)*(NZ-1) cells.
            data_dims = (dims[0] - 1, dims[1] - 1, dims[2] - 1)
        else:
            data_dims = dims

        return GridMetadata(
            dimensions=data_dims,
            origin=origin,
            spacing=spacing
        )

class AthenaBinaryDataParser:
    """Parses binary data blocks for fluid variables in Athena++ VTK files.

    Specifically handles 'dens' (scalar) and 'vel' (vector) data following
    the VTK binary keyword.
    """

    def read_density_field(self, stream: BinaryIO, grid: GridMetadata) -> np.ndarray:
        """Parses the 3D scalar density field from the current stream position.

        Args:
            stream: The VTK file stream positioned at the start of the density binary block.
            grid: Metadata describing the grid dimensions used to reshape the data.

        Returns:
            A 3D numpy array of shape (NX, NY, NZ) containing density values.

        Raises:
            IOError: If the block size does not match the grid cell count.
        """
        nx, ny, nz = grid.dimensions
        num_elements = nx * ny * nz
        
        stream.seek(0)
        content = stream.read()
        
        # Look for SCALARS dens float. Using regex to be robust against whitespace.
        match = re.search(rb'SCALARS\s+dens\s+float', content)
        if not match:
            raise IOError("Density field 'dens' not found in VTK stream.")
        
        idx = match.start()
        # Find the next LOOKUP_TABLE line
        lookup_match = re.search(rb'LOOKUP_TABLE\s+\w+', content[idx:])
        if not lookup_match:
            raise IOError("LOOKUP_TABLE not found for dens.")
        
        lookup_idx = idx + lookup_match.start()
        # Data starts after the newline following the LOOKUP_TABLE line
        data_start = content.find(b'\n', lookup_idx) + 1
        
        data_bytes = content[data_start : data_start + num_elements * 4]
        if len(data_bytes) < num_elements * 4:
            raise IOError(f"Remaining stream size {len(data_bytes)} is less than expected {num_elements * 4}.")
            
        data = np.frombuffer(data_bytes, dtype='>f4').copy()
        # VTK is x-fastest, then y, then z -> (nz, ny, nx) in numpy C-order
        return data.reshape((nz, ny, nx)).transpose(2, 1, 0)

    def read_velocity_field(self, stream: BinaryIO, grid: GridMetadata) -> np.ndarray:
        """Parses the 3D vector velocity field from the current stream position.

        Args:
            stream: The VTK file stream positioned at the start of the velocity binary block.
            grid: Metadata describing the grid dimensions used to reshape the data.

        Returns:
            A 4D numpy array of shape (3, NX, NY, NZ) containing (vx, vy, vz).

        Raises:
            IOError: If the block size does not match 3x the grid cell count.
        """
        nx, ny, nz = grid.dimensions
        num_elements = nx * ny * nz
        
        stream.seek(0)
        content = stream.read()
        
        # Try to find VECTORS vel float (one possibility in VTK)
        vec_match = re.search(rb'VECTORS\s+vel\s+float', content)
        if vec_match:
            idx = vec_match.start()
            data_start = content.find(b'\n', idx) + 1
            data_bytes = content[data_start : data_start + num_elements * 3 * 4]
            if len(data_bytes) < num_elements * 3 * 4:
                raise IOError(f"Insufficient data for velocity vectors: expected {num_elements * 3 * 4}, got {len(data_bytes)}.")
            
            data = np.frombuffer(data_bytes, dtype='>f4').copy()
            # VTK x-fastest -> (nz, ny, nx, 3) in numpy C-order
            return data.reshape((nz, ny, nx, 3)).transpose(3, 2, 1, 0)
        
        # Alternatively, find individual scalars velx, vely, velz (common in Athena++)
        scalars = []
        for comp in [b'velx', b'vely', b'velz']:
            s_match = re.search(rb'SCALARS\s+' + comp + rb'\s+float', content)
            if not s_match:
                raise IOError(f"Velocity component {comp.decode()} not found.")
            
            idx = s_match.start()
            lookup_match = re.search(rb'LOOKUP_TABLE\s+\w+', content[idx:])
            if not lookup_match:
                raise IOError(f"LOOKUP_TABLE not found for {comp.decode()}.")
            
            lookup_idx = idx + lookup_match.start()
            data_start = content.find(b'\n', lookup_idx) + 1
            data_bytes = content[data_start : data_start + num_elements * 4]
            if len(data_bytes) < num_elements * 4:
                raise IOError(f"Insufficient data for {comp.decode()}.")
            
            data = np.frombuffer(data_bytes, dtype='>f4').copy()
            scalars.append(data.reshape((nz, ny, nx)).transpose(2, 1, 0))
            
        return np.stack(scalars, axis=0)
