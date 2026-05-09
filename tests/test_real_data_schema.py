import os
import pytest
import numpy as np
from pathlib import Path
from src.io.vtk.parser import AthenaHeaderParser, VtkGridExtractor

@pytest.mark.timeout(10)
def test_real_data_schema_compatibility():
    """
    Guarded smoke test to validate schema compatibility with real data.
    Reads only headers and a tiny bounded amount of data.
    """
    raw_data_dir_env = os.environ.get("RAW_DATA_DIR", "/raw_data")
    raw_data_dir = Path(raw_data_dir_env)
    
    if not raw_data_dir.exists():
        pytest.skip(f"RAW_DATA_DIR {raw_data_dir} does not exist.")
        
    vtk_files = sorted(list(raw_data_dir.glob("Turb.hydro_w.*.vtk")))
    if not vtk_files:
        pytest.skip(f"No VTK files found in {raw_data_dir}.")
        
    sample_file = vtk_files[0]
    
    header_parser = AthenaHeaderParser()
    grid_extractor = VtkGridExtractor()
    
    with open(sample_file, "rb") as f:
        # Validate Header Compatibility
        metadata = header_parser.parse_header_metadata(f)
        assert "time" in metadata, "Metadata missing 'time' field"
        assert "variables" in metadata, "Metadata missing 'variables' field"
        
        # Validate Grid / Schema Compatibility
        grid = grid_extractor.extract_grid(f)
        # For DS1, we expect 128^3 grid
        assert grid.dimensions == (128, 128, 128), f"Unexpected grid dimensions: {grid.dimensions}"
        assert len(grid.spacing) == 3, "Grid spacing must be 3D"
        assert all(s > 0 for s in grid.spacing), "Grid spacing must be positive"

        # Bounded read of the binary data start (satisfying the "tiny bounded number of rows" req)
        # We don't use the production BinaryDataParser here to avoid loading the full 40MB,
        # instead we just verify we can find the start of a data block and read a few bytes.
        f.seek(0)
        content_head = f.read(30000) # Read enough to cover the header and start of data
        
        # Look for density scalars
        assert b"SCALARS dens float" in content_head, "Required 'dens' field not found in VTK header"
        
        # Find where the binary data would start
        lookup_idx = content_head.find(b"LOOKUP_TABLE default")
        if lookup_idx != -1:
            data_start = content_head.find(b"\n", lookup_idx) + 1
            if data_start > 0 and data_start < len(content_head):
                # Read 5 "rows" worth of data (assuming each "row" is a float)
                # In VTK binary, it's just a stream of big-endian floats.
                f.seek(data_start)
                tiny_data = f.read(5 * 4) # 5 floats
                assert len(tiny_data) == 20
                vals = np.frombuffer(tiny_data, dtype='>f4')
                assert len(vals) == 5
                assert not np.isnan(vals).any()
