import os
import pytest
import numpy as np
import io as python_io
from pathlib import Path
from src.io.athena import (
    AthenaSnapshotSequence, 
    VelocitySnapshotStream, 
    create_ds1_sequence,
    SnapshotData
)
from src.io.vtk.loader import (
    AthenaVtkLoader, 
    VtkPathResolver, 
    VtkSnapshotFields
)
from src.io.vtk.parser import (
    AthenaHeaderParser, 
    AthenaBinaryDataParser, 
    VtkGridExtractor
)
from src.io.binary import RawByteStreamReader
from src.fields.grid import GridMetadata

@pytest.fixture
def raw_data_dir():
    # Attempt to get from env, fallback to expected location
    return Path(os.environ.get('RAW_DATA_DIR', '/raw_data'))

@pytest.fixture
def athena_loader():
    header_parser = AthenaHeaderParser()
    data_parser = AthenaBinaryDataParser()
    grid_extractor = VtkGridExtractor()
    return AthenaVtkLoader(header_parser, data_parser, grid_extractor)

def test_vtk_path_resolver(raw_data_dir):
    """Test that VtkPathResolver correctly identifies and locates VTK files."""
    resolver = VtkPathResolver(raw_data_dir)
    indices = resolver.list_available_indices()
    assert len(indices) > 0
    # 18903 is known to be in the DS1 dataset
    assert 18903 in indices
    
    path = resolver.get_path(18903)
    assert path.exists()
    assert path.name == "Turb.hydro_w.18903.vtk"
    
    with pytest.raises(FileNotFoundError):
        resolver.get_path(1)

def test_athena_vtk_loader_integration(raw_data_dir, athena_loader):
    """Integrates AthenaVtkLoader with its parsers to load real VTK data and verify contracts."""
    resolver = VtkPathResolver(raw_data_dir)
    path = resolver.get_path(18903)
    
    # Test metadata extraction integration
    metadata = athena_loader.load_metadata(path)
    assert isinstance(metadata, GridMetadata)
    assert metadata.dimensions == (128, 128, 128)
    # Check spacing for the 128^3 grid on a unit domain (default extent 1.0)
    assert metadata.spacing == pytest.approx((1/128, 1/128, 1/128), rel=1e-3)
    
    # Test simulation time extraction
    time = athena_loader.load_time(path)
    assert isinstance(time, (float, int))
    assert time >= 0.0
    
    # Test full field data loading (density and velocity)
    fields = athena_loader.load_fields(path)
    assert isinstance(fields, VtkSnapshotFields)
    nx, ny, nz = metadata.dimensions
    # Athena data is typically stored as (3, NX, NY, NZ) for velocity and (NX, NY, NZ) for density
    assert fields.velocity.shape == (3, nx, ny, nz)
    assert fields.density.shape == (nx, ny, nz)
    
    # Verify data sanity
    assert not np.isnan(fields.density).any()
    assert not np.isnan(fields.velocity).any()

def test_snapshot_sequence_integration(raw_data_dir, athena_loader):
    """Tests the interaction between resolver, loader and the lazy sequence iterator."""
    resolver = VtkPathResolver(raw_data_dir)
    # Test with a small subset of indices to verify temporal sequencing
    indices = [18903, 18913, 18923]
    sequence = AthenaSnapshotSequence(resolver, athena_loader, indices)
    
    snapshots = list(sequence)
    assert len(snapshots) == 3
    for i, snapshot in enumerate(snapshots):
        assert isinstance(snapshot, SnapshotData)
        assert snapshot.index == indices[i]
        assert snapshot.metadata.dimensions == (128, 128, 128)
        assert isinstance(snapshot.physical_time, float)
        assert snapshot.fields.velocity.shape == (3, 128, 128, 128)

def test_velocity_stream_integration(raw_data_dir, athena_loader):
    """Tests filtering and reshaping of snapshot sequence into a velocity stream for analysis."""
    resolver = VtkPathResolver(raw_data_dir)
    indices = [18903]
    sequence = AthenaSnapshotSequence(resolver, athena_loader, indices)
    stream = VelocitySnapshotStream(sequence)
    
    results = list(stream)
    assert len(results) == 1
    time, velocity = results[0]
    assert isinstance(time, float)
    # VelocitySnapshotStream transposes (3, NX, NY, NZ) to (NX, NY, NZ, 3) for experiment consumption
    assert velocity.shape == (128, 128, 128, 3)

def test_create_ds1_sequence_factory(raw_data_dir, athena_loader):
    """Tests the factory function for creating the standard DS1 dataset sequence."""
    sequence = create_ds1_sequence(raw_data_dir, athena_loader)
    assert isinstance(sequence, AthenaSnapshotSequence)
    # DS1 is defined as 18903 to 19893 with step 10, which is 100 snapshots
    assert len(sequence.indices) == 100
    assert sequence.indices[0] == 18903
    assert sequence.indices[-1] == 19893
    
    # Verify we can at least load the first snapshot in the DS1 sequence
    snapshot = next(iter(sequence))
    assert snapshot.index == 18903

def test_raw_byte_stream_reader():
    """Tests the low-level binary reader component independently."""
    # Create a dummy binary stream with big-endian floats
    data = np.array([1.0, 2.0, 3.0, 4.0], dtype='>f4').tobytes()
    stream = python_io.BytesIO(data)
    reader = RawByteStreamReader(stream)
    
    # Read numeric block
    read_data = reader.read_numeric_block(2, dtype='float32', byte_order='>')
    assert len(read_data) == 2
    assert np.allclose(read_data, [1.0, 2.0])
    
    # Skip bytes (skipping float 3.0, which is 4 bytes)
    reader.skip_bytes(4) 
    
    # Read remaining float 4.0
    remaining = reader.read_numeric_block(1, dtype='float32', byte_order='>')
    assert remaining[0] == 4.0
    
    # Expect EOF when reading beyond available data
    with pytest.raises(EOFError):
        reader.read_numeric_block(1, dtype='float32', byte_order='>')

def test_error_handling_invalid_vtk(tmp_path):
    """Tests error handling for files that are not valid VTK files."""
    invalid_file = tmp_path / "invalid.vtk"
    invalid_file.write_text("This is not a VTK file header\n")
    
    header_parser = AthenaHeaderParser()
    with open(invalid_file, "rb") as f:
        with pytest.raises(ValueError, match="Not a valid VTK file"):
            header_parser.parse_header_metadata(f)

def test_error_handling_athena_specific_format(tmp_path):
    """Tests detection of non-Athena compliant VTK headers."""
    non_athena_file = tmp_path / "not_athena.vtk"
    # Valid VTK header but missing Athena-style comment line with time/cycle
    non_athena_file.write_text("# vtk DataFile Version 2.0\nJust a normal VTK file comment line\n")
    
    header_parser = AthenaHeaderParser()
    with open(non_athena_file, "rb") as f:
        with pytest.raises(ValueError, match=r"Header does not match the expected Athena\+\+ format"):
            header_parser.parse_header_metadata(f)

def test_error_handling_missing_grid_parameters(tmp_path):
    """Tests failure when mandatory VTK grid parameters are missing."""
    bad_grid_file = tmp_path / "bad_grid.vtk"
    # Missing SPACING
    bad_grid_file.write_text("# vtk DataFile Version 2.0\ntime=0.0 cycle=0 variables=dens\nASCII\nDATASET STRUCTURED_POINTS\nDIMENSIONS 129 129 129\nORIGIN 0 0 0\n")
    
    grid_extractor = VtkGridExtractor()
    with open(bad_grid_file, "rb") as f:
        with pytest.raises(RuntimeError, match=r"Mandatory grid parameters \(DIMENSIONS, SPACING\) are missing\."):
            grid_extractor.extract_grid(f)
