import numpy as np
from ..grid import GridMetadata

class WavenumberIndexMap:
    """Storage for wavenumber index mapping in spectral space.

    Maps discrete grid indices to physical wavenumbers k and scaled integer 
    wavenumbers n relative to the domain size L. Crucial for applying the 
    sharp spectral filter (n=1 to 3) defined in EXP2.
    """

    def __init__(self, metadata: GridMetadata) -> None:
        self._metadata = metadata
        nx, ny, nz = metadata.dimensions
        
        # Determine integer wavenumbers for each dimension
        # np.fft.fftfreq(N) * N gives the integer wavenumbers [0, 1, ..., N/2-1, -N/2, ..., -1]
        nx_idx = np.fft.fftfreq(nx) * nx
        ny_idx = np.fft.fftfreq(ny) * ny
        nz_idx = np.fft.fftfreq(nz) * nz
        
        # Create 3D grid of integer wavenumbers using 'ij' indexing to match spatial dimensions (x, y, z)
        kv_x, kv_y, kv_z = np.meshgrid(nx_idx, ny_idx, nz_idx, indexing='ij')
        
        # Calculate scaled integer wavenumber magnitude n = sqrt(nx^2 + ny^2 + nz^2)
        self._n_magnitude = np.sqrt(kv_x**2 + kv_y**2 + kv_z**2)
        
        # Calculate physical wavenumber k = (2*pi/L) * n
        # metadata.extent is L
        self._k_magnitude = (2.0 * np.pi / metadata.extent) * self._n_magnitude

    @property
    def k_magnitude(self) -> np.ndarray:
        """Returns the magnitude |k| for every point in the 3D spectral grid."""
        return self._k_magnitude

    def get_mask(self, min_n: float, max_n: float) -> np.ndarray:
        """Generates a boolean mask for wavenumbers n in the range [min_n, max_n].

        Args:
            min_n: Lower bound of the integer-scaled wavenumber filter range.
            max_n: Upper bound of the integer-scaled wavenumber filter range.

        Returns:
            np.ndarray: Boolean mask suitable for element-wise multiplication with spectral data.
        """
        return (self._n_magnitude >= min_n) & (self._n_magnitude <= max_n)


class SpectralCoefficientBuffer:
    """Storage for complex Fourier coefficients.

    Encapsulates the complex Fourier coefficients produced by 3D transforms. 
    Provides a standard container for intermediate filtering operations.
    """

    def __init__(self, coefficients: np.ndarray) -> None:
        # Standardize storage to complex128 as per requirements
        if not np.iscomplexobj(coefficients):
            self._data = coefficients.astype(np.complex128)
        else:
            self._data = coefficients.astype(np.complex128, copy=False)

    @property
    def data(self) -> np.ndarray:
        """Returns the underlying complex-valued spectral array (complex128)."""
        return self._data
