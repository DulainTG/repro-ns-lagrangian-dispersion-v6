import numpy as np
from typing import Protocol, runtime_checkable
from .transforms import WavenumberIndexMap, SpectralCoefficientBuffer

@runtime_checkable
class SpectralMaskKernel(Protocol):
    """Interface for logic that generates amplitude weights in the spectral domain.

    Implementations define how to weight or zero-out specific Fourier modes 
    based on wavenumber topography.
    """

    def compute_weights(self, index_map: WavenumberIndexMap) -> np.ndarray:
        """Generate a complex or real-valued masking array matching the spectral grid.

        Args:
            index_map: Map containing geometric and magnitude data for wavenumbers.

        Returns:
            Arithmetic mask array (scalars) with dimensions matching the grid components.
        """
        ...


class WavenumberBandMask:
    """Filter kernel that operates on shells of wavenumbers (n).

    Used to select or suppress energy scales within specific physical 
    frequency ranges [k_min, k_max].
    """

    def __init__(self, k_min: float, k_max: float, is_sharp: bool = True) -> None:
        """Initialize the band mask.

        Args:
            k_min: Lower wavenumber cutoff.
            k_max: Upper wavenumber cutoff.
            is_sharp: Whether to use a step function (0/1) or a tapered decay.
        """
        self.k_min = k_min
        self.k_max = k_max
        self.is_sharp = is_sharp

    def compute_weights(self, index_map: WavenumberIndexMap) -> np.ndarray:
        """Calculate the 3D weights array based on k-magnitude shells.

        Args:
            index_map: The coordinate map of wavenumbers.

        Returns:
            Array of shape (Nz, Ny, Nx) with mask values.
        """
        k = index_map.k_magnitude
        
        if self.is_sharp:
            mask = (k >= self.k_min) & (k <= self.k_max)
            return mask.astype(np.float64)
        
        # Tapered decay (using a smooth transition)
        # Using a transition width of 10% of k_max
        trans = 0.1 * self.k_max if self.k_max > 0 else 1.0
        
        # We want weights to be 1 between k_min and k_max
        # And fall to 0 outside.
        
        # Fall-off at k_min:
        if self.k_min > 0:
            # t = 0 at k = k_min - trans/2, t = 1 at k = k_min + trans/2
            t_low = (k - (self.k_min - trans/2)) / trans
            t_low = np.clip(t_low, 0, 1)
            low_weights = 3 * t_low**2 - 2 * t_low**3
        else:
            low_weights = 1.0
            
        # Fall-off at k_max:
        # t = 1 at k = k_max - trans/2, t = 0 at k = k_max + trans/2
        t_high = ((self.k_max + trans/2) - k) / trans
        t_high = np.clip(t_high, 0, 1)
        high_weights = 3 * t_high**2 - 2 * t_high**3
        
        return (low_weights * high_weights).astype(np.float64)


def apply_band_mask(buffer: SpectralCoefficientBuffer, mask: np.ndarray) -> SpectralCoefficientBuffer:
    """Performs direct element-wise multiplication of a mask and spectral coefficients.

    Args:
        buffer: The complex domain representation of the variable field.
        mask: The pre-computed mask array (3D or 4D).

    Returns:
        A filtered SpectralCoefficientBuffer.

    Raises:
        ValueError: If array shapes are incompatible for broadcasting.
    """
    data = buffer.data
    
    # Attempt to align mask to data for broadcasting if it's 3D and data is 4D (NX, NY, NZ, C)
    # Note: Case (C, NX, NY, NZ) and (NX, NY, NZ) works natively in NumPy.
    if data.ndim == 4 and mask.ndim == 3:
        if data.shape[:3] == mask.shape:
            # Handle (NX, NY, NZ, C) case by adding a trailing dimension to mask
            mask = mask[..., np.newaxis]
            
    try:
        filtered_data = data * mask
        return SpectralCoefficientBuffer(filtered_data)
    except Exception as e:
        if isinstance(e, ValueError):
            raise e
        raise ValueError(f"Incompatible shapes for masking: buffer {data.shape} and mask {mask.shape}. Error: {str(e)}")
