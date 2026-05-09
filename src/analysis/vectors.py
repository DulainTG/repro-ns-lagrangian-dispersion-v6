import numpy as np
from dataclasses import dataclass

@dataclass(frozen=True)
class VectorDecomposition:
    """Container for a vector ensemble decomposed into parallel and perpendicular components.

    Specific to the requirements of EXP2 (Anisotropy of Filtered Dispersion) where 
    displacements are projected relative to the local large-scale velocity field.

    Attributes:
        parallel: The component vectors parallel to the reference direction, shape (N, 3).
        perpendicular: The component vectors perpendicular to the reference direction, shape (N, 3).
    """
    parallel: np.ndarray
    perpendicular: np.ndarray


class VectorProjector:
    """Logic for vector alignment and parallel/perpendicular decomposition in flow fields.

    This class provides the geometric kernels required to calculate the dynamic 
    anisotropy ratio lambda(t) as defined in EXP2 Eq 4 and Eq 5.
    """

    def calculate_alignment_cosine(self, vectors: np.ndarray, directions: np.ndarray) -> np.ndarray:
        """Calculate the cosine of the angle (dot product) between vectors and reference directions.

        Args:
            vectors: Array of vectors to analyze, shape (N, 3).
            directions: Array of reference unit vectors (e.g., V_LS_unit), shape (N, 3).

        Returns:
            Array of dot products representing alignment, shape (N,).

        Raises:
            ValueError: If input shapes are incompatible or directions are not unit vectors.
        """
        if vectors.shape != directions.shape:
            raise ValueError(f"Input shapes must match. Got {vectors.shape} and {directions.shape}")
        
        # Verify directions are unit vectors
        mags = np.linalg.norm(directions, axis=-1)
        if not np.allclose(mags, 1.0, atol=1e-6):
            raise ValueError("Reference directions must be unit vectors.")

        # cos(theta) = (v . d) / (|v| * |d|). Since |d| = 1, cos(theta) = (v . d) / |v|
        dot_product = np.sum(vectors * directions, axis=-1)
        vector_mags = np.linalg.norm(vectors, axis=-1)
        
        # Handle zero-magnitude vectors to avoid division by zero
        cosine = np.divide(dot_product, vector_mags, 
                          out=np.zeros_like(dot_product), 
                          where=vector_mags > 1e-12)
        return cosine

    def calculate_parallel_projection(self, vectors: np.ndarray, unit_directions: np.ndarray) -> np.ndarray:
        """Project vectors onto a reference unit direction (scalar projection).

        Corresponds to the term (delta_x(t) . V_LS_unit) in EXP2 Eq 4.

        Args:
            vectors: Array of vectors (e.g., displacements), shape (N, 3).
            unit_directions: Array of reference unit vectors, shape (N, 3).

        Returns:
            Scalar projection magnitudes, shape (N,).

        Raises:
            ValueError: If directions are not normalized.
        """
        if vectors.shape != unit_directions.shape:
            raise ValueError(f"Input shapes must match. Got {vectors.shape} and {unit_directions.shape}")
            
        mags = np.linalg.norm(unit_directions, axis=-1)
        if not np.allclose(mags, 1.0, atol=1e-6):
            raise ValueError("Unit directions must be normalized (unit vectors).")
            
        return np.sum(vectors * unit_directions, axis=-1)

    def decompose_vector_field(self, vectors: np.ndarray, unit_directions: np.ndarray) -> VectorDecomposition:
        """Perform a full parallel and perpendicular decomposition of a vector ensemble.

        Implements the logic for both MSD_parallel and MSD_perp components as 
        defined in EXP2 (Equations 4 and 5).

        Args:
            vectors: The vectors to decompose (e.g., tracer displacements), shape (N, 3).
            unit_directions: The reference local flow unit vectors (V_LS_unit), shape (N, 3).

        Returns:
            A VectorDecomposition object containing the split components.

        Raises:
            ValueError: If input dimensions do not match or directions are not unit vectors.
        """
        if vectors.shape != unit_directions.shape:
            raise ValueError(f"Input shapes must match. Got {vectors.shape} and {unit_directions.shape}")
        
        # Verify directions are unit vectors
        # (Though calculate_parallel_projection also checks this, it's good to be explicit here 
        # or just rely on it. Since calculate_parallel_projection is already called, 
        # we know it will be checked there.)
        
        # Scalar projection (N,)
        v_parallel_mag = self.calculate_parallel_projection(vectors, unit_directions)
        
        # Parallel vector component: v_parallel = (v . d) * d
        # Shape: (N, 1) * (N, 3) -> (N, 3)
        v_parallel = v_parallel_mag[..., np.newaxis] * unit_directions
        
        # Perpendicular vector component: v_perp = v - v_parallel
        v_perp = vectors - v_parallel
        
        return VectorDecomposition(parallel=v_parallel, perpendicular=v_perp)
