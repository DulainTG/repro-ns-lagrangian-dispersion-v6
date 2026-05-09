import numpy as np
from typing import Final
from .operators import DecomposedGradient

class VortexDiagnostic:
    """
    Diagnostic suite for identifying and analyzing vortex structures in turbulent flows.

    Provides methods to implement the Q-criterion and associated rotation/strain 
    dominance calculations required for characterising vortex trapping events in EXP3.
    """
    VORTEX_THRESHOLD: Final[float] = 0.0

    def calculate_rotation_dominance_magnitude(self, gradient: DecomposedGradient) -> np.ndarray:
        """
        Calculates the raw magnitude of rotation dominance over deformation.

        Computes the difference in the squared Frobenius norms of the vorticity tensor 
        (Omega) and the strain-rate tensor (S): ||Omega||^2 - ||S||^2.

        Args:
            gradient: The decomposed velocity gradient components (S and Omega tensors).

        Returns:
            np.ndarray: A scalar field of rotation dominance values across the grid.
        """
        # Frobenius norm squared: sum of squares of all elements in the 3x3 tensor at each point.
        # Vorticity tensor Omega and strain-rate tensor S have shape (3, 3, Nx, Ny, Nz).
        omega_sq = np.sum(gradient.vorticity_tensor**2, axis=(0, 1))
        s_sq = np.sum(gradient.strain_rate_tensor**2, axis=(0, 1))
        return omega_sq - s_sq

    def generate_q_criterion_field(self, gradient: DecomposedGradient) -> np.ndarray:
        """
        Generates the scalar Q-criterion field according to Equation 1 from the paper.

        Definition: Q = 0.5 * (||Omega||^2 - ||S||^2).
        This field is the primary metric for defining vortex residence in EXP3 analysis.

        Args:
            gradient: The decomposed velocity gradient components.

        Returns:
            np.ndarray: The Q-criterion scalar field.
        """
        return 0.5 * self.calculate_rotation_dominance_magnitude(gradient)

    def compute_dominance_indicator(self, q_field: np.ndarray, threshold: float=VORTEX_THRESHOLD) -> np.ndarray:
        """
        Creates a binary indicator field identifying where rotation dominates strain.

        Determines the 'vortex regions' by identifying where the Q-criterion field
        exceeds a specified threshold (typically 0.0).

        Args:
            q_field: The pre-calculated Q-criterion scalar field.
            threshold: The numerical threshold for dominance. Defaults to VORTEX_THRESHOLD.

        Returns:
            np.ndarray: A boolean scalar field with True identifying rotation-dominant cells.
        """
        return q_field > threshold
