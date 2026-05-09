import numpy as np
from typing import Optional
from .domain import DomainBox

def seed_uniform_randomly(count: int, low: np.ndarray, high: np.ndarray, seed: Optional[int]=None) -> np.ndarray:
    """
    Generates a set of 3D points uniformly distributed within specified bounds.

    Args:
        count: Number of particles/tracers to generate.
        low: array-like of shape (3,) representing the minimum [x, y, z] coordinates.
        high: array-like of shape (3,) representing the maximum [x, y, z] coordinates.
        seed: Optional integer seed for the random number generator to ensure reproducibility.

    Returns:
        An array of shape (count, 3) representing the generated spatial positions.

    Raises:
        ValueError: If count is negative or if low/high bounds are incompatible.
    """
    if count < 0:
        raise ValueError(f"count must be non-negative, got {count}")
    
    # Ensure inputs are numpy arrays
    low = np.asanyarray(low)
    high = np.asanyarray(high)
    
    if low.shape != (3,) or high.shape != (3,):
        raise ValueError(f"low and high must have shape (3,), got {low.shape} and {high.shape}")

    if count == 0:
        return np.empty((0, 3))

    rng = np.random.default_rng(seed)
    # Generate points in [low, high)
    return rng.uniform(low, high, size=(count, 3))

class ParticleSeeder:
    """
    Responsible for domain-wide initialization of Lagrangian tracers for turbulence experiments.
    
    This class ensures that tracer particles are initialized within the valid physical bounds
    of the simulation domain defined in experiments such as EXP1 (Lagrangian Tracer Integration).
    """

    def __init__(self, domain: DomainBox, seed: Optional[int]=42) -> None:
        """
        Initialize the seeder with domain constraints and a PRNG state.

        Args:
            domain: The DomainBox specifying the cubic volume (extent and origin).
            seed: Random seed for deterministic initialization as required for paper reproduction.
        """
        self._domain = domain
        self._seed = seed

    def generate_initial_positions(self, num_tracers: int) -> np.ndarray:
        """
        Calculates initial positions for a tracer ensemble distributed across the entire domain.

        This implements the 'Initialize 8,000 tracers' step of EXP1 in the paper's procedure.

        Args:
            num_tracers: Total number of tracers to seed (e.g., 8,000).

        Returns:
            Array of coordinates with shape (num_tracers, 3).

        Raises:
            RuntimeError: If initialization fails due to invalid domain configuration.
        """
        try:
            low, high = self._domain.bounds
            return seed_uniform_randomly(num_tracers, low, high, seed=self._seed)
        except (ValueError, TypeError, AttributeError) as e:
            # Catch potential errors from domain.bounds or seed_uniform_randomly
            raise RuntimeError(f"Initialization failed: {e}") from e
