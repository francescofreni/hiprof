import numpy as np
from abc import ABC, abstractmethod
from typing import Tuple, List


class ParameterDistribution(ABC):
    """
    Abstract base distribution class.
    """

    def __init__(
        self,
        seed: int = 42,
    ) -> None:
        """
        Initialize the class.

        Parameters
        ----------
        seed : int, optional
            Seed for the random number generator, by default 42.
        """
        self.rng = np.random.default_rng(seed)

    @abstractmethod
    def sample(self) -> float:
        """
        Draw an observation from the distribution.
        """
        pass


class Normal(ParameterDistribution):
    """
    Gaussian distribution.
    """

    def __init__(
        self,
        loc: float = 0,
        scale: float = 1,
        seed: int = 42,
    ) -> None:
        """
        Initialize a normal distribution.

        Parameters
        ----------
        loc : float, optional
            Mean of the normal distribution, by default 0.
        scale : float, optional
            Standard deviation of the normal distribution, by default 1.
        seed : int, optional
            Seed for the random number generator, by default 42.
        """
        super().__init__(seed)
        self.loc = loc
        self.scale = scale

    def sample(self) -> float:
        """
        Draw an observation from the normal distribution.

        Returns
        -------
        float
            A draw from N(loc, scale^2).
        """
        return self.rng.normal(loc=self.loc, scale=self.scale)


class Gamma(ParameterDistribution):
    """
    Gamma distribution.
    """

    def __init__(
        self,
        shape: float = 2,
        scale: float = 2,
        seed: int = 42,
    ) -> None:
        """
        Initialize a gamma distribution.

        Parameters
        ----------
        shape : float, optional
            Shape parameter of the gamma distribution, by default 2.
        scale : float, optional
            Scale parameter of the gamma distribution, by default 2.
        seed : int, optional
            Seed for the random number generator, by default 42.
        """
        super().__init__(seed)
        self.shape = shape
        self.scale = scale

    def sample(self) -> float:
        """
        Draw an observation from the gamma distribution.

        Returns
        -------
        float
            A draw from Gamma(shape, scale).
        """
        return self.rng.gamma(shape=self.shape, scale=self.scale)
