import numpy as np


class KalmanFilter1D:
    """
    1D scalar Kalman filter with RTS smoother and multi-sensor fusion.

    Parameters
    ----------
    x0 : float   Initial state estimate
    P0 : float   Initial state uncertainty (variance)
    A  : float   State transition coefficient
    H  : float   Observation coefficient
    Q  : float   Process noise variance
    R  : float   Default measurement noise variance
    """

    def __init__(self, x0: float, P0: float, A: float, H: float,
                 Q: float, R: float):
        self.x = x0
        self.P = P0
        self.A = A
        self.H = H
        self.Q = Q
        self.R = R

        self._xs:     list[float] = []
        self._Ps:     list[float] = []
        self._xpreds: list[float] = []
        self._Ppreds: list[float] = []

    # ------------------------------------------------------------------
    # Core steps
    # ------------------------------------------------------------------

    def predict(self) -> None:
        """Propagate state and covariance forward one timestep."""
        self.x = self.A * self.x
        self.P = self.A * self.P * self.A + self.Q

    def update(self, z: float, R: float | None = None) -> None:
        """Incorporate one measurement, updating x and P in place."""
        R      = R if R is not None else self.R
        K      = self.P / (self.P + self.H * self.H * R)
        self.x = self.x + K * (z - self.H * self.x)
        self.P = (1 - K * self.H) * self.P

    def step(self, measurements) -> None:
        """
        One full timestep: predict then update.

        measurements : float  — single sensor reading
                     | list of (z, R) tuples — multiple sensors
        """
        self.predict()
        self._xpreds.append(self.x)
        self._Ppreds.append(self.P)

        if hasattr(measurements, '__iter__'):
            for z, R in measurements:
                self.update(z, R)
        else:
            self.update(measurements)

        self._xs.append(self.x)
        self._Ps.append(self.P)

    # ------------------------------------------------------------------
    # Batch operations
    # ------------------------------------------------------------------

    def filter(self, measurements) -> np.ndarray:
        """Run the filter over a sequence of measurements."""
        for z in measurements:
            self.step(z)
        return self.estimates

    def smooth(self) -> np.ndarray:
        """RTS backward pass — call after filter()."""
        if not self._xs:
            raise RuntimeError("No data to smooth — call filter() first.")
        xs     = np.array(self._xs)
        Ps     = np.array(self._Ps)
        xpreds = np.array(self._xpreds)
        Ppreds = np.array(self._Ppreds)

        smoothed = xs.copy()
        for k in range(len(xs) - 2, -1, -1):
            C           = Ps[k] * self.A / Ppreds[k + 1]
            smoothed[k] = xs[k] + C * (smoothed[k + 1] - xpreds[k + 1])
        return smoothed

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def estimates(self) -> np.ndarray:
        """Filtered state estimates."""
        return np.array(self._xs)

    @property
    def uncertainty(self) -> np.ndarray:
        """Posterior standard deviations (sqrt of variance)."""
        return np.sqrt(np.array(self._Ps))
