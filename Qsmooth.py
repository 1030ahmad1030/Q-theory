"""
QSmooth: Response smoothing for persistence timescale analysis.
Pure smoothing functions only - no extra functionality.
Version: 1.0
"""

import numpy as np
from scipy.signal import savgol_filter, butter, filtfilt
from scipy.ndimage import gaussian_filter1d


class QSmooth:
    """
    QSmooth: Collection of smoothing filters for response signals.
    """

    @staticmethod
    def savgol(
        t: np.ndarray,
        R: np.ndarray,
        window_frac: float = 0.05,
        polyorder: int = 3
    ) -> np.ndarray:
        """
        Savitzky-Golay smoothing.

        BEST FOR: τ_s, τ_u (derivative-critical estimators)
        PRESERVES: Local polynomial structure, peak positions, analytical derivatives
        """
        R = np.asarray(R, dtype=float)
        n = len(R)

        window_length = max(5, int(window_frac * n))
        if window_length % 2 == 0:
            window_length += 1

        if window_length > n:
            window_length = n - 1 if (n % 2 == 0) else n

        if window_length < polyorder + 2:
            window_length = polyorder + 2
            if window_length % 2 == 0:
                window_length += 1

        return savgol_filter(R, window_length, polyorder)

    @staticmethod
    def butterworth(
        t: np.ndarray,
        R: np.ndarray,
        cutoff_frac: float = 0.9,
        order: int = 4
    ) -> np.ndarray:
        """
        Zero-phase Butterworth filter (filtfilt).

        BEST FOR: τ_env², τ_E³ (frequency/phase-critical estimators)
        PRESERVES: Phase, frequency content, zero-phase distortion
        """
        t = np.asarray(t, dtype=float)
        R = np.asarray(R, dtype=float)

        dt = np.mean(np.diff(t))
        if dt <= 0:
            raise ValueError("Invalid time array: dt must be positive.")

        cutoff_normalized = float(np.clip(cutoff_frac, 0.01, 0.99))

        b, a = butter(order, cutoff_normalized, btype='low', analog=False)
        return filtfilt(b, a, R)

    @staticmethod
    def gaussian(
        t: np.ndarray,
        R: np.ndarray,
        sigma_frac: float = 0.02
    ) -> np.ndarray:
        """
        Gaussian smoothing.

        BEST FOR: τ², τ³ (shape-preserving estimators)
        PRESERVES: Global shape, monotonicity, smoothness
        """
        t = np.asarray(t, dtype=float)
        R = np.asarray(R, dtype=float)

        dt = np.mean(np.diff(t))
        if dt <= 0:
            raise ValueError("Invalid time array: dt must be positive.")

        t_span = t[-1] - t[0]
        sigma_time = sigma_frac * t_span
        sigma_points = sigma_time / dt

        return gaussian_filter1d(R, sigma=sigma_points)


__all__ = ["QSmooth"]
