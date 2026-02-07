"""
QSignature: Clean, organized causal persistence timescale estimators.
Version: 1.0
"""

import numpy as np
from scipy.signal import hilbert, welch
from typing import Optional, Literal, Dict


class QSignature:
    """
    QSignature: Collection of timescale estimators and diagnostics
    for causal persistence analysis.
    """

    # ============================================================================
    # CORE ESTIMATORS (6 essential functions)
    # ============================================================================

    @staticmethod
    def tau_g(t: np.ndarray, R: np.ndarray, R_inf: Optional[float] = None) -> float:
        """
        τ_g: Generalized memory persistence timescale.

        τ_g = ∫₀^∞ t·|R(t) - R∞| dt / ∫₀^∞ |R(t) - R∞| dt
        """
        t = np.asarray(t, dtype=float)
        R = np.asarray(R, dtype=float)

        if R_inf is None:
            R_inf = R[-1]

        deviation = np.abs(R - R_inf)

        if np.all(deviation == 0) or np.sum(deviation) == 0:
            return 0.0

        numerator = np.trapezoid(t * deviation, t)
        denominator = np.trapezoid(deviation, t)

        if denominator == 0 or not np.isfinite(denominator):
            return np.nan

        return numerator / denominator

    @staticmethod
    def tau_s(t: np.ndarray, R: np.ndarray, R_inf: Optional[float] = None) -> float:
        """
        τ_s: Signed centroid timescale.

        τ_s = (1/R∞) ∫₀^∞ t·dR/dt dt
        """
        t = np.asarray(t, dtype=float)
        R = np.asarray(R, dtype=float)

        if R_inf is None:
            R_inf = R[-1]

        dt = np.mean(np.diff(t))
        dR = np.gradient(R, dt)

        integral = np.trapezoid(t * dR, t)

        return integral / R_inf

    @staticmethod
    def tau_u(t: np.ndarray, R: np.ndarray, R_inf: Optional[float] = None) -> float:
        """
        τ_u: Unsigned centroid timescale.

        τ_u = ∫₀^∞ t·|dR/dt| dt / ∫₀^∞ |dR/dt| dt
        """
        t = np.asarray(t, dtype=float)
        R = np.asarray(R, dtype=float)

        if R_inf is None:
            R_inf = R[-1]

        dt = np.mean(np.diff(t))
        dR = np.gradient(R, dt)

        numerator = np.trapezoid(t * np.abs(dR), t)
        denominator = np.trapezoid(np.abs(dR), t)

        return numerator / denominator if denominator != 0 else np.nan

    @staticmethod
    def tau_2(t: np.ndarray, R: np.ndarray, R_inf: Optional[float] = None) -> float:
        """
        τ²: Step-response timescale.

        τ² = ∫₀^∞ [1 - R(t)/R∞] dt
        """
        t = np.asarray(t, dtype=float)
        R = np.asarray(R, dtype=float)

        if R_inf is None:
            R_inf = R[-1]

        f = 1 - R / R_inf
        return np.trapezoid(f, t)

    @staticmethod
    def tau_3(
        t: np.ndarray,
        R: np.ndarray,
        method: Literal['autocorrelation', 'impulse'] = 'autocorrelation'
    ) -> float:
        """
        τ³: Timescale estimator.

        - autocorrelation: τ³ = ∫₀^∞ C_RR(τ)/C_RR(0) dτ
        - impulse: τ³ = 1/(2∫₀^∞ g²(t) dt)
        """
        t = np.asarray(t, dtype=float)

        if method == 'autocorrelation':
            R = np.asarray(R, dtype=float)
            dt = t[1] - t[0]

            R_centered = R - np.mean(R)
            n = len(R_centered)

            autocorr = np.correlate(R_centered, R_centered, mode='full')[n - 1:]
            autocorr = autocorr / (n - np.arange(n))
            autocorr = autocorr / autocorr[0]

            idx = np.where(autocorr < 0.05)[0]
            cutoff = idx[0] if len(idx) > 0 else len(autocorr) // 2

            return np.trapezoid(autocorr[:cutoff], dx=dt)

        elif method == 'impulse':
            g = np.asarray(R, dtype=float)

            if np.all(g == 0):
                return np.nan

            integral_g2 = np.trapezoid(g**2, t)
            if integral_g2 == 0:
                return np.nan

            return 1.0 / (2.0 * integral_g2)

        else:
            raise ValueError(f"Method must be 'autocorrelation' or 'impulse', got {method}")

    @staticmethod
    def tau_pole(t: np.ndarray, R: np.ndarray, R_inf: Optional[float] = None) -> float:
        """
        τ_pole: Spectral pole timescale.

        τ_pole = |R̃'(0)/R̃(0)|
        """
        t = np.asarray(t, dtype=float)
        R = np.asarray(R, dtype=float)

        if R_inf is None:
            R_inf = R[-1]

        dt = np.mean(np.diff(t))
        dR = np.gradient(R, dt)

        R_tilde_0 = R_inf
        R_tilde_prime_0 = -np.trapezoid(t * dR, t)

        return np.abs(R_tilde_prime_0 / R_tilde_0)

    # ============================================================================
    # ENVELOPE ESTIMATOR (1 function, 2 methods)
    # ============================================================================

    @staticmethod
    def tau_env(
        t: np.ndarray,
        R: np.ndarray,
        method: Literal['hilbert', 'peak'] = 'hilbert'
    ) -> float:
        """
        τ_env²: Envelope decay timescale.
        """
        t = np.asarray(t, dtype=float)
        R = np.asarray(R, dtype=float)

        if method == 'hilbert':
            dt = t[1] - t[0]
            dR = np.gradient(R, dt)

            analytic = hilbert(dR)
            envelope = np.abs(analytic)

            E0 = envelope[0] if envelope[0] > 0 else np.max(envelope)
            idx = np.where(envelope / E0 < 0.05)[0]
            cutoff = idx[0] if len(idx) > 0 else len(t)

            return np.trapezoid(envelope[:cutoff] / E0, t[:cutoff])

        elif method == 'peak':
            peaks = []
            for i in range(1, len(R) - 1):
                if abs(R[i]) > abs(R[i - 1]) and abs(R[i]) > abs(R[i + 1]):
                    peaks.append((t[i], abs(R[i])))

            if len(peaks) < 3:
                return np.nan

            t_peaks = np.array([p[0] for p in peaks])
            R_peaks = np.array([p[1] for p in peaks])

            coeff = np.polyfit(t_peaks, np.log(R_peaks), 1)
            alpha = -coeff[0]

            return 1 / alpha if alpha > 0 else np.nan

        else:
            raise ValueError(f"Method must be 'hilbert' or 'peak', got {method}")

    # ============================================================================
    # ENERGY ESTIMATOR (1 function, 2 methods)
    # ============================================================================

    @staticmethod
    def tau_E(
        t: np.ndarray,
        R: np.ndarray,
        method: Literal['autocorrelation', 'peak'] = 'autocorrelation',
        omega0: Optional[float] = None
    ) -> float:
        """
        τ_E³: Energy decay timescale.
        """
        t = np.asarray(t, dtype=float)
        R = np.asarray(R, dtype=float)
        dt = t[1] - t[0]

        if method == 'autocorrelation':
            dR = np.gradient(R, dt)

            if omega0 is None:
                freqs, psd = welch(R, fs=1 / dt, nperseg=min(1024, len(R)))
                omega0 = 2 * np.pi * freqs[np.argmax(psd[1:]) + 1]

            E = 0.5 * dR**2 + 0.5 * omega0**2 * R**2
            E_centered = E - np.mean(E)
            n = len(E_centered)

            autocorr = np.correlate(E_centered, E_centered, mode='full')[n - 1:]
            autocorr = autocorr / (n - np.arange(n))
            autocorr = autocorr / autocorr[0]

            idx = np.where(autocorr < 0.05)[0]
            cutoff = idx[0] if len(idx) > 0 else n // 2

            return np.trapezoid(autocorr[:cutoff], dx=dt)

        elif method == 'peak':
            dR = np.gradient(R, dt)

            peaks = []
            for i in range(1, len(R) - 1):
                if R[i] > R[i - 1] and R[i] > R[i + 1]:
                    peaks.append(t[i])

            if len(peaks) > 2:
                T = np.mean(np.diff(peaks[:3]))
                omega0 = 2 * np.pi / T if T > 0 else 1.0
            else:
                omega0 = 1.0

            E = 0.5 * dR**2 + 0.5 * omega0**2 * R**2

            E_peaks = []
            for i in range(1, len(E) - 1):
                if E[i] > E[i - 1] and E[i] > E[i + 1]:
                    E_peaks.append((t[i], E[i]))

            if len(E_peaks) < 3:
                return np.nan

            t_peaks = np.array([p[0] for p in E_peaks])
            E_vals = np.array([p[1] for p in E_peaks])

            coeff = np.polyfit(t_peaks, np.log(E_vals), 1, w=np.sqrt(E_vals))
            decay_rate = -coeff[0]

            return 1 / decay_rate if decay_rate > 0 else np.nan

        else:
            raise ValueError(f"Method must be 'autocorrelation' or 'peak', got {method}")

    # ============================================================================
    # DIAGNOSTIC RATIOS (4 essential functions)
    # ============================================================================

    @staticmethod
    def Delta_su(
        t: np.ndarray,
        R: np.ndarray,
        R_inf: Optional[float] = None,
        ensure_proper_window: bool = True,
        verbose: bool = False
    ) -> float:
        """
        Δₛᵤ = (τ_s - τ_u) / τ_u
        """
        tau_s_val = QSignature.tau_s(t, R, R_inf)
        tau_u_val = QSignature.tau_u(t, R, R_inf)

        if tau_u_val == 0 or not np.isfinite(tau_s_val) or not np.isfinite(tau_u_val):
            return np.nan

        if ensure_proper_window and tau_s_val < 0 and verbose:
            print(f"Warning: τ_s = {tau_s_val:.3f} < 0. Time window may be too short.")
            print(f"  For α ≈ {1/tau_u_val:.4f}, need t_max > {10*tau_u_val:.1f}")

        return (tau_s_val - tau_u_val) / tau_u_val

    @staticmethod
    def Delta_23_env(
        t: np.ndarray,
        R: np.ndarray,
        method: Literal['hilbert', 'peak'] = 'hilbert'
    ) -> float:
        """
        Δ₂₃ᵉⁿᵛ = (τ_env² - τ_E³) / τ_env²
        """
        tau_env_val = QSignature.tau_env(t, R, method=method)

        if method == 'hilbert':
            tau_E_val = QSignature.tau_E(t, R, method='autocorrelation')
        else:
            tau_E_val = QSignature.tau_E(t, R, method='peak')

        if tau_env_val == 0 or not np.isfinite(tau_env_val) or not np.isfinite(tau_E_val):
            return np.nan

        ratio = (tau_env_val - tau_E_val) / tau_env_val
        return np.clip(ratio, -1, 1)

    @staticmethod
    def rho_13_step(t: np.ndarray, R: np.ndarray, R_inf: Optional[float] = None) -> float:
        """
        ρ₁₃(step) = τ_s / τ³ (autocorrelation)
        """
        tau_s_val = QSignature.tau_s(t, R, R_inf)
        tau_3_val = QSignature.tau_3(t, R, method='autocorrelation')

        if tau_3_val == 0 or not np.isfinite(tau_s_val) or not np.isfinite(tau_3_val):
            return np.nan

        return tau_s_val / tau_3_val

    @staticmethod
    def rho_13_impulse(t: np.ndarray, g: np.ndarray) -> float:
        """
        ρ(impulse) = τ_g_s / τ³ (impulse version)
        """
        t = np.asarray(t, dtype=float)
        g = np.asarray(g, dtype=float)

        if np.all(g == 0):
            return np.nan

        H = np.trapezoid(g, t)
        if H == 0:
            return np.nan

        tau_gs = np.trapezoid(t * g, t) / H
        tau_3_val = QSignature.tau_3(t, g, method='impulse')

        if tau_3_val == 0 or not np.isfinite(tau_gs) or not np.isfinite(tau_3_val):
            return np.nan

        return tau_gs / tau_3_val

    # ============================================================================
    # MAIN COMPUTATION FUNCTION (1 unified function)
    # ============================================================================

    @staticmethod
    def compute_all(
        t: np.ndarray,
        R: np.ndarray,
        R_inf: Optional[float] = None,
        g: Optional[np.ndarray] = None,
        method: Literal['practical', 'theorem', 'both'] = 'practical',
        verbose: bool = False
    ) -> Dict[str, float]:
        """
        Compute all timescale estimators and diagnostic ratios.
        """
        t = np.asarray(t, dtype=float)
        R = np.asarray(R, dtype=float)

        if R_inf is None:
            R_inf = R[-1]

        results: Dict[str, float] = {}

        # Core estimators
        results['tau_g'] = QSignature.tau_g(t, R, R_inf)
        results['tau_s'] = QSignature.tau_s(t, R, R_inf)
        results['tau_u'] = QSignature.tau_u(t, R, R_inf)
        results['tau_2'] = QSignature.tau_2(t, R, R_inf)
        results['tau_3'] = QSignature.tau_3(t, R, method='autocorrelation')
        results['tau_pole'] = QSignature.tau_pole(t, R, R_inf)

        # Diagnostics
        results['Delta_su'] = QSignature.Delta_su(t, R, R_inf, ensure_proper_window=True, verbose=verbose)
        results['rho_13_step'] = QSignature.rho_13_step(t, R, R_inf)

        # Practical methods
        if method in ['practical', 'both']:
            results['tau_env_practical'] = QSignature.tau_env(t, R, method='hilbert')
            results['tau_E_practical'] = QSignature.tau_E(t, R, method='autocorrelation')
            results['Delta_23_env_practical'] = QSignature.Delta_23_env(t, R, method='hilbert')

        # Theorem methods
        if method in ['theorem', 'both']:
            results['tau_env_theorem'] = QSignature.tau_env(t, R, method='peak')
            results['tau_E_theorem'] = QSignature.tau_E(t, R, method='peak')
            results['Delta_23_env_theorem'] = QSignature.Delta_23_env(t, R, method='peak')

        # Theorem 3.4 impulse-based
        if g is not None:
            g = np.asarray(g, dtype=float)
            H = np.trapezoid(g, t)

            if H != 0:
                results['tau_g_s'] = np.trapezoid(t * g, t) / H
                results['tau_3_impulse'] = QSignature.tau_3(t, g, method='impulse')
                results['rho_13_impulse'] = QSignature.rho_13_impulse(t, g)
                results['H_normalization'] = H

        return results

    # ============================================================================
    # UTILITY FUNCTIONS (2 functions)
    # ============================================================================

    @staticmethod
    def print_summary(results: Dict[str, float], title: str = "τ Computation Results") -> None:
        """
        Print formatted summary of tau computation results.
        """
        print("\n" + "=" * 60)
        print(title)
        print("=" * 60)

        categories = {
            'Core Timescales (τ)': ['tau_g', 'tau_s', 'tau_u', 'tau_2', 'tau_3', 'tau_pole'],
            'Envelope & Energy': ['tau_env_practical', 'tau_E_practical',
                                  'tau_env_theorem', 'tau_E_theorem'],
            'Diagnostic Ratios': ['Delta_su', 'Delta_23_env_practical',
                                  'Delta_23_env_theorem', 'rho_13_step', 'rho_13_impulse'],
            'Theorem 3.4 (Impulse)': ['tau_g_s', 'tau_3_impulse', 'H_normalization']
        }

        for category, keys in categories.items():
            print(f"\n📊 {category}:")
            printed = False
            for key in keys:
                if key in results and results[key] is not None and np.isfinite(results[key]):
                    print(f"  {key:25} = {results[key]:.6f}")
                    printed = True
            if not printed:
                print(f"  (No {category.lower()} available)")

        print("\n" + "=" * 60)

    @staticmethod
    def validate_theorems(
        t: np.ndarray,
        R: np.ndarray,
        R_inf: Optional[float] = None,
        g: Optional[np.ndarray] = None
    ) -> Dict[str, Dict[str, any]]:
        """
        Validate all theorems from the paper.
        """
        results = QSignature.compute_all(t, R, R_inf, g, method='both', verbose=False)

        validation = {}

        # Theorem 3.1: τ_s = τ²
        validation['Theorem 3.1'] = {
            'tau_s': results.get('tau_s', np.nan),
            'tau_2': results.get('tau_2', np.nan),
            'difference': abs(results.get('tau_s', 0) - results.get('tau_2', 0)),
            'relative_error': abs(results.get('tau_s', 1) - results.get('tau_2', 1)) /
                              (results.get('tau_s', 1) + 1e-10) * 100,
            'valid': abs(results.get('tau_s', 0) - results.get('tau_2', 0)) < 1e-4
        }

        # Theorem 3.3: τ_s = τ_pole
        validation['Theorem 3.3'] = {
            'tau_s': results.get('tau_s', np.nan),
            'tau_pole': results.get('tau_pole', np.nan),
            'difference': abs(results.get('tau_s', 0) - results.get('tau_pole', 0)),
            'relative_error': abs(results.get('tau_s', 1) - results.get('tau_pole', 1)) /
                              (results.get('tau_s', 1) + 1e-10) * 100,
            'valid': abs(results.get('tau_s', 0) - results.get('tau_pole', 0)) < 1e-4
        }

        # Theorem 3.4: impulse diagnostics
        if g is not None and 'rho_13_impulse' in results:
            rho = results['rho_13_impulse']
            validation['Theorem 3.4'] = {
                'rho': rho,
                'is_exponential': abs(rho - 1) < 0.1,
                'is_non_exponential': rho > 1.5,
                'valid': np.isfinite(rho)
            }

        # Theorem 3.6: oscillation signature
        delta_su = results.get('Delta_su', np.nan)
        validation['Theorem 3.6'] = {
            'Delta_su': delta_su,
            'is_oscillatory': delta_su < -0.8,
            'is_monotonic': abs(delta_su) < 0.2,
            'valid': np.isfinite(delta_su)
        }

        return validation

    # ============================================================================
    # EXTRA UTILITY: Robust estimation of R_infinity
    # ============================================================================

    @staticmethod
    def estimate_Rinf(t: np.ndarray, R: np.ndarray, method: str = 'auto') -> float:
        """
        Robust estimation of R_infinity.

        Methods:
        - 'integral': ∫ Ṙ dt
        - 'tail'    : mean of last 10%
        - 'last'    : R[-1]
        - 'auto'    : choose tail if oscillatory, else integral
        """
        t = np.asarray(t, dtype=float)
        R = np.asarray(R, dtype=float)

        if method == 'integral':
            dt = np.mean(np.diff(t))
            dR = np.gradient(R, dt)
            return np.trapezoid(dR, t)

        elif method == 'tail':
            tail_len = max(5, len(R) // 10)
            return np.mean(R[-tail_len:])

        elif method == 'last':
            return R[-1]

        elif method == 'auto':
            dt = np.mean(np.diff(t))
            dR = np.gradient(R, dt)
            sign_changes = np.sum((dR[:-1] * dR[1:]) < 0)

            if sign_changes >= 3:
