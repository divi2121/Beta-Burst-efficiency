"""
Lagged coherence of time domain signals.

Author: James Bonaiuto <james.bonaiuto@isc.cnrs.fr>

Adaptation: Sotirios Papadopoulos <sotirios.papadopoulos@univ-lyon1.fr>
"""

import numpy as np
from scipy.signal import hilbert

from joblib import Parallel, delayed


def phase_shuffle(epochs, n_shuffles=100):
    
    # Create time
    n_channels = epochs.shape[0]
    n_pts = epochs.shape[-1]
    
    # Pre-allocate memory for shuffled_matrix
    shuffled_matrix = np.zeros([n_shuffles, n_channels, n_pts])

    # Fourier transform of matrix (per trial)
    ts_fourier = np.fft.rfft(epochs, axis=-1)
    ts_fourier = np.repeat(ts_fourier[np.newaxis,:], n_shuffles, axis=0)

    # Generate random phases
    random_phases = np.exp(np.random.uniform(0, np.pi, ts_fourier.shape) * 1.0j)

    # Apply random phases to Fourier transform
    ts_fourier_new = ts_fourier * random_phases

    # Inverse Fourier transform to get shuffled matrix
    shuffled_matrix[:,:,:-1] = np.fft.irfft(ts_fourier_new, axis=-1)
    
    return shuffled_matrix


def lagged_continous_surrogate_coherence(epochs, freqs, srate, n_shuffles=100, lag=1, n_jobs=1):
    
    # Number of frequencies
    n_freqs = len(freqs)

    # Results initialization.
    n_trials = epochs.shape[0]
    n_channels = epochs.shape[1]
    n_pts = epochs.shape[-1]
    lcs = np.zeros([n_trials, n_channels, n_freqs, n_pts])

    # Create time
    T = n_pts * 1 / srate
    time = np.linspace(0, T, int(T * srate))

    # Frequency resolution
    df = np.diff(freqs)[0]
    sigma = df * .5

    #for t in range(n_trials):

    def do_trial(epochs, t, lcs, n_channels, n_pts, T, time, sigma, n_shuffles=10):
        # Phase shuffled signal.
        rand_signal = phase_shuffle(epochs[t,:,:], n_shuffles=n_shuffles)

        padd_rand_signal = []
        for c in range(n_channels):
            padd_rand_signal.append(np.hstack([np.zeros((n_shuffles, n_pts)), rand_signal[:,c,:], np.zeros((n_shuffles, n_pts))]))
        padd_rand_signal = np.rollaxis(np.array(padd_rand_signal), 1, 0)
        
        # Get analytic signal (phase and amplitude)
        analytic_signal = hilbert(padd_rand_signal, N=None, axis=-1)[:, :, n_pts:2 * n_pts]

        # Analytic signal at n=0...-1
        f1 = analytic_signal[:, :, 0:-1]
        # Analytic signal at n=1,...
        f2 = analytic_signal[:, :, 1:]

        amp_prod = np.abs(f1) * np.abs(f2)
        thresh = np.percentile(amp_prod, 1, axis=(0,-1))
        
        padd_signal = np.hstack([np.zeros((n_channels, n_pts)), epochs[t,:,:], np.zeros((n_channels, n_pts))])
        signal_fft = np.fft.rfft(padd_signal, axis=-1)
        fft_frex = np.fft.rfftfreq(padd_signal.shape[-1], d=1 / srate)
            
        for f_idx in range(n_freqs):
        
            freq = freqs[f_idx]

            kernel = np.exp(-((fft_frex - freq) ** 2 / (2.0 * sigma ** 2)))

            fsignal_fft = np.multiply(signal_fft, kernel)
            f_signal = np.fft.irfft(fsignal_fft, axis=-1)
            
            # Get analytic signal (phase and amplitude)
            analytic_signal = hilbert(f_signal, N=None, axis=-1)[:, n_pts:2 * n_pts]

            # Duration of this lag in pts
            lag_dur_s = (lag * 1 / freq)

            n_evals = int(np.floor(T / lag_dur_s)) - 1
            end_time = n_evals * lag_dur_s
            end_time_pt = np.argmin(np.abs(time - end_time))

            for pt in range(end_time_pt):
                
                next_time_pt = np.argmin(np.abs(time-(time[pt]+lag_dur_s)))

                # Analytic signal at n=0...-1
                f1 = analytic_signal[:, pt]
                
                # Analytic signal at n=1,...
                f2 = analytic_signal[:, next_time_pt]

                # Phase difference between time points
                phase_diff = np.angle(f2) - np.angle(f1)
                
                # Product of amplitudes at two time points
                amp_prod = np.abs(f1) * np.abs(f2)
                
                # Numerator 
                num = np.abs(amp_prod * np.exp(complex(0, 1) * phase_diff))
                
                # denominator is scaling factor
                denom = np.sqrt(np.abs(np.power(f1, 2)) * np.abs(np.power(f2, 2)))
                
                # Apply threshold
                valid = np.where(denom > thresh)[0]
                lcs[t,valid,f_idx,pt] = num[valid] / denom[valid]
    
    # Parallel processing.
    Parallel(n_jobs=n_jobs, require="sharedmem")(
        delayed(do_trial)(epochs, t, lcs, n_channels, n_pts, T, time, sigma, n_shuffles=n_shuffles)
        for t in range(n_trials)
    )

    return lcs