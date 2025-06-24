import numpy as np
import scipy.signal

def denoise_audio_ml(audio):
    b, a = scipy.signal.butter(6, 0.1, btype='lowpass')
    return scipy.signal.filtfilt(b, a, audio)

def denoise_audio_lms(noisy, mu=0.01, filter_order=8):
    n_samples = len(noisy)
    lms_output = np.zeros(n_samples)
    h = np.zeros(filter_order)
    for n in range(filter_order, n_samples):
        x = noisy[n-filter_order:n][::-1]
        d = noisy[n]
        y = np.dot(h, x)
        e = d - y
        h += 2 * mu * e * x
        lms_output[n] = y
    return lms_output

def hybrid_filter(lms_output, ml_output):
    return 0.5 * (lms_output + ml_output)
