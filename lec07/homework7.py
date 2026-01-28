import numpy as np

def major_chord(f, Fs):
    '''
    Generate a one-half-second major chord, based at frequency f, with sampling frequency Fs.

    @param:
    f (scalar): frequency of the root tone, in Hertz
    Fs (scalar): sampling frequency, in samples/second

    @return:
    x (array): a one-half-second waveform containing the chord
    
    A major chord is three notes, played at the same time:
    (1) The root tone (f)
    (2) A major third, i.e., four semitones above f
    (3) A major fifth, i.e., seven semitones above f
    '''
    duration = 0.5
    N = int(Fs * duration)
    t = np.arange(N) / Fs

    f_root  = f
    f_third = f * (2 ** (4/12))
    f_fifth = f * (2 ** (7/12))

    x = ( np.sin(2*np.pi*f_root*t)
        + np.sin(2*np.pi*f_third*t)
        + np.sin(2*np.pi*f_fifth*t) )

    return x

def dft_matrix(N):
    '''
    Create a DFT transform matrix, W, of size N.
    
    @param:
    N (scalar): number of columns in the transform matrix
    
    @result:
    W (NxN array): a matrix of dtype='complex' whose (k,n)^th element is:
           W[k,n] = cos(2*np.pi*k*n/N) - j*sin(2*np.pi*k*n/N)
    '''
    n = np.arange(N)
    k = n.reshape((N,1))
    angle = 2 * np.pi * k * n / N

    W = np.cos(angle) - 1j*np.sin(angle)
    return W.astype(complex)

def spectral_analysis(x, Fs):
    '''
    Find the three loudest frequencies in x.

    @param:
    x (array): the waveform
    Fs (scalar): sampling frequency (samples/second)

    @return:
    f1, f2, f3: The three loudest frequencies (in Hertz)
      These should be sorted so f1 < f2 < f3.
    '''
<<<<<<< HEAD
    N = len(x)
    X = np.fft.fft(x)

    half = N // 2
    mag = np.abs(X[:half])

    idx = np.argsort(mag)[-3:]
    freqs = idx * Fs / N

    freqs_sorted = np.sort(freqs)
    return float(freqs_sorted[0]), float(freqs_sorted[1]), float(freqs_sorted[2])

=======
    raise RuntimeError("You need to write this part")
>>>>>>> release/main
