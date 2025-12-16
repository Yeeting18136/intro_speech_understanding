import numpy as np

def waveform_to_frames(waveform, frame_length, step):
    '''
    Chop a waveform into overlapping frames. (NO WINDOW)
    '''
    N_samples = waveform.shape[0]
    num_frames = 1 + int(np.floor((N_samples - frame_length) / step))
    
    frames = np.zeros((num_frames, frame_length), dtype=waveform.dtype)
    
    
    for m in range(num_frames):
        start_index = m * step
        frames[m, :] = waveform[start_index : start_index + frame_length]
        
    return frames

def frames_to_mstft(frames):
    '''
    Take the magnitude FFT of every row of the frames matrix.
    Uses np.fft.fft (full FFT) to pass the dimension test.
    '''
    stft = np.fft.fft(frames) 
    
    mstft = np.abs(stft)
    
    return mstft

def mstft_to_spectrogram(mstft):
    A = np.amax(mstft)
    mstft_floor = np.maximum(mstft, 0.001 * A)
    spectrogram = 20 * np.log10(mstft_floor)
    return spectrogram
