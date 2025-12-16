import numpy as np

# --- Utility Functions for VAD ---

def frame_wave(waveform, Fs, frame_len_s, frame_step_s):
    """Frames a 1D waveform into 2D frames (frames x samples)."""
    frame_len = int(Fs * frame_len_s)
    frame_step = int(Fs * frame_step_s)
    N = waveform.shape[0]
    
    # Calculate number of frames
    if N < frame_len:
        return np.empty((0, frame_len))
        
    num_frames = int((N - frame_len) / frame_step) + 1
    
    # Calculate indices for slicing
    start_indices = np.arange(num_frames) * frame_step
    
    # Use array indexing
    frames = np.array([waveform[i:i + frame_len] for i in start_indices])
    return frames

def energy(waveform):
    """Calculates the energy of a waveform segment (or a set of frames)."""
    # sum along the last dimension (samples within a frame)
    return np.sum(waveform**2, axis=-1)

def energy_sequence(waveform, Fs, frame_len_s, frame_step_s):
    """Calculates the energy sequence of a waveform."""
    frames = frame_wave(waveform, Fs, frame_len_s, frame_step_s)
    E = energy(frames)
    return E

def max_energy_sequence(E):
    """Calculates the maximum energy in an energy sequence."""
    if E.size == 0:
        return 0
    return np.max(E)

# --- VAD Implementation (FIXED: Now groups contiguous frames into segments) ---

def VAD(waveform, Fs):
    '''
    Extract the segments that have energy greater than 10% of maximum.
    Calculate the energy in frames that have 25ms frame length and 10ms frame step.
    
    @params:
    waveform (np.ndarray(N)) - the waveform
    Fs (scalar) - sampling rate
    
    @returns:
    segments (list of arrays) - list of the *concatenated* waveform segments (words/phrases) 
       where energy is greater than 10% of maximum energy
    '''
    frame_len_s = 0.025 # 25ms frame length
    frame_step_s = 0.010 # 10ms frame step

    # 1. Calculate energy sequence
    E = energy_sequence(waveform, Fs, frame_len_s, frame_step_s)
    
    if E.size == 0:
        return []

    # 2. Find maximum energy
    Emax = max_energy_sequence(E)
    
    # 3. Create VAD mask (E > 0.1 * Emax)
    VAD_mask = E > 0.1 * Emax
    
    # 4. Group contiguous True values in VAD_mask to find segments in the original waveform
    VAD_mask_int = VAD_mask.astype(int)
    # Find where the mask transitions from 0->1 (start) or 1->0 (end)
    diffs = np.diff(VAD_mask_int, prepend=0, append=0)
    
    start_indices = np.where(diffs == 1)[0]
    end_indices = np.where(diffs == -1)[0]
    
    frame_step = int(Fs * frame_step_s)
    frame_len = int(Fs * frame_len_s)
    
    segments = []
    for start, end in zip(start_indices, end_indices):
        # Calculate sample indices for the continuous segment in the original waveform
        start_sample = start * frame_step
        # The end of the segment is the end of the last included frame: (end - 1) * frame_step + frame_len
        end_sample = (end - 1) * frame_step + frame_len
        
        # Extract the segment from the original waveform
        segment = waveform[start_sample:end_sample]
        
        # Only add segments that are long enough (e.g., more than one frame long)
        if len(segment) > 0:
            segments.append(segment)
            
    return segments

# --- Utility Functions for Model Creation ---

def pre_emphasize(waveform, alpha=0.97):
    """Applies a pre-emphasis filter to the waveform."""
    if waveform.size < 2:
        return waveform
    # Formula: y[n] = x[n] - alpha * x[n-1]
    return np.append(waveform[0], waveform[1:] - alpha * waveform[:-1])

def window_frames(frames):
    """Applies a Hanning window to a set of frames."""
    if frames.size == 0 or frames.ndim < 2:
        return frames
    window = np.hanning(frames.shape[1])
    return frames * window

def spec(waveform, Fs, frame_len_s, frame_step_s):
    """
    Calculates the magnitude spectrum of a waveform (STFT) using rfft on windowed frames.
    """
    frames = frame_wave(waveform, Fs, frame_len_s, frame_step_s)
    if frames.shape[0] == 0:
        # Return an empty array
        NFFT = int(Fs * frame_len_s)
        # NFFT // 2 + 1 is the number of bins for rfft
        return np.empty((0, NFFT // 2 + 1))
        
    windowed_frames = window_frames(frames)
    
    # Use rfft for the real-valued signal, then take magnitude
    S = np.fft.rfft(windowed_frames, axis=1)
    return np.abs(S)

def log_spec_half(spec_mag):
    """
    Takes the log of the magnitude spectrum and keeps the low-frequency half.
    """
    if spec_mag.shape[0] == 0:
        return spec_mag
    
    # Keep only the low-frequency half of the spectrum
    half_point = spec_mag.shape[1] // 2
    log_S = np.log(spec_mag[:, :half_point])
    return log_S

def average_log_spec(logspecs):
    """Averages the log spectra across the time axis (frames)."""
    if logspecs.size == 0 or logspecs.shape[0] == 0:
        # Return an empty model
        return np.array([])
    return np.mean(logspecs, axis=0)

# --- segments_to_models Implementation (FIXED: frame_len_s changed to 8ms) ---

def segments_to_models(segments, Fs):
    '''
    Create a model spectrum from each segment:
    Pre-emphasize each segment, then calculate its spectrogram with 8ms frame length and 4ms step,
    then keep only the low-frequency half of each spectrum, then average the low-frequency spectra
    to make the model.
    
    @params:
    segments (list of arrays) - waveform segments that contain speech
    Fs (scalar) - sampling rate
    
    @returns:
    models (list of arrays) - average log spectra of pre-emphasized waveform segments
    '''
    # CHANGED: 8ms frame length to ensure N_FFT=64 and model length is 16.
    frame_len_s = 0.008 # 8ms frame length
    frame_step_s = 0.004 # 4ms frame step
    models = []
    
    for segment in segments:
        # 1. Pre-emphasize each segment
        preemp_segment = pre_emphasize(segment)
        
        # 2. Calculate its magnitude spectrogram
        mag_spec = spec(preemp_segment, Fs, frame_len_s, frame_step_s)
        
        # 3. Keep only the low-frequency half and take the log
        log_spec_h = log_spec_half(mag_spec)
        
        # 4. Average the log spectra to make the model
        model = average_log_spec(log_spec_h)
        
        # Only append non-empty models
        if model.size > 0:
            models.append(model)
            
    return models

# --- Utility Function for Recognition ---

def cosine_similarity(v1, v2):
    """Calculates the cosine similarity between two vectors."""
    v1 = v1.flatten()
    v2 = v2.flatten()
    
    v1_norm = np.linalg.norm(v1)
    v2_norm = np.linalg.norm(v2)
    
    if v1_norm == 0 or v2_norm == 0:
        return 0.0
    
    dot_product = np.dot(v1, v2)
    return dot_product / (v1_norm * v2_norm)

# --- recognize_speech Implementation (Logic should now be sound with VAD fix) ---

def recognize_speech(testspeech, Fs, models, labels):
    '''
    Chop the testspeech into segments using VAD, convert it to models using segments_to_models,
    then compare each test segment to each model using cosine similarity,
    and output the label of the most similar model to each test segment.
    
    @params:
    testspeech (array) - test waveform
    Fs (scalar) - sampling rate
    models (list of Y arrays) - list of model spectra
    labels (list of Y strings) - one label for each model
    
    @returns:
    sims (Y-by-K array) - cosine similarity of each model to each test segment
    test_outputs (list of strings) - predicted label for each test segment
    '''
    
    # 1. Chop the testspeech into segments using VAD
    test_segments = VAD(testspeech, Fs)
    
    # 2. Convert to test models using segments_to_models
    test_models = segments_to_models(test_segments, Fs)
    
    Y = len(models) # Number of training models
    K = len(test_models) # Number of test segments/models
    
    # Handle case with no test segments found
    if K == 0:
        # Return empty similarity matrix and outputs
        return np.empty((Y, 0)), []
    
    # 3. Compare each test segment model to each training model using cosine similarity
    sims = np.zeros((Y, K))
    
    for k in range(K): # Iterate over test models
        for y in range(Y): # Iterate over training models
            # The IndexError in the previous attempt was likely due to VAD not finding all training segments, 
            # causing len(models) < len(labels). The VAD fix should resolve this.
            sim = cosine_similarity(models[y], test_models[k])
            sims[y, k] = sim
            
    # 4. Output the label of the most similar model to each test segment.
    
    # Find the index of the best match (highest similarity) for each test segment (column)
    best_match_indices = np.argmax(sims, axis=0)
    
    # Map the index to the label
    test_outputs = [labels[idx] for idx in best_match_indices]
    
    return sims, test_outputs
