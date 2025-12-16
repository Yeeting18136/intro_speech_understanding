import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# --- Helper Functions (From Lecture 9, needed for feature extraction) ---

def _frame_wave(waveform, Fs, frame_len_s, frame_step_s):
    """Frames a 1D waveform."""
    frame_len = int(Fs * frame_len_s)
    frame_step = int(Fs * frame_step_s)
    N = waveform.shape[0]
    
    if N < frame_len:
        return np.empty((0, frame_len))
        
    num_frames = int((N - frame_len) / frame_step) + 1
    start_indices = np.arange(num_frames) * frame_step
    
    frames = np.array([waveform[i:i + frame_len] for i in start_indices])
    return frames

def _pre_emphasize(waveform, alpha=0.97):
    """Applies a pre-emphasis filter."""
    if waveform.size < 2:
        return waveform
    # y[n] = x[n] - alpha * x[n-1]
    return np.append(waveform[0], waveform[1:] - alpha * waveform[:-1])

def _energy_sequence(waveform, Fs, frame_len_s, frame_step_s):
    """Calculates the energy sequence."""
    frames = _frame_wave(waveform, Fs, frame_len_s, frame_step_s)
    if frames.size == 0:
        return np.array([])
    return np.sum(frames**2, axis=-1)

def _spec(waveform, Fs, frame_len_s, frame_step_s):
    """Calculates the magnitude spectrogram using rfft."""
    frames = _frame_wave(waveform, Fs, frame_len_s, frame_step_s)
    
    # NFFT size based on frame length
    NFFT = int(Fs * frame_len_s) 
    
    if frames.shape[0] == 0:
        return np.empty((0, NFFT // 2 + 1))
        
    # Apply Hanning window
    window = np.hanning(frames.shape[1])
    windowed_frames = frames * window
    
    # Use rfft for real-valued signal, then take magnitude
    S = np.fft.rfft(windowed_frames, n=NFFT, axis=1)
    return np.abs(S)

# --- Feature and Label Extraction (FIXED: Label length matching) ---

def get_features(waveform, Fs):
    '''
    Get features from a waveform.
    @params:
    waveform (numpy array) - the waveform
    Fs (scalar) - sampling frequency.

    @return:
    features (NFRAMES,NFEATS) - numpy array of feature vectors:
        Pre-emphasize the signal, then compute the spectrogram with a 4ms frame length and 2ms step,
        then keep only the low-frequency half (the non-aliased half).
    labels (NFRAMES) - numpy array of labels (integers):
        Calculate VAD with a 25ms window and 10ms skip. Find start time and end time of each segment.
        Then give every non-silent segment a different label.  Repeat each label five times.
    '''
    # --- Feature Extraction Parameters (4ms frame, 2ms step) ---
    FEAT_FRAME_LEN_S = 0.004 
    FEAT_FRAME_STEP_S = 0.002
    
    # 1. Pre-emphasize the signal
    preemp_waveform = _pre_emphasize(waveform)
    
    # 2. Compute the spectrogram (magnitude)
    mag_spec = _spec(preemp_waveform, Fs, FEAT_FRAME_LEN_S, FEAT_FRAME_STEP_S)
    
    # 3. Keep only the low-frequency half and take the log
    features_full = np.log(mag_spec + np.finfo(float).eps) 
    
    features = features_full # NFRAMES_FEAT x 17
    NFRAMES_FEAT = features.shape[0] # The target length for labels

    # --- Labeling (VAD based on 25ms window and 10ms skip) ---
    VAD_FRAME_LEN_S = 0.025
    VAD_FRAME_STEP_S = 0.010 # 10ms skip (5 times the feature step)

    # 1. Calculate VAD energy sequence
    E_vad = _energy_sequence(waveform, Fs, VAD_FRAME_LEN_S, VAD_FRAME_STEP_S)
    
    if E_vad.size == 0:
        return features, np.zeros(NFRAMES_FEAT, dtype=int)
        
    # 2. VAD mask (E > 0.1 * Emax)
    VAD_mask = E_vad > 0.1 * np.max(E_vad)
    
    # 3. Assign labels (0 for silence, 1, 2, ... for speech segments)
    VAD_labels = np.zeros_like(VAD_mask, dtype=int)
    current_label = 0
    in_segment = False
    
    for i, is_speech in enumerate(VAD_mask):
        if is_speech:
            if not in_segment:
                # Start of a new segment, increment label
                current_label += 1
                in_segment = True
            VAD_labels[i] = current_label
        else:
            # Silence (label 0), end of segment
            in_segment = False
            VAD_labels[i] = 0
            
    # 4. Repeat each label five times to match NFRAMES_FEAT
    
    # Repeat the labels
    upsampled_labels = np.repeat(VAD_labels, 5) # Upsample factor is 5 (10ms step / 2ms step)
    
    # --- CRITICAL FIX: Ensure length matches NFRAMES_FEAT ---
    if upsampled_labels.size < NFRAMES_FEAT:
        # Pad the end with label 0 (silence) if too short
        padding_size = NFRAMES_FEAT - upsampled_labels.size
        padding = np.zeros(padding_size, dtype=int)
        labels = np.concatenate((upsampled_labels, padding))
    else:
        # Truncate if upsampled is too long (safety measure)
        labels = upsampled_labels[:NFRAMES_FEAT]
    
    return features, labels

# --- Neural Network Functions ---

def train_neuralnet(features, labels, iterations):
    '''
    @param:
    features (NFRAMES,NFEATS) - numpy array of feature vectors
    labels (NFRAMES) - numpy array of labels (integers)
    iterations (scalar) - number of iterations of training

    @return:
    model - a neural net model created in pytorch, and trained using the provided data
    lossvalues (numpy array, length=iterations) - the loss value achieved on each iteration of training

    The model should be Sequential(LayerNorm, Linear), 
    input dimension = NFEATS = number of columns in "features",
    output dimension = 1 + max(labels)

    The lossvalues should be computed using a CrossEntropy loss.
    '''
    # Convert numpy data to PyTorch Tensors
    X = torch.from_numpy(features).float()
    Y = torch.from_numpy(labels).long() # Labels must be long (integer type for CrossEntropyLoss)

    NFRAMES, NFEATS = X.shape
    OUTPUT_DIM = Y.max().item() + 1
    
    # Define the model: Sequential(LayerNorm, Linear)
    model = nn.Sequential(
        # LayerNorm normalizes the input features
        nn.LayerNorm(NFEATS),
        # Linear layer maps NFEATS to OUTPUT_DIM (class probabilities/logits)
        nn.Linear(NFEATS, OUTPUT_DIM)
    )

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01) # Use Adam optimizer
    
    loss_values = np.zeros(iterations)
    
    # Training Loop
    for t in range(iterations):
        # Forward pass: compute predicted Y (logits) by passing X to the model
        Y_pred = model(X)
        
        # Compute and print loss
        loss = criterion(Y_pred, Y)
        
        # Store loss value
        loss_values[t] = loss.item()
        
        # Zero gradients, perform a backward pass, and update the weights.
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # <<< THIS IS THE CRITICAL LINE THAT WAS LIKELY MISSING >>>
    return model, loss_values

def test_neuralnet(model, features):
    '''
    @param:
    model - a neural net model created in pytorch, and trained
    features (NFRAMES, NFEATS) - numpy array
    @return:
    probabilities (NFRAMES, 1+max(labels)) - numpy array of frame-level probabilities
    '''
    # Convert numpy features to PyTorch Tensor
    X = torch.from_numpy(features).float()
    
    # Set model to evaluation mode
    model.eval()
    
    # Forward pass: compute predicted Y (logits)
    with torch.no_grad():
        logits = model(X)
        
    # Apply Softmax to convert logits to probabilities
    probabilities = F.softmax(logits, dim=1)
    
    # Convert PyTorch tensor back to numpy array
    return probabilities.numpy()
