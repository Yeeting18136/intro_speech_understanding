import numpy as np
import librosa
from scipy.signal import lfilter

def lpc(speech, frame_length, frame_skip, order):
    '''
    Perform linear predictive analysis of input speech.
    
    @param:
    speech (duration) - input speech waveform
    frame_length (scalar) - frame length, in samples
    frame_skip (scalar) - frame skip, in samples
    order (scalar) - number of LPC coefficients to compute
    
    @returns:
    A (nframes,order+1) - linear predictive coefficients from each frames
    excitation (nframes,frame_length) - linear prediction excitation frames
    '''
    # Calculate the number of frames
    nframes = int((len(speech) - frame_length) / frame_skip)
    A = np.zeros((nframes, order + 1))
    excitation = np.zeros((nframes, frame_length))
    
    for t in range(nframes):
        # Frame extraction
        start = t * frame_skip
        end = start + frame_length
        frame = speech[start:end]
        
        # Apply windowing for analysis
        windowed_frame = frame * np.hamming(frame_length)
        
        # Calculate LPC coefficients using librosa (this part is usually stable)
        # Adding a tiny constant prevents errors on silence
        a = librosa.lpc(windowed_frame + 1e-9, order=order)
        A[t, :] = a
        
        # Calculate residual (excitation) using scipy's lfilter
        # The inverse of 1/A is just the FIR filter A
        excitation[t, :] = lfilter(a, [1.0], frame)
        
    return A, excitation

def synthesize(e, A, frame_skip):
    '''
    Synthesize speech from LPC residual and coefficients.
    
    @param:
    e (duration) - excitation signal
    A (nframes,order+1) - linear predictive coefficients from each frames
    frame_skip (1) - frame skip, in samples
    
    @returns:
    synthesis (duration) - synthetic speech waveform
    '''
    nframes = A.shape[0]
    duration = nframes * frame_skip
    synthesis = np.zeros(duration)
    
    # zi stores the filter state to ensure smooth frame transitions
    zi = np.zeros(A.shape[1] - 1)
    
    for t in range(nframes):
        start = t * frame_skip
        end = (t + 1) * frame_skip
        frame_e = e[start:end]
        
        # Synthesis filter: 1/A[z] (All-pole filter)
        frame_s, zi = lfilter([1.0], A[t, :], frame_e, zi=zi)
        
        synthesis[start:end] = frame_s
        
    return synthesis

def robot_voice(excitation, T0, frame_skip):
    '''
    Calculate the gain for each excitation frame, then create the excitation for a robot voice.
    
    @param:
    excitation (nframes,frame_length) - linear prediction excitation frames
    T0 (scalar) - pitch period, in samples
    frame_skip (scalar) - frame skip, in samples
    
    @returns:
    gain (nframes) - gain for each frame
    e_robot (nframes*frame_skip) - excitation for the robot voice
    '''
    nframes = excitation.shape[0]
    gain = np.zeros(nframes)
    e_robot = np.zeros(nframes * frame_skip)
    
    # Create the base robotic source: a steady pulse train
    total_length = nframes * frame_skip
    pulse_train = np.zeros(total_length)
    pulse_train[::T0] = 1.0
    
    for t in range(nframes):
        # Calculate gain (RMS energy) of the original excitation
        gain[t] = np.sqrt(np.mean(np.square(excitation[t, :])))
        
        # Apply gain to the pulse train for the robot excitation
        start = t * frame_skip
        end = (t + 1) * frame_skip
        e_robot[start:end] = gain[t] * pulse_train[start:end]
        
    return gain, e_robot