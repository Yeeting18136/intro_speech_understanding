import unittest, homework8
import numpy as np
import librosa

speech, Fs = librosa.load('train.m4a', sr=8000)

# TestSequence
class Test(unittest.TestCase):
    def test_waveform_to_frames(self):
        frame_length = int(0.025*Fs)
        step = int(0.01*Fs)
        frames = homework8.waveform_to_frames(speech, frame_length, step)
        num_frames = int((len(speech)-frame_length)/step)
        self.assertEqual(len(frames.shape), 2, 'waveform_to_frames should return a matrix')
        self.assertEqual(frames.shape[1],frame_length,'waveform_to_frames # columns should be frame_length')
        self.assertGreaterEqual(frames.shape[0],num_frames,
                                '''
                                waveform_to_frames # rows should be >=
                                (len(waveform)-frame_length)/step
                                ''')
        self.assertEqual(frames[5,3], speech[5*step + 3],
                         'waveform_to_frames[3,5] should be waveform[5*step+3]')

    def test_frames_to_mstft(self):
        frame_length = int(0.025*Fs)
        step = int(0.01*Fs)
        frames = homework8.waveform_to_frames(speech, frame_length, step)
        mstft = homework8.frames_to_mstft(frames)
        
        self.assertEqual(len(mstft.shape), 2, 'frames_to_mstft should return a matrix')
        self.assertEqual(mstft.shape[0],frames.shape[0],
                         'frames_to_mstft input and output should be same size')
        self.assertEqual(mstft.shape[1],frames.shape[1],
                         'frames_to_mstft input and output should be same size')
        ref = np.abs(np.fft.fft(frames,axis=1))
        self.assertAlmostEqual(np.sum(ref), np.sum(mstft), places=1,
                               msg="frames_to_stft output doesn't match correct answer")
        
    def test_mstft_to_spectrogram(self):
        frame_length = int(0.025*Fs)
        step = int(0.01*Fs)
        frames = homework8.waveform_to_frames(speech, frame_length, step)
        mstft = homework8.frames_to_mstft(frames)
        spectrogram = homework8.mstft_to_spectrogram(mstft)
        self.assertEqual(len(spectrogram.shape),2,'stft_to_spectrogram should return a matrix')
        self.assertEqual(mstft.shape[0],spectrogram.shape[0],
                         'mstft_to_spectrogram input and output should be same size')
        self.assertEqual(mstft.shape[1],spectrogram.shape[1],
                         'mstft_to_spectrogram input and output should be same size')
        self.assertGreaterEqual(np.amin(spectrogram),np.amax(spectrogram)-60,msg='np.amin(spectrogram) should be >=np.amax(spectrogram)-60')
        ref = 20*np.log10(np.maximum(0.001*np.amax(mstft),mstft))
        self.assertAlmostEqual(np.sum(ref), np.sum(spectrogram), places=1,
                               msg="mstft_to_spectrogram output doesn't match correct answer")
        
suite = unittest.defaultTestLoader.loadTestsFromTestCase(Test)
result = unittest.TextTestRunner().run(suite)

n_success = result.testsRun - len(result.errors) - len(result.failures)
print('%d successes out of %d tests run'%(n_success, result.testsRun))
print('Score: %d%%'%(int(100*(n_success/result.testsRun))))
                                
                    
                        

