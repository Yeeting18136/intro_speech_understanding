import unittest, homework13, librosa
import numpy as np

# TestSequence
class Test(unittest.TestCase):
    def test_lpc(self):
        speech, fs = librosa.load('speech_waveform.wav',sr=8000)
        frame_length = int(fs*0.025)
        frame_skip = int(fs*0.01)
        order = 10
        A, excitation = homework13.lpc(speech, frame_length, frame_skip, order)
        self.assertEqual(len(A.shape), 2, 'A should be a matrix')
        nframes = int((len(speech)-frame_length)/frame_skip)
        self.assertEqual(A.shape[0],nframes,'A should have %d rows'%(nframes))
        self.assertEqual(A.shape[1],order+1,'A should have %d columns'%(order+1))
        self.assertEqual(len(excitation.shape),2,'excitation should be a matrix')
        self.assertEqual(excitation.shape[0],nframes,'excitation should have %d rows'%(nframes))
        self.assertEqual(excitation.shape[1],frame_length,'excitation should have %d cols'%(frame_length))
        
    def test_synthesize(self):
        speech, fs = librosa.load('speech_waveform.wav',sr=8000)
        frame_length = int(fs*0.025)
        frame_skip = int(fs*0.01)
        order = 10
        A, excitation = homework13.lpc(speech, frame_length, frame_skip, order)
        e = np.hstack(excitation[:,(frame_length-frame_skip):frame_length])
        synthesis = homework13.synthesize(e, A, frame_skip)
        self.assertEqual(len(synthesis.shape), 1, 'synthesis should be a 1d array')
        self.assertEqual(len(synthesis), frame_skip*len(A),
                         'synthesis should have %d samples'%(frame_skip*len(A)))
        self.assertGreater(np.average(np.square(synthesis)),np.average(np.square(np.abs(e))),
                           'synthesis power should be greater than excitation power')
        
    def test_robot_voice(self):
        speech, fs = librosa.load('speech_waveform.wav',sr=8000)
        frame_length = int(fs*0.025)
        frame_skip = int(fs*0.01)
        nframes = int((len(speech)-frame_length)/frame_skip)
        order = 10
        A, excitation = homework13.lpc(speech, frame_length, frame_skip, order)
        T0 = int(fs/100)
        gain, e_robot = homework13.robot_voice(excitation, T0, frame_skip)
        self.assertEqual(len(gain.shape), 1, 'gain should be a 1d array')
        self.assertEqual(len(gain), nframes, 'len(gain) should be nframes')
        self.assertEqual(len(e_robot.shape), 1, 'e_robot should be a 1d array')
        self.assertEqual(len(e_robot), frame_skip*nframes, 'len(e_robot) should be nframes*frame_skip')
        

suite = unittest.defaultTestLoader.loadTestsFromTestCase(Test)
result = unittest.TextTestRunner().run(suite)

n_success = result.testsRun - len(result.errors) - len(result.failures)
print('%d successes out of %d tests run'%(n_success, result.testsRun))
print('Score: %d%%'%(int(100*(n_success/result.testsRun))))
