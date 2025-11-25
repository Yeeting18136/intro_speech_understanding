import unittest, homework7
import numpy as np

# TestSequence
class Test(unittest.TestCase):
    def test_W_is_ndarray(self):
        W = homework7.dft_matrix(30)
        ref = np.zeros((30,30))
        self.assertIs(type(W), type(ref), "dft_matrix should return a numpy array")
        
    def test_W_is_complex(self):
        W = homework7.dft_matrix(30)
        self.assertEqual(W.dtype,complex,'W=dft_matrix(N) should be complex')
        
    def test_W_is_N_by_N(self):
        W = homework7.dft_matrix(30)
        self.assertEqual(len(W),30,'W=dft_matrix(N) should have N rows')
        self.assertEqual(len(W[0]),30,'W=dft_matrix(N) should have N columns')

    def test_W_gives_DFT_of_x(self):
        N = 30
        n = np.arange(N)
        x = np.cos(2*np.pi*3*n/N + np.pi/4)
        ref = np.round(np.abs(np.fft.fft(x))).astype('int')
        W = homework7.dft_matrix(N)
        X = np.round(np.abs(np.matmul(W,x))).astype('int')
        self.assertEqual(X[3],ref[3],'abs(X[3]) should be %d, not %d'%(ref[3],X[3]))


    def test_major_chord(self):
        x = homework7.major_chord(440, 8000)
        self.assertEqual(len(x), 4000, 'major_chord should return a one-half-second array')

    def test_spectral_analysis(self):
        Fs = 8000
        f1ref = 440
        f2ref = 440*np.power(2,4/12)
        f3ref = 440*np.power(2,7/12)
        n = np.arange(Fs)
        x = np.cos(2*np.pi*f1ref*n/Fs)+np.sin(2*np.pi*f2ref*n/Fs)+np.cos(2*np.pi*(f3ref*n/Fs+0.125))
        f1, f2, f3 = homework7.spectral_analysis(x, Fs)
        self.assertEqual(int(np.round(f1)),int(np.round(f1ref)),
                         msg="Lowest frequency is %d, not %d"%(f1ref,f1))
        self.assertEqual(int(np.round(f2)),int(np.round(f2ref)),
                         msg="Second-lowest frequency is %d, not %d"%(f2ref,f2))
        self.assertEqual(int(np.round(f3)),int(np.round(f3ref)),
                         msg="Third-lowest frequency is %d, not %d"%(f3ref,f3))

suite = unittest.defaultTestLoader.loadTestsFromTestCase(Test)
result = unittest.TextTestRunner().run(suite)

n_success = result.testsRun - len(result.errors) - len(result.failures)
print('%d successes out of %d tests run'%(n_success, result.testsRun))
print('Score: %d%%'%(int(100*(n_success/result.testsRun))))
