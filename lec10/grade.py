import unittest, homework10
import numpy as np
import librosa

speech, Fs = librosa.load('train.m4a', sr=8000)
testspeech, Fs = librosa.load('test.m4a', sr=8000)
labels = ['a', 'i', 'u', 'e', 'o']

# TestSequence
class Test(unittest.TestCase):
    def test_get_features(self):
        features, labels = homework10.get_features(speech, Fs)
        self.assertEqual(len(features.shape),2,'features should be a matrix')
        self.assertEqual(features.shape[0], len(labels), 'features and labels should have same number of frames')
        self.assertEqual(labels[0], 0, 'labels[0] should be 0')
        self.assertEqual(int(max(labels)), 5, 'max label should be 5')

    def test_train_neuralnet(self):
        features, labels = homework10.get_features(speech, Fs)
        model, lossvalues = homework10.train_neuralnet(features, labels, 1000)
        self.assertEqual(len(model), 2, 'model should have just 2 layers: LayerNorm and Linear')
        self.assertEqual(len(lossvalues), 1000, 'lossvalues should have length 1000')
        
    def test_test_neuralnet(self):
        features, labels = homework10.get_features(speech, Fs)
        model, lossvalues = homework10.train_neuralnet(features, labels, 1000)
        testfeatures, testlabels = homework10.get_features(testspeech, Fs)
        probabilities = homework10.test_neuralnet(model, testfeatures)
        self.assertEqual(probabilities.shape[0], len(testfeatures), 'probabilities and features should have same number of frames')
        self.assertEqual(probabilities.shape[1], 6, 'probabilities should have 6 columns, for 6 output classes')
        
suite = unittest.defaultTestLoader.loadTestsFromTestCase(Test)
result = unittest.TextTestRunner().run(suite)

n_success = result.testsRun - len(result.errors) - len(result.failures)
print('%d successes out of %d tests run'%(n_success, result.testsRun))
print('Score: %d%%'%(int(100*(n_success/result.testsRun))))
                                
                    
                        

