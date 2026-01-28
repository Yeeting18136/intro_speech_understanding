import unittest, pathlib, homework14, os.path
from contextlib import contextmanager,redirect_stderr,redirect_stdout
from os import devnull

# TestSequence
class Test(unittest.TestCase):
    def test_synthesize_exists(self):
        homework14.synthesize("This is speech synthesis!","en","english.mp3")
        self.assertTrue("homework14.py contains a method called 'synthesize'")
       
    def test_synthesize_creates_file(self):
        homework14.synthesize("This is speech synthesis!","en","english.mp3")
        assert(os.path.isfile("english.mp3"), "english.mp3 was not created")

    def test_make_a_corpus_returns_strings(self):
        recog = homework14.make_a_corpus(['this is a test'],['en'],['testfile'])
        assert(isinstance(recog,list),'make_a_corpus should return a list')
        self.assertEqual(len(recog),1,'make_a_corpus should return a list with one string')
        
suite = unittest.defaultTestLoader.loadTestsFromTestCase(Test)
result = unittest.TextTestRunner().run(suite)

n_success = result.testsRun - len(result.errors) - len(result.failures)
print('%d successes out of %d tests run'%(n_success, result.testsRun))
print('Score: %d%%'%(int(100*(n_success/result.testsRun))))
                                
                    
                        

