import unittest, homework2

class Test(unittest.TestCase):
    def test_strstr(self):
        self.assertEqual(homework2.arithmetic("foo","bar"),"foobar")
        
    def test_strfloat(self):
        self.assertEqual(homework2.arithmetic("foo",3.12),"foofoofoo")

    def test_floatstr(self):
        self.assertEqual(homework2.arithmetic(3.12, "foo"), "3.12foo")

    def test_floatfloat(self):
        self.assertAlmostEqual(homework2.arithmetic(3.0, 2.0), 6.0)
                         
        
suite = unittest.defaultTestLoader.loadTestsFromTestCase(Test)
result = unittest.TextTestRunner().run(suite)

n_success = result.testsRun - len(result.errors) - len(result.failures)
print('%d successes out of %d tests run'%(n_success, result.testsRun))
print('Score: %d%%'%(int(100*(n_success/result.testsRun))))
