import unittest, homework4

# TestSequence
class Test(unittest.TestCase):

    def test_next_birthday(self):
        birthdays = { (1,10):['bob','alice'], (5,30):['david','carol'] }
        birthday, list_of_names = homework4.next_birthday((4,20), birthdays)
        self.assertEqual(birthday, (5,30), 'next birthday after (4,20) should be (5,30)!')
        self.assertEqual(len(list_of_names), 2, 'there should be two people with that birthday!')
        self.assertEqual(list_of_names[0],'david','The first person with that birthday should be david!')

suite = unittest.defaultTestLoader.loadTestsFromTestCase(Test)
result = unittest.TextTestRunner().run(suite)

n_success = result.testsRun - len(result.errors) - len(result.failures)
print('%d successes out of %d tests run'%(n_success, result.testsRun))
print('Score: %d%%'%(int(100*(n_success/result.testsRun))))
