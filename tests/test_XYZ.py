import unittest

from XYZ import function1, function2


class TestXYZ(unittest.TestCase):
    def test_function1(self):
        # Test case 1
        self.assertEqual(function1(5), 25)
        # Test case 2
        self.assertEqual(function1(0), 0)
        # Test case 3
        self.assertEqual(function1(-5), 25)

    def test_function2(self):
        # Test case 1
        self.assertEqual(function2(10), 100)
        # Test case 2
        self.assertEqual(function2(2), 4)
        # Test case 3
        self.assertEqual(function2(-3), 9)

if __name__ == '__main__':
    unittest.main()
