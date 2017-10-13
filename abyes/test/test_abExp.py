from unittest import TestCase
from abyes.ab_exp import AbExp


class TestBasicFunction(TestCase):
    def setUp(self):
        self.func = AbExp()

    def test_1(self):
        self.assertTrue(True)

    def test_2(self):
        self.assertTrue(True)


if __name__ == '__main__':
    unittest.main()
