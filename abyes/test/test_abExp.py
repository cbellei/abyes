from unittest import TestCase
from abyes.ab_exp import AbExp
from scipy.stats import norm
import numpy as np
import logging
import sys


class TestFunctions(TestCase):

    def test_hpd(self):
        """
        Test that the hpd functions returns the correct interval
        for a normal standard distribution,
        when alpha=68.3%, alpha=95.4% and alpha=99.7%
        :return:
        """
        x = np.linspace(-4, 4, 10000)
        dx = x[1] - x[0]
        y = norm.pdf(x)
        bins = np.append(x - 0.5*dx, x[-1] + dx)
        pdf = (y, bins)
        posterior = dict({'avar': pdf})

        one_sigma = 0.682689492137
        exp1 = AbExp(alpha=one_sigma)
        hpd1 = exp1.hpd(posterior, 'avar')
        min1, max1 = [min(hpd1), max(hpd1)]

        two_sigma = 0.954499736104
        exp2 = AbExp(alpha=two_sigma)
        hpd2 = exp2.hpd(posterior, 'avar')
        min2, max2 = [min(hpd2), max(hpd2)]

        three_sigma = 0.997300203937
        exp3 = AbExp(alpha=three_sigma)
        hpd3 = exp3.hpd(posterior, 'avar')
        min3, max3 = [min(hpd3), max(hpd3)]

        self.assertAlmostEqual(min1, -1.0, places=2)
        self.assertAlmostEqual(max1, 1.0, places=2)

        self.assertAlmostEqual(min2, -2.0, places=2)
        self.assertAlmostEqual(max2, 2.0, places=2)

        self.assertAlmostEqual(min3, -3.0, places=1)
        self.assertAlmostEqual(max3, 3.0, places=1)

    def test_rope_decision(self):
        """
        Test the function rope_decision
        :return:
        """
        exp = AbExp(rope=(-0.1, 0.1))

        result1 = exp.rope_decision([-1., -0.11])
        self.assertEqual(result1, -1.0)

        result2 = exp.rope_decision([-1., -0.10])
        self.assertTrue(result2 != result2)  # np.nan

        result3 = exp.rope_decision([-0.1, 0.1])
        self.assertEqual(result3, 0.0)

        result4 = exp.rope_decision([0.1, 0.2])
        self.assertTrue(result4 != result4)  # np.nan

        result5 = exp.rope_decision([0.11, 0.2])
        self.assertEqual(result5, 1.0)

    def test_expected_loss_decision(self):
        """
        Test expected_loss_decision function using normal standard distribution.
        Should get expected loss = 0.5 * sqrt(2/pi), where sqrt(2/pi) is the expected value of a half-gaussian
        :return:
        """

        x = np.linspace(-4, 4, 10000)
        dx = x[1] - x[0]
        y = norm.pdf(x)
        bins = np.append(x - 0.5 * dx, x[-1] + dx)
        pdf = (y, bins)
        posterior = dict({'avar': pdf})

        exp = AbExp(toc=0.01)
        result1 = exp.expected_loss_decision(posterior, 'avar')
        self.assertTrue(result1 != result1)  # np.nan

        exp = AbExp(toc=0.8)
        result2 = exp.expected_loss_decision(posterior, 'avar')
        self.assertEqual(result2, 0.0)

    def test_analytic_method_it(self):
        """
        Integration test. Verify that the analytic method of solution works,
        with scenarios that effectively at 100% give rise to conclusive results.
        """
        ns = 100000

        pa = 0.8
        pb = 0.2
        data = [np.random.binomial(1, pa, size=ns), np.random.binomial(1, pb, size=ns)]

        exp1 = AbExp(alpha=0.95, method='analytic', rule='rope')
        posterior = exp1.posterior_analytic(data)
        hpd1 = exp1.hpd(posterior, 'pES')
        result1 = exp1.rope_decision(hpd1)

        exp2 = AbExp(alpha=0.95, method='analytic', rule='loss')
        posterior = exp2.posterior_analytic(data)
        result2 = exp2.expected_loss_decision(posterior, 'delta')

        pa = 0.2
        pb = 0.8
        data = [np.random.binomial(1, pa, size=ns), np.random.binomial(1, pb, size=ns)]

        exp3 = AbExp(alpha=0.95, method='analytic', rule='rope')
        posterior = exp3.posterior_analytic(data)
        hpd3 = exp3.hpd(posterior, 'pES')
        result3 = exp3.rope_decision(hpd3)

        exp4 = AbExp(alpha=0.95, method='analytic', rule='loss')
        posterior = exp4.posterior_analytic(data)
        result4 = exp4.expected_loss_decision(posterior, 'delta')

        self.assertEqual(result1, -1.0)
        self.assertEqual(result2, -1.0)
        self.assertEqual(result3,  1.0)
        self.assertEqual(result4,  1.0)

    def test_mcmc_method_it(self):
        """
        Integration test. Verify that the mcmc method of solution works,
        with scenarios that effectively at 100% give rise to conclusive results.
        """
        ns = 10000

        pa = 0.8
        pb = 0.2
        data = [np.random.binomial(1, pa, size=ns), np.random.binomial(1, pb, size=ns)]

        exp1 = AbExp(alpha=0.95, method='mcmc', rule='rope', iterations=2500)
        posterior = exp1.posterior_mcmc(data)
        hpd1 = exp1.hpd(posterior, 'pES')
        result1 = exp1.rope_decision(hpd1)

        exp2 = AbExp(alpha=0.95, method='mcmc', rule='loss', iterations=2500)
        posterior = exp2.posterior_mcmc(data)
        result2 = exp2.expected_loss_decision(posterior, 'delta')

        pa = 0.2
        pb = 0.8
        data = [np.random.binomial(1, pa, size=ns), np.random.binomial(1, pb, size=ns)]

        exp3 = AbExp(alpha=0.95, method='mcmc', rule='rope', iterations=2500)
        posterior = exp1.posterior_mcmc(data)
        hpd3 = exp3.hpd(posterior, 'pES')
        result3 = exp3.rope_decision(hpd3)

        exp4 = AbExp(alpha=0.95, method='mcmc', rule='loss', iterations=2500)
        posterior = exp4.posterior_mcmc(data)
        result4 = exp4.expected_loss_decision(posterior, 'delta')

        self.assertEqual(result1, -1.0)
        self.assertEqual(result2, -1.0)
        self.assertEqual(result3,  1.0)
        self.assertEqual(result4,  1.0)


if __name__ == "__main__":
    logging.basicConfig(stream=sys.stderr)
    logging.getLogger("TestFunctions.test_hpd").setLevel(logging.DEBUG)
    unittest.main()
