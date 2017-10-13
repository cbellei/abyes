import numpy as np
from scipy.stats import beta
from .utils import check_size, print_result
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline
import pymc3 as pm


class AbExp:
    """
    Define a Bayesian A/B Test on conversion rate experimental data.
    Parameters
    ----------
    method : `str`
        choose method for analysis (options: 'analytic', 'numerical', 'compare')
        [default: 'analytic']
    rule : `str`
        choose decision rule (options: 'rope', 'loss')
        [default: 'rope']
    alpha : `float`
        alpha parameter for rope calculation [default: 0.95]
    alpha_prior : `float`
        alpha parameter for the prior (beta distribution)
        [default: 1]
    beta_prior : `float`
        beta parameter for the prior (beta distribution)
        [default: 1]
    rope : `tuple(float, float)`
        define region of practical equivalence
        [default: (-0.1, 0.1)]
    toc : `float`
        define threshold of caring
        [default: 0.01]
    """
    def __init__(self, method='analytic', rule='rope',
                 alpha=0.95, alpha_prior=1, beta_prior=1,
                 resolution=500, rope=(-0.1, 0.1), toc=1.e-2):
        self.method = method
        self.rule = rule
        self.alpha = alpha
        self.alpha_prior = alpha_prior
        self.beta_prior = beta_prior
        self.resolution = resolution
        self.rope = rope
        self.toc = toc
        self.data = np.array([])

    def experiment(self, data):
        """
        Run experiment with data provided
        Parameters
        ----------
        data : `List(np.array, np.array)`
        """
        self.data = np.asarray(data, dtype=float)
        check_size(self.data, dim=2)

        posterior = self.find_posterior()
        return self.decision(posterior)

    def find_posterior(self):
        """
        Find posterior distribution
        """
        if self.method == 'analytic':
            posterior = self.posterior_analytic()
        elif self.method == 'mcmc':
            posterior = self.posterior_mcmc()
        elif self.method == 'compare':
            posterior = [self.posterior_mcmc(), self.posterior_analytic()]
        else:
            raise Exception('method not recognized')

        return posterior

    def decision(self, posterior):
        """
        Make decision on the experiment
        :param posterior:
        :return:
        """
        if self.rule=='rope':
            hpd = self.HPD(posterior)
            result = self.rope_decision(hpd)
        elif self.rule=='loss':
            self.HPD(posterior)
            result = self.expected_loss_decision(posterior)
        else:
            hpd1 = self.HPD(posterior[0])
            result1 = self.rope_decision(hpd1)
            hpd2 = self.HPD(posterior[1])
            result2 = self.rope_decision(hpd2)
            result = [result1, result2]

        return print_result(result)


    def posterior_analytic(self):
        """
        Find posterior distribution for the analytic method of solution
        :return:
        """

        cA = np.sum(self.data[0])
        nA = len(self.data[0])

        cB = np.sum(self.data[1])
        nB = len(self.data[1])

        #find posterior of A and B from analytic solution
        x = np.linspace(0, 1, self.resolution-1)
        dx = x[1] - x[0]
        pA = (np.array([beta.pdf(xx, self.alpha_prior + cA, self.beta_prior + nA - cA) for xx in x]),
                np.append(x,x[-1]+dx) - 0.5*dx)
        pB = (np.array([beta.pdf(xx, self.alpha_prior + cB, self.beta_prior + nB - cB) for xx in x]),
                np.append(x, x[-1] + dx) - 0.5 * dx)

        #bootstrapping now
        A_rvs = beta.rvs(self.alpha_prior + cA, self.beta_prior + nA - cA, size=400*self.resolution)
        B_rvs = beta.rvs(self.alpha_prior + cB, self.beta_prior + nB - cB, size=400*self.resolution)

        rvs = B_rvs - A_rvs
        bins = np.linspace(np.min(rvs) - 0.2 * abs(np.min(rvs)), 1.2*np.max(rvs), self.resolution)
        delta = np.histogram(rvs, bins=bins, normed=True)

        bins = np.linspace(0, 1, self.resolution)
        sigmaA_rvs = np.sqrt(A_rvs * (1 - A_rvs))
        sigmaB_rvs = np.sqrt(B_rvs * (1 - B_rvs))
        psigmaA = np.histogram(sigmaA_rvs, bins=bins, normed=True)
        psigmaB = np.histogram(sigmaB_rvs, bins=bins, normed=True)

        rvs = (B_rvs - A_rvs) / np.sqrt(0.5 * (sigmaA_rvs**2 + sigmaB_rvs**2))
        bins = np.linspace(np.min(rvs) - 0.2 * abs(np.min(rvs)), 1.2*np.max(rvs), self.resolution)
        pES = np.histogram(rvs, bins=bins, normed=True)

        posterior = {'pA': pA, 'pB': pB, 'psigmaA': psigmaA, 'psigmaB': psigmaB,
                     'delta': delta, 'pES': pES, 'prior': self.prior()}

        return posterior

    def posterior_mcmc(self):
        """
        Find posterior distribution for the numerical method of solution
        :return:
        """

        with pm.Model() as ab_model:
            # priors
            pA = pm.distributions.continuous.Beta('pA', alpha=self.alpha_prior, beta=self.beta_prior)
            pB = pm.distributions.continuous.Beta('pB', alpha=self.alpha_prior, beta=self.beta_prior)
            # likelihoods
            pm.Bernoulli('likelihood_A', pA, observed=self.data[0])
            pm.Bernoulli('likelihood_B', pB, observed=self.data[1])

            # find distribution of difference
            pm.Deterministic('delta', pA - pB)
            # find distribution of effect size
            sigmaA = pm.Deterministic('sigmaA', np.sqrt(pA * (1 - pA)))
            sigmaB = pm.Deterministic('sigmaB', np.sqrt(pB * (1 - pB)))
            pm.Deterministic('effect_size', (pB - pA) / (np.sqrt(0.5 * (sigmaA ** 2 + sigmaB ** 2))))

            start = pm.find_MAP()
            step = pm.NUTS(state=start)
            trace = pm.sample(10000, step, start=start)

        bins = np.linspace(0, 1, self.resolution)
        pA = np.histogram(trace['pA'][500:], bins=bins, normed=True)
        pB = np.histogram(trace['pB'][500:], bins=bins, normed=True)
        psigmaA = np.histogram(trace['sigmaA'][500:], bins=bins, normed=True)
        psigmaB = np.histogram(trace['sigmaB'][500:], bins=bins, normed=True)

        rvs = trace['delta'][500:]
        bins = np.linspace(np.min(rvs) - 0.2 * abs(np.min(rvs)), 1.2*np.max(rvs), self.resolution)
        delta = np.histogram(rvs, bins=bins, normed=True)

        rvs = trace['effect_size'][500:]
        bins = np.linspace(np.min(rvs) - 0.2 * abs(np.min(rvs)), 1.2*np.max(rvs), self.resolution)
        pES = np.histogram(rvs, bins=bins, normed=True)

        posterior = {'pA': pA, 'pB': pB, 'psigmaA': psigmaA, 'psigmaB': psigmaB,
                     'delta': delta, 'pES': pES, 'prior': self.prior()}

        return posterior

    def prior(self):
        """
        Find out prior distribution
        :return:
        """
        return [beta.pdf(x, self.alpha_prior, self.beta_prior) for x in np.linspace(0, 1, self.resolution)]

    def HPD(self, posterior):
        """
        Find out High Posterior Density Region
        :param posterior:
        :return:
        """

        bins = posterior['pES'][1]
        x = 0.5 * (bins[0:-1] + bins[1:])
        pdf = posterior['pES'][0]
        k = np.linspace(0, max(pdf), 1000)
        area_above = np.array([np.trapz(pdf[pdf >= kk], x[pdf >= kk]) for kk in k ])
        index = np.argwhere(np.abs(area_above - self.alpha) == np.min(np.abs(area_above - self.alpha)) )[0]

        self.plot_rope_posterior(index, k, x, pdf, posterior)

        return x[pdf >= k[index]]

    def rope_decision(self, hpd):
        """
        Apply decision rule for ROPE method
        :param hpd:
        :return:
        """

        if all(h < min(self.rope) for h in hpd):
            result = -1
        elif all(h > max(self.rope) for h in hpd):
            result = 1
        elif all(h >= min(self.rope) and h <= max(self.rope)for h in hpd):
            result = 0
        else:
            result = np.nan

        return result

    def expected_loss_decision(self, posterior):
        """
        Calculate expected loss and apply decision rule
        :param posterior:
        :return:
        """

        dl = posterior['delta'][1]
        dl = 0.5 * (dl[0:-1] + dl[1:])
        fdl = posterior['delta'][0]
        intA = np.maximum(-dl, 0) * fdl
        intB = np.maximum(dl, 0) * fdl

        elA = np.trapz(intA, dl)
        elB = np.trapz(intB, dl)

        plt.plot(dl, fdl, dl, intA, dl, intB)
        plt.plot([elA, elA],[0,1],'g')
        plt.plot([elB, elB],[0,1],'r')

        if elA <= self.toc and elB <= self.toc:
            result = 0
        elif elA < self.toc:
            result = 1
        elif elB < self.toc:
            result = -1
        else:
            result = np.nan

        print(elA, elB, self.toc)
        return result

    def plot_rope_posterior(self, index, k, x, pdf, posterior):

        plt.plot(x[pdf >= k[index]], 0 * x[pdf >= k[index]], linewidth=4)
        plt.plot(x, pdf, linewidth=4)
        b = posterior['pA'][1]
        plt.plot(0.5*(b[0:-1]+b[1:]), posterior['pA'][0])
        b = posterior['pB'][1]
        plt.plot(0.5*(b[0:-1]+b[1:]), posterior['pB'][0])
        #s = UnivariateSpline(x, pdf, s=1, k=2)
        #ys = s(x)
        #plt.plot(x, ys, 'k', lw=5)
        plt.xlim([np.minimum(np.min(x),-1), np.maximum(1,np.max(x))])
        plt.plot([self.rope[0], self.rope[0]], [0, 4], 'g--', linewidth=5)
        plt.plot([self.rope[1], self.rope[1]], [0, 4], 'g--', linewidth=5)

# exp = AbExp(alpha=0.95, method='analytic', rule='loss')
# data = [np.random.binomial(1, 0.5, size=10000), np.random.binomial(1, 0.5, size=10000)]
# exp.experiment(data)
# plt.show()










