import numpy as np
from scipy.stats import beta
from .utils import check_size, print_result, print_info
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline
import pymc3 as pm


class AbExp:
    """
    Define a Bayesian A/B Test on conversion rate experimental data.
    Parameters
    ----------
    method : `str`
        choose method for analysis (options: 'analytic', 'mcmc', 'compare')
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
                 resolution=500, rope=(-0.1, 0.1), toc=1.e-2, plot=False):
        self.method = method
        self.rule = rule
        self.alpha = alpha
        self.alpha_prior = alpha_prior
        self.beta_prior = beta_prior
        self.resolution = resolution
        self.rope = rope
        self.toc = toc
        self.data = np.array([])
        self.plot = plot

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

        decision = self.decision(posterior)

        if (plt.plot):
            plt.show()

        return decision

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
        if self.rule == 'rope':
            hpd = self.hpd(posterior, 'pES')
            result = self.rope_decision(hpd)
        elif self.rule == 'loss':
            #self.hpd(posterior, 'pES')
            result = self.expected_loss_decision(posterior, 'delta')
        else:
            hpd1 = self.hpd(posterior[0], 'pES')
            result1 = self.rope_decision(hpd1)

            hpd2 = self.hpd(posterior[1], 'pES')
            result2 = self.rope_decision(hpd2)

            result = [result1, result2]

        print_info(self)

        return print_result(result)

    def posterior_analytic(self):
        """
        Find posterior distribution for the analytic method of solution
        :return:
        """

        ca = np.sum(self.data[0])
        na = len(self.data[0])

        cb = np.sum(self.data[1])
        nb = len(self.data[1])

        # find posterior of A and B from analytic solution
        x = np.linspace(0, 1, self.resolution-1)
        dx = x[1] - x[0]
        pa = (np.array([beta.pdf(xx, self.alpha_prior + ca, self.beta_prior + na - ca) for xx in x]),
              np.append(x, x[-1]+dx) - 0.5*dx)
        pb = (np.array([beta.pdf(xx, self.alpha_prior + cb, self.beta_prior + nb - cb) for xx in x]),
              np.append(x, x[-1] + dx) - 0.5 * dx)

        # bootstrapping now
        a_rvs = beta.rvs(self.alpha_prior + ca, self.beta_prior + na - ca, size=400*self.resolution)
        b_rvs = beta.rvs(self.alpha_prior + cb, self.beta_prior + nb - cb, size=400*self.resolution)

        rvs = b_rvs - a_rvs
        bins = np.linspace(np.min(rvs) - 0.2 * abs(np.min(rvs)), np.max(rvs) + 0.2 * abs(np.max(rvs)), self.resolution)
        delta = np.histogram(rvs, bins=bins, normed=True)

        bins = np.linspace(0, 1, self.resolution)
        sigma_a_rvs = np.sqrt(a_rvs * (1 - a_rvs))
        sigma_b_rvs = np.sqrt(b_rvs * (1 - b_rvs))
        psigma_a = np.histogram(sigma_a_rvs, bins=bins, normed=True)
        psigma_b = np.histogram(sigma_b_rvs, bins=bins, normed=True)

        rvs = (b_rvs - a_rvs) / np.sqrt(0.5 * (sigma_a_rvs**2 + sigma_b_rvs**2))
        bins = np.linspace(np.min(rvs) - 0.2 * abs(np.min(rvs)), np.max(rvs) + 0.2 * abs(np.max(rvs)), self.resolution)
        pes = np.histogram(rvs, bins=bins, normed=True)

        posterior = {'muA': pa, 'muB': pb, 'psigma_a': psigma_a, 'psigma_b': psigma_b,
                     'delta': delta, 'pES': pes, 'prior': self.prior()}

        return posterior

    def posterior_mcmc(self):
        """
        Find posterior distribution for the numerical method of solution
        :return:
        """

        with pm.Model() as ab_model:
            # priors
            mua = pm.distributions.continuous.Beta('muA', alpha=self.alpha_prior, beta=self.beta_prior)
            mub = pm.distributions.continuous.Beta('muB', alpha=self.alpha_prior, beta=self.beta_prior)
            # likelihoods
            pm.Bernoulli('likelihoodA', mua, observed=self.data[0])
            pm.Bernoulli('likelihoodB', mub, observed=self.data[1])

            # find distribution of difference
            pm.Deterministic('delta', mua - mub)
            # find distribution of effect size
            sigma_a = pm.Deterministic('sigmaA', np.sqrt(mua * (1 - mua)))
            sigma_b = pm.Deterministic('sigmaB', np.sqrt(mub * (1 - mub)))
            pm.Deterministic('effect_size', (mub - mua) / (np.sqrt(0.5 * (sigma_a ** 2 + sigma_b ** 2))))

            start = pm.find_MAP()
            step = pm.Slice()
            trace = pm.sample(10000, step=step, start=start)

        bins = np.linspace(0, 1, self.resolution)
        mua = np.histogram(trace['muA'][500:], bins=bins, normed=True)
        mub = np.histogram(trace['muB'][500:], bins=bins, normed=True)
        sigma_a = np.histogram(trace['sigmaA'][500:], bins=bins, normed=True)
        sigma_b = np.histogram(trace['sigmaB'][500:], bins=bins, normed=True)

        rvs = trace['delta'][500:]
        bins = np.linspace(np.min(rvs) - 0.2 * abs(np.min(rvs)), 1.2*np.max(rvs), self.resolution)
        delta = np.histogram(rvs, bins=bins, normed=True)

        rvs = trace['effect_size'][500:]
        bins = np.linspace(np.min(rvs) - 0.2 * abs(np.min(rvs)), 1.2*np.max(rvs), self.resolution)
        pes = np.histogram(rvs, bins=bins, normed=True)

        posterior = {'muA': mua, 'muB': mub, 'sigmaA': sigma_a, 'sigmaB': sigma_b,
                     'delta': delta, 'pES': pes, 'prior': self.prior()}

        return posterior

    def prior(self):
        """
        Find out prior distribution
        :return:
        """
        return [beta.pdf(x, self.alpha_prior, self.beta_prior) for x in np.linspace(0, 1, self.resolution)]

    def hpd(self, posterior, var):
        """
        Find out High Posterior Density Region
        """

        bins = posterior[var][1]
        x = 0.5 * (bins[0:-1] + bins[1:])
        pdf = posterior[var][0]
        k = np.linspace(0, max(pdf), 1000)
        area_above = np.array([np.trapz(pdf[pdf >= kk], x[pdf >= kk]) for kk in k])
        index = np.argwhere(np.abs(area_above - self.alpha) == np.min(np.abs(area_above - self.alpha)))[0]

        if self.plot:
            self.plot_rope_posterior(index, k, x, posterior, var)

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
        elif all(min(self.rope) <= h <= max(self.rope) for h in hpd):
            result = 0
        else:
            result = np.nan

        return result

    def expected_loss_decision(self, posterior, var):
        """
        Calculate expected loss and apply decision rule
        """
        dl = posterior[var][1]
        dl = 0.5 * (dl[0:-1] + dl[1:])
        fdl = posterior[var][0]
        inta = np.maximum(dl, 0) * fdl
        intb = np.maximum(-dl, 0) * fdl

        ela = np.trapz(inta, dl)
        elb = np.trapz(intb, dl)

        if self.plot:
            plt.figure(figsize=(11, 7))
            plt.subplot(1,2,1)
            b = posterior['muA'][1]
            plt.plot(0.5*(b[0:-1]+b[1:]), posterior['muA'][0],lw=2,label='A')
            b = posterior['muB'][1]
            plt.plot(0.5*(b[0:-1]+b[1:]), posterior['muB'][0],lw=2,label='B')
            plt.xlabel('$\mu_A,\  \mu_B$')
            plt.title('Conversion Rate')
            plt.locator_params(nticks=6)
            plt.legend()

            plt.subplot(1,2,2)
            plt.plot(dl, fdl, 'b', lw=3, label=r'$\mu_B - \mu_A$')
            plt.plot([ela, ela], [0, 0.3*np.max(fdl)], 'r', lw=3, label='A: Expected Loss')
            plt.plot([elb, elb], [0, 0.3*np.max(fdl)], 'c', lw=3, label='B: Expected Loss')
            plt.plot([self.toc, self.toc], [0, 0.3*np.max(fdl)], 'k--', lw=3, label='Threshold of Caring')
            plt.xlabel(r'$\mu_B-\mu_A$')
            plt.title('Expected Loss')
            plt.gca().locator_params(axis='x', numticks=6)
            plt.legend()

        if ela <= self.toc and elb <= self.toc:
            result = 0
        elif elb < self.toc:
            result = 1
        elif ela < self.toc:
            result = -1
        else:
            result = np.nan

        return result

    def plot_rope_posterior(self, index, k, x, posterior, var):

        plt.figure(figsize=(11, 7))
        plt.subplot(1, 2, 1)
        b = posterior['muA'][1]
        plt.plot(0.5 * (b[0:-1] + b[1:]), posterior['muA'][0], lw=2, label='A')
        b = posterior['muB'][1]
        plt.plot(0.5 * (b[0:-1] + b[1:]), posterior['muB'][0], lw=2, label='B')
        plt.xlabel('$\mu_A,\  \mu_B$')
        plt.title('Conversion Rate')
        plt.locator_params(nticks=6)
        plt.legend()

        plt.subplot(1, 2, 2)
        pdf = posterior[var][0]
        plt.plot(x[pdf >= k[index]], 0 * x[pdf >= k[index]], linewidth=4)
        plt.plot(x, pdf, linewidth=4)
        plt.xlim([np.minimum(np.min(x), -1), np.maximum(1, np.max(x))])
        plt.plot([self.rope[0], self.rope[0]], [0, 4], 'g--', linewidth=5, label='ROPE')
        plt.plot([self.rope[1], self.rope[1]], [0, 4], 'g--', linewidth=5)
        plt.gca().locator_params(axis='x', numticks=6)
        plt.legend()
        if(var=='pES'):
            plt.xlabel(r'$(\mu_B-\mu_A)/\sqrt{\sigma_A^2 + \sigma_B^2)}$')
            plt.title("Effect Size")
        elif(var=='delta'):
            plt.xlabel(r'$\mu_B-\mu_A$')
