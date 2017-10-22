import numpy as np
from scipy.stats import beta
from .utils import check_size, print_result, print_info
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline
import warnings
import pymc3 as pm


class AbExp:
    '''
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
    '''
    def __init__(self, method='analytic', rule='rope',
                 alpha=0.95, alpha_prior=1, beta_prior=1,
                 resolution=500, rope=(-0.1, 0.1), toc=1.e-2,
                 iterations=5000, plot=False, decision_var='es'):
        self.method = method
        self.rule = rule
        self.alpha = alpha
        self.alpha_prior = alpha_prior
        self.beta_prior = beta_prior
        self.resolution = resolution
        self.rope = rope
        self.toc = toc
        self.iterations = iterations
        self.plot = plot
        self.decision_var = decision_var

        if(method == 'compare' and not rule == 'rope'):
            warnings.warn('For "compare" method, only ROPE decision rule is currently supported. Setting rule to ROPE.')
            self.rule = 'rope'

        if(rule == 'loss' and decision_var == 'es'):
            warnings.warn('For "loss" decision rule, only "lift" decision variable is currently supported. Setting decision_var to "lift".')
            self.decision_var = 'lift'

    def experiment(self, data):
        '''
        Run experiment with data provided
        Parameters
        ----------
        data : `List(np.array, np.array)`
        '''
        check_size(data, dim=2)

        posterior = self.find_posterior(data)

        decision = self.decision(posterior)

        if (plt.plot):
            plt.show()

        return decision

    def find_posterior(self, data):
        '''
        Find posterior distribution
        '''
        if self.method == 'analytic':
            posterior = self.posterior_analytic(data)
        elif self.method == 'mcmc':
            posterior = self.posterior_mcmc(data)
        elif self.method == 'compare':
            posterior = [self.posterior_analytic(data), self.posterior_mcmc(data)]
        else:
            raise Exception('method not recognized')

        return posterior

    def decision(self, posterior):
        '''
        Make decision on the experiment
        :param posterior:
        :return:
        '''
        if self.plot:
            plt.figure(figsize=(9, 6))

        if self.method == 'compare':
            hpd1 = self.hpd(posterior[0], self.decision_var, {'clr':'r', 'label1':'analytic', 'label2':'',
                                                  'label3':'', 'label4':'', 'label':'analytic'})
            result1 = self.rope_decision(hpd1)

            hpd2 = self.hpd(posterior[1], self.decision_var, {'clr':'k', 'ls':'--', 'label1':'mcmc',
                                                  'label2':'', 'label3':'', 'label4':'', 'label':'mcmc'})
            result2 = self.rope_decision(hpd2)
            result = [result1, result2]
        else:
            if self.rule == 'rope':
                hpd = self.hpd(posterior, self.decision_var)
                result = self.rope_decision(hpd)
            elif self.rule == 'loss':
                result = self.expected_loss_decision(posterior, self.decision_var)

        print_info(self)

        return print_result(result)

    def posterior_analytic(self, data):
        '''
        Find posterior distribution for the analytic method of solution
        :return:
        '''

        ca = np.sum(data[0])
        na = len(data[0])

        cb = np.sum(data[1])
        nb = len(data[1])

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
        lift = np.histogram(rvs, bins=bins, normed=True)

        bins = np.linspace(0, 1, self.resolution)
        sigma_a_rvs = np.sqrt(a_rvs * (1 - a_rvs))
        sigma_b_rvs = np.sqrt(b_rvs * (1 - b_rvs))
        psigma_a = np.histogram(sigma_a_rvs, bins=bins, normed=True)
        psigma_b = np.histogram(sigma_b_rvs, bins=bins, normed=True)

        rvs = (b_rvs - a_rvs) / np.sqrt(0.5 * (sigma_a_rvs**2 + sigma_b_rvs**2))
        bins = np.linspace(np.min(rvs) - 0.2 * abs(np.min(rvs)), np.max(rvs) + 0.2 * abs(np.max(rvs)), self.resolution)
        pes = np.histogram(rvs, bins=bins, normed=True)

        posterior = {'muA': pa, 'muB': pb, 'psigma_a': psigma_a, 'psigma_b': psigma_b,
                     'lift': lift, 'es': pes, 'prior': self.prior()}

        return posterior

    def posterior_mcmc(self, data):
        '''
        Find posterior distribution for the numerical method of solution
        :return:
        '''

        with pm.Model() as ab_model:
            # priors
            mua = pm.distributions.continuous.Beta('muA', alpha=self.alpha_prior, beta=self.beta_prior)
            mub = pm.distributions.continuous.Beta('muB', alpha=self.alpha_prior, beta=self.beta_prior)
            # likelihoods
            pm.Bernoulli('likelihoodA', mua, observed=data[0])
            pm.Bernoulli('likelihoodB', mub, observed=data[1])

            # find distribution of difference
            pm.Deterministic('lift', mub - mua)
            # find distribution of effect size
            sigma_a = pm.Deterministic('sigmaA', np.sqrt(mua * (1 - mua)))
            sigma_b = pm.Deterministic('sigmaB', np.sqrt(mub * (1 - mub)))
            pm.Deterministic('effect_size', (mub - mua) / (np.sqrt(0.5 * (sigma_a ** 2 + sigma_b ** 2))))

            start = pm.find_MAP()
            step = pm.Slice()
            trace = pm.sample(self.iterations, step=step, start=start)

        bins = np.linspace(0, 1, self.resolution)
        mua = np.histogram(trace['muA'][500:], bins=bins, normed=True)
        mub = np.histogram(trace['muB'][500:], bins=bins, normed=True)
        sigma_a = np.histogram(trace['sigmaA'][500:], bins=bins, normed=True)
        sigma_b = np.histogram(trace['sigmaB'][500:], bins=bins, normed=True)

        rvs = trace['lift'][500:]
        bins = np.linspace(np.min(rvs) - 0.2 * abs(np.min(rvs)), np.max(rvs) + 0.2 * abs(np.max(rvs)), self.resolution)
        lift = np.histogram(rvs, bins=bins, normed=True)

        rvs = trace['effect_size'][500:]
        bins = np.linspace(np.min(rvs) - 0.2 * abs(np.min(rvs)), np.max(rvs) + 0.2 * abs(np.max(rvs)), self.resolution)
        pes = np.histogram(rvs, bins=bins, normed=True)

        posterior = {'muA': mua, 'muB': mub, 'sigmaA': sigma_a, 'sigmaB': sigma_b,
                     'lift': lift, 'es': pes, 'prior': self.prior()}

        return posterior

    def prior(self):
        '''
        Find out prior distribution
        :return:
        '''
        return [beta.pdf(x, self.alpha_prior, self.beta_prior) for x in np.linspace(0, 1, self.resolution)]

    def hpd(self, posterior, var, *type):
        '''
        Find out High Posterior Density Region
        '''

        bins = posterior[var][1]
        x = 0.5 * (bins[0:-1] + bins[1:])
        pdf = posterior[var][0]
        k = np.linspace(0, max(pdf), 1000)
        area_above = np.array([np.trapz(pdf[pdf >= kk], x[pdf >= kk]) for kk in k])
        index = np.argwhere(np.abs(area_above - self.alpha) == np.min(np.abs(area_above - self.alpha)))[0]

        if self.plot:
            self.plot_rope_posterior(index, k, x, posterior, var, *type)

        return x[pdf >= k[index]]

    def rope_decision(self, hpd):
        '''
        Apply decision rule for ROPE method
        :param hpd:
        :return:
        '''

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
        '''
        Calculate expected loss and apply decision rule
        '''
        dl = posterior[var][1]
        dl = 0.5 * (dl[0:-1] + dl[1:])
        fdl = posterior[var][0]
        inta = np.maximum(dl, 0) * fdl
        intb = np.maximum(-dl, 0) * fdl

        ela = np.trapz(inta, dl)
        elb = np.trapz(intb, dl)

        if self.plot:
            plt.subplot(1,2,1)
            b = posterior['muA'][1]
            plt.plot(0.5*(b[0:-1]+b[1:]), posterior['muA'][0], lw=2, label=r'$f(\mu_A)$')
            b = posterior['muB'][1]
            plt.plot(0.5*(b[0:-1]+b[1:]), posterior['muB'][0], lw=2, label=r'$f(\mu_B)$')
            plt.xlabel('$\mu_A,\  \mu_B$')
            plt.xlim([0, 1])
            plt.title('Conversion Rate')
            plt.locator_params(nticks=6)
            plt.gca().set_ylim(bottom=0)
            plt.legend()

            plt.subplot(1,2,2)
            plt.plot(dl, fdl, 'b', lw=3, label=r'f$(\mu_B - \mu_A)$')
            plt.plot([ela, ela], [0, 0.3*np.max(fdl)], 'r', lw=3, label='A: Expected Loss')
            plt.plot([elb, elb], [0, 0.3*np.max(fdl)], 'c', lw=3, label='B: Expected Loss')
            plt.plot([self.toc, self.toc], [0, 0.3*np.max(fdl)], 'k--', lw=3, label='Threshold of Caring')
            plt.xlabel(r'$\mu_B-\mu_A$')
            plt.title('Expected Loss')
            plt.gca().set_ylim(bottom=0)
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

    def plot_rope_posterior(self, index, k, x, posterior, var, *args):

        label1 = r'$f(\mu_A)$'
        label2 = r'$f(\mu_B)$'
        label3 = 'HPD'
        label4 = 'ROPE'
        if var=='es':
            label = '$f$(ES)'
        elif var=='lift':
            label = r'$f(\mu_B - \mu_A)$'
        ls = '-'

        for arg in args:
            if 'ls' in arg:
                ls = arg['ls']
            if 'clr' in arg:
                clr = arg['clr']
            if 'label1' in arg:
                label1 = arg['label1']
            if 'label2' in arg:
                label2 = arg['label2']
            if 'label3' in arg:
                label3 = arg['label3']
            if 'label4' in arg:
                label4 = arg['label4']
            if 'label' in arg:
                label = arg['label']

        plt.subplot(1, 2, 1)
        b = posterior['muA'][1]
        line, = plt.plot(0.5 * (b[0:-1] + b[1:]), posterior['muA'][0], ls=ls, lw=2, label=label1)
        if 'clr' in locals():
            line.set_color(clr)
        b = posterior['muB'][1]
        line, = plt.plot(0.5 * (b[0:-1] + b[1:]), posterior['muB'][0], ls=ls, lw=2, label=label2)
        if 'clr' in locals():
            line.set_color(clr)
        plt.xlabel('$\mu_A,\  \mu_B$')
        plt.xlim([0,1])
        plt.title('Conversion Rate')
        plt.gca().set_ylim(bottom=0)
        plt.locator_params(nticks=6)
        plt.legend()

        plt.subplot(1, 2, 2)
        pdf = posterior[var][0]
        line, = plt.plot(x, pdf, lw=3, ls='-', label=label)
        if 'clr' in locals():
            line.set_color(clr)
        plt.plot(x[pdf >= k[index]], 0 * x[pdf >= k[index]], linewidth=4, label=label3)
        plt.xlim([np.minimum(np.min(x), -1), np.maximum(1, np.max(x))])
        plt.plot([self.rope[0], self.rope[0]], [0, 4], 'g--', linewidth=5, label=label4)
        plt.plot([self.rope[1], self.rope[1]], [0, 4], 'g--', linewidth=5)
        plt.gca().set_ylim(bottom=0)
        plt.gca().locator_params(axis='x', numticks=6)
        plt.legend()
        if(var=='es'):
            plt.xlabel(r'$(\mu_B-\mu_A)/\sqrt{\sigma_A^2 + \sigma_B^2)}$')
            plt.title('Effect Size')
        elif(var=='lift'):
            plt.xlabel(r'$\mu_B-\mu_A$')
            plt.title(r'Lift')
