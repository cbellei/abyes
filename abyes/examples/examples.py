import abyes as ab
import numpy as np

# --- ANALYTIC METHOD ---
# warning: only decision_var='lift' currently supported when rule=='loss'
data = [np.random.binomial(1, 0.5, size=10000), np.random.binomial(1, 0.5, size=10000)]
# Example 1: analytic method, rope decision rule, lift decision variable
exp1 = ab.AbExp(alpha=0.95, method='analytic', rule='rope', decision_var='lift', plot=True)
exp1.experiment(data)
# Example 2: analytic method, rope decision rule, effect size decision variable
exp2 = ab.AbExp(alpha=0.95, method='analytic', rule='rope', decision_var='es', plot=True)
exp2.experiment(data)
# Example 3: analytic method, loss decision rule, lift decision variable
exp3 = ab.AbExp(alpha=0.95, method='analytic', rule='loss', decision_var='lift', plot=True)
exp3.experiment(data)

# --- MCMC METHOD ---
# warning: only decision_var='lift' currently supported when rule=='loss'
data = [np.random.binomial(1, 0.5, size=10000), np.random.binomial(1, 0.7, size=10000)]
# Example 1: mcmc method, rope decision rule, lift decision variable
exp1 = ab.AbExp(alpha=0.95, method='mcmc', rule='rope', decision_var='lift', resolution=1000, plot=True)
exp1.experiment(data)
# Example 2: mcmc method, rope decision rule, effect size decision variable
exp2 = ab.AbExp(alpha=0.95, method='mcmc', rule='rope', decision_var='es', resolution=1000, plot=True)
exp2.experiment(data)
# Example 3: mcmc method, loss decision rule, lift decision variable
exp3 = ab.AbExp(alpha=0.95, method='mcmc', rule='loss', decision_var='lift', plot=True, resolution=1000)
exp3.experiment(data)

# --- COMPARE ANALYTIC vs. MCMC METHOD ---
# warning: only ROPE method currently supported in "compare" mode
data = [np.random.binomial(1, 0.8, size=2500), np.random.binomial(1, 0.2, size=2500)]
exp1 = ab.AbExp(alpha=0.95, method='compare', rule='rope', decision_var='lift', plot=True, resolution=1000)
exp1.experiment(data)
