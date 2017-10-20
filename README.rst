abyes is a Python package for Bayesian A/B Testing.

Installation
============
(Suggestion: install the package in a virtual environment)

* In your target folder, clone the repository with the command::

git clone https://github.com/cbellei/abyes.git

* Then, inside the same folder::

pip install .

* To check that the package has been installed, in iPython type::

import abyes

* If everything works correctly, the package will be imported without errors.

Dependencies
============
* abyes is tested on Python 3.5 and depends on NumPy, Scipy, Matplotlib, Pymc3 (see ``requirements.txt`` for version
information).

Limitations
===========
abyes currently:
* only focuses on conversion rate experiments
* only allows for two variants at a time to be tested
These shortcomings may be improved in future versions of ABYes.

Example
=======
* In ipython, type::
import abyes
import numpy as np

data = [np.random.binomial(1, 0.5, size=10000), np.random.binomial(1, 0.3, size=10000)]
exp = abyes.AbExp(alpha=0.95, method='analytic', rule='loss')
exp.experiment(data)

* This will give the following result::

*** abyes ***

Method = analytic
Decision Rule = loss
Threshold of Caring = 0.01

Result is conclusive: A variant is winner!

* There are more examples available in the file ``example.py``, which can be run from the root directory with the command::
python abyes/examples/example.py

Licence
=======
`Apache License, Version
2.0 <https://github.com/cbellei/abyes/blob/master/LICENSE>`__