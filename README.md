# Casino Probabilistic Modelling Library

This is awesome.


## machine.py / engine.py / kernel.py

Contains the virtual machine that evaluates and samples the probabilistic evaluations. Samples return a digest instance that gets wrapped in a random variable. C space.

## digest.py / histogram.py / core.py

This library will construct a digest approximation based on the samples. This is mainly in c space.

## ranvar.py / random.py / algebra.py

This is Python space and will be used to compile the over libraries.
