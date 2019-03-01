#  ForQuE

Computes the noise power spectrum for the CMB lensing quadratic estimators from temperature and polarization from Hu & Okamoto 2002, for any CMB experiment. The minimum-variance estimator takes into account the noise covariance of the various quadratic estimators.

Requires the Monte Carlo library vegas (https://pypi.org/project/vegas):
```
pip install vegas
```
Just clone and run:
```
python driver_cmblensrec.py
```
Hope you find this code useful! Please cite https://arxiv.org/abs/1607.01761 if you use this code in a publication. Do not hesitate to contact me with any questions: eschaan@lbl.gov

