## AddShare: a Privacy-Preserving Approach for Federated Learning - DPM 2032

This repository has all the code used in the experiments carried out in the paper *"AddShare: a Privacy-Preserving Approach for Federated Learning"* [1].


This repository is organized as follows:

* **code** folder - contains all the code for reproducing the experiments described in the paper;
* **results** folder - contains all the figures obtained from the experimental evaluation on 4 standard data sets;


### Requirements

The experimental design was implemented in Python language. Both code and data are in a format suitable for R environment.

In order to replicate these experiments you will need a working installation
  of python. Check [https://www.python.org/downloads/]  if you need to download and install it.

In your python installation you also need to install the following key python packages:

  - Tensorflow
  - Keras
  - Cryptography


  All the above packages, together with other unmentioned dependencies, can be installed via pip. Essentially you need to issue the following command within the code sub-folder:

```r
pip install -r requirements.txt
```

*****

### References
[1] Asare, B. A. and Branco, P. and Kiringa, I. and Yeap, T. (2017) *"AddShare: a Privacy-Preserving Approach for Federated Learning"*  Procedings of Machine Learning Research, DPM 2023, Short Paper, The Hague, Netherlands. (to appear).