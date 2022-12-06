# Data-Driven Algorithms for Gaussian Measurement Matrix Design in Compressive Sensing

This repository provides code to reproduce results of the paper: [Model-Based and Graph-Based Priors for Group Testing].

---
### Requirements: 
1. Python 3.7
2. gurobipy 10.0.0

---
### Reproducing quantitative results
1. Main experiment scripts:
     - ```$ ./mnist/train.sh```
     - ```$ python main.py```  
2. Sensitivity to graph mismatch:
     - ```--noise-std``` the variance (square of standard deviation) of noise
     - ```$ python main.py```  
3. Sensitivity to parameter mismatch:  
     - ```$ python main.py```  

