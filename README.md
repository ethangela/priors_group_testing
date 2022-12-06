# Ising model based group testing 
 
This repository provides code to reproduce results of the paper: [Model-Based and Graph-Based Priors for Group Testing].

---
### Requirements
1. Python 3.7
2. gurobipy 10.0.0

---
### Reproducing quantitative results
1. Main experiment scripts:
     - ```python main.py```  
2. Sensitivity to graph mismatch:
     - ```python varying_graph_test.py```  
3. Sensitivity to parameter mismatch:  
     - ```python varying_lambda_test.py```  
     
---
### Additional info
We provide two graph examples (grid and block) produced by Gibbs sampling. Feel free to create your own graph samples in replace of them. 

