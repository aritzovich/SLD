Climate Model Simulation Crashes Data Set 

Abstract: Given Latin hypercube samples of 18 climate model input parameter values, predict climate model simulation crashes and determine the parameter value combinations that cause the failures.

Data Set Characteristics:  

Multivariate

Number of Instances:

540

Area:

Physical

Attribute Characteristics:

Real

Number of Attributes:

18

Date Donated

2013-06-18

Associated Tasks:

Classification

Missing Values?

N/A

Number of Web Hits:

94290


Source:

D. Lucas (ddlucas .at. alum.mit.edu), Lawrence Livermore National Laboratory 
R. Klein (rklein .at. astron.berkeley.edu), Lawrence Livermore National Laboratory & U.C. Berkeley 
J. Tannahill (tannahill1 .at. llnl.gov), Lawrence Livermore National Laboratory 
D. Ivanova (ivanova2 .at. llnl.gov), Lawrence Livermore National Laboratory 
S. Brandon (brandon1 .at. llnl.gov), Lawrence Livermore National Laboratory 
D. Domyancic (domyancic1 .at. llnl.gov), Lawrence Livermore National Laboratory 
Y. Zhang (zhang24 .at. llnl.gov), Lawrence Livermore National Laboratory 

This data was constructed using LLNL's UQ Pipeline, was created under the auspices of the US Department of Energy by Lawrence Livermore National Laboratory under Contract DE-AC52-07NA27344, was funded by LLNL's Uncertainty Quantification Strategic Initiative Laboratory Directed Research and Development Project under tracking code 10-SI-013, and is released under UCRL number LLNL-MISC-633994.


Data Set Information:

This dataset contains records of simulation crashes encountered during climate model uncertainty quantification (UQ) ensembles. 

Ensemble members were constructed using a Latin hypercube method in LLNL's UQ Pipeline software system to sample the uncertainties of 18 model parameters within the Parallel Ocean Program (POP2) component of the Community Climate System Model (CCSM4). 

Three separate Latin hypercube ensembles were conducted, each containing 180 ensemble members. 46 out of the 540 simulations failed for numerical reasons at combinations of parameter values. 

The goal is to use classification to predict simulation outcomes (fail or succeed) from input parameter values, and to use sensitivity analysis and feature selection to determine the causes of simulation crashes. 

Further details about the data and methods are given in the publication 'Failure Analysis of Parameter-Induced Simulation Crashes in Climate Models,' Geoscientific Model Development ([Web Link]).


Attribute Information:

The goal is to predict climate model simulation outcomes (column 21, fail or succeed) given scaled values of climate model input parameters (columns 3-20). 

Column 1: Latin hypercube study ID (study 1 to study 3) 

Column 2: simulation ID (run 1 to run 180) 

Columns 3-20: values of 18 climate model parameters scaled in the interval [0, 1] 

Column 21: simulation outcome (0 = failure, 1 = success)


Relevant Papers:

Lucas, D. D., Klein, R., Tannahill, J., Ivanova, D., Brandon, S., Domyancic, D., and Zhang, Y.: Failure analysis of parameter-induced simulation crashes in climate models, Geosci. Model Dev. Discuss., 6, 585-623, [Web Link], 2013. 
[[Web Link]]



Citation Request:

Please cite our final revised paper in Geoscientific Model Development.