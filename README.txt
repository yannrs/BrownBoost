######################################
## Project BDS - CS 8803 - Fall 2016
## Group 15
## Authors: Christophe Rannou, Yann Ravel-Sibillot, Alan Chern, Pierrick Calmels

# First Python implementation


# Required Libraries
Python version 		: 2.7.12
scikit-learn >0.18 	: http://scikit-learn.org/stable/; Machine Learning Library, and also used for measuring performances
numpy				: http://www.numpy.org/; the fundamental package for scientific computing with Python
scipy				: https://www.scipy.org/;  Python-based ecosystem of open-source software for mathematics, science, and engineering


# Files:
launcher.py			: File to launch to run a benchmark
benchmark.py		: File which contains all the functions to perform the benchmark
brownboost.py		: BrownBoost Implementation

results/		: Folder which will contain results as one file per benchmark (will be created if does not exist)
resources/		: Folder which contain dataset for testing (any path can be used in the benchmark)


# Documentation
launcher.py -t <train data path> -s <test path> -o <output name> -n <proportion of noisy points in the dataset> -b <number of bags if needed> -c <decreasing age in Brownboost> -l <Browboost threshold to prevent divergence>
you may have only a datafile, if so, the dataset will be split into 80\%-20\%


# Examples for line to run:
python launcher.py -h
python launcher.py -t ./resources/messidor_dataset -o benchmark_results -n 0.2 -b 10 -c 8


# Output of the Benchmark
located in the results/ directory (which is created if does not exist), the program returns 3 files:
prediction_BrownBoostClassifier : csv file containing the results of the Browboost classification and the ground truth labels
prediction_AdaBoostClassifier   : csv file containing the results of the Adaboost classification and the ground truth labels
<output name>.csv 				: metrics of the algorithms compared during the benchmark


# Dataset used for the benchmark : https://archive.ics.uci.edu/ml/datasets/Diabetic+Retinopathy+Debrecen+Data+Set
resources\messidor_dataset 	: Diabetic Retinopathy Debrecen 
Data Set Characteristics: 	Multivariate
Number of Instances: 		1151
Area: 	  					Life  Attribute 
Characteristics: 	  		Integer, Real 	  
Number of Attributes: 	  	20 	  
Date Donated 	  			2014-11-03  
Associated Tasks: 	  		Classification 	  
Missing Values? 	  		N/A 	  
Number of Web Hits: 	  	22269

