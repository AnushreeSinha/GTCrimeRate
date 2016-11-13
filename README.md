# GTCrimeRate
A CS8803-Big Data analysis project

## Python libs setup for running py programs
- Make sure you install pip first. Necessary py libs so far include: numpy, scipy, matplotlib
- http://chrisstrelioff.ws/sandbox/2014/06/04/install_and_setup_python_and_packages_on_ubuntu_14_04.html

## The algorithm is in kde*.py:
1. Instantiate a DataMiner object: `dm = DataMiner()`
  * converts original datasets to a new file called formatted consisting of only 3 cols of interest (done once)
  * does kde on the new formatted data file
2. Get densities using the generated kde model: `dm.getDensity(datapoints)`
  * datapoints should in the format like \[\[lats],\[lngs],\[hrs]] : ndarray of 3*N
  * will return an np-array of 1*N
- Public methods for DataMiner:
  * buildModel()
    - reads from formatted and does kde alg
  * convt_data()
    - converts original to formatted 
  * getDensity()
  	- return kde output densities
  	- save output along with datapoints variables
  * selectBdwd()
  	- a test method to select and compare bandwidths for each dimension by plotting
- Public classes:
  * CustBdwdKDE(dataset,bandwidth)
  	- A Customized class inheriting stats.gaussian_kde to set fixed bandwidth for overwriting scipy's default covariance determination
- TODO:
  - [ ] evaluate kde outputs and find good bandwidths/normalize data (currently having scatter and grid plots)
  - [ ] support passing in new datasets and append the new datapoints of 3 cols to formatted.csv
  - [ ] test if the distribution is normal(gaussian)
