# GTCrimeRate
A CS8803-Big Data analysis project

## Python libs setup for running py programs
- Make sure you install pip first. Necessary py libs so far include: numpy, scipy, matplotlib
- Instructions: http://chrisstrelioff.ws/sandbox/2014/06/04/install_and_setup_python_and_packages_on_ubuntu_14_04.html

## The algorithm is in kde_model.py:
1. Instantiate a DataMiner object: `dm = DataMiner()`
  * converts original datasets (dataset.csv **not with-georgia-tech-blahblah.csv**) to a new file called transformed.csv consisting of only 3 cols of interest (done once)
  * does kde on the new formatted data file
- Public methods for DataMiner:
  * buildModel()
    - reads from transformed and does kde alg
  * convt_data()
    - converts original to transformed 
  * getDensity()
    - datapoints should be in the format like \[\[lats],\[lngs],\[hrs]] : ndarray of 3*N
  	 - return kde output densities
  	 - save output along with datapoints variables in the format of (lat,lng,density)
  * selectBdwd()
  	 - a test method to select and compare bandwidths for each dimension by plotting
- Public classes:
  * CustBdwdKDE(dataset,bandwidth)
  	 - A Customized class inheriting stats.gaussian_kde to set fixed bandwidth for overwriting scipy's default covariance determination
- TODO:
  - [x] **Evaluate kde outputs and find good bandwidths/normalize data (currently having scatter and grid plots)**
  - [ ] Support passing in new datasets and append the new datapoints of 3 cols to formatted.csv
  - [ ] Test if the distribution is normal(gaussian)
  - [ ] **Compare UI map density distributions to other existing solutions**
