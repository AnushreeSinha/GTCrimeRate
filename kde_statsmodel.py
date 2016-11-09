#!/usr/bin/python
from statsmodels.nonparametric.kernel_density import KDEMultivariate
import numpy as np
from scipy import stats
from statsmodels.distributions.mixture_rvs import mixture_rvs
import matplotlib.pyplot as plt

class DataInfo:
	COL_LAT = 12
	COL_LONG = 13
	COL_HR = 17	

class DataMiner:

	DATAINFO = [12,13,17]

	def __init__(self):
		self.dataset_path = "./test.csv" #"./With-Weight-Georgia_Institute_of_Technology.csv" 
		self.formated_path = "./testout.csv"
		
	def buildModel(self):
#		self._convt_data()
		self.data = self._read_formData()
#		self._kde_statsm()

	def _read_formData(self):
		form_file = open(self.formated_path,"r")
		record = form_file.readline()
		data = []
		while(record!=""):
			attrs = record[:-1].split(",")
#			print(attrs)
			data.append(attrs)
			record = form_file.readline()
		print(data)
		form_file.close()
		return data

	def _convt_data(self):
		orig_file = open(self.dataset_path,"r")
		#### add a flag indicating whether appending
		form_file = open(self.formated_path,"w")
		record = orig_file.readline()
		record = orig_file.readline()	
#		print("here: "+record)	
#		cnt = 0
#		data = np.loadtxt(file, delimiter=',', usecols=(DataInfo.COL_LAT, DataInfo.COL_LONG, DataInfo.COL_HR))
		while(record!=""):
			attrs = record.split("\t")
			ndarr = ""
			for ind in self.DATAINFO:
				colstr = attrs[ind]
				if(colstr==""):
					ndarr = []
					break
				else:
					ndarr = ndarr+(colstr)+","
			print(ndarr)
			if(ndarr!=[]):
				form_file.write(ndarr[:-1]+"\n")
			record = orig_file.readline()
#			ndarr.append((int)attrs[DataInfo.COL_LAT])
#			ndarr.appesnd((int)attrs[DataInfo.COL_LONG])
#			ndarr.append((int)attrs[DataInfo.COL_HR])
#			ndarr = np.loadtxt(record, delimiter=',', usecols=(DataInfo.COL_LAT, DataInfo.COL_LONG, DataInfo.COL_HR))
#			np.vstack((data,ndarr))
#			cnt = cnt+1
		orig_file.close()
		form_file.close()

	def _kde_statsm(self, bandwidth=0.2, **kwargs):
		kde = KDEMultivariate(self.data, bw=bandwidth,
		                      var_type='c', **kwargs)
		#kde.fit()
		#fig = plt.figure(figsize=(12,8))
		#ax = fig.add_subplot(111)
		#ax.hist(self.data, bins=10, normed=True, color='red')
		#ax.plot(kde.support, kde.density, lw=2, color='black');
		#return kde.pdf(x_grid)
		
dm = DataMiner()
dm.buildModel()
#np.random.seed(12345)
#obs_dist1 = mixture_rvs([.25,.75], size=10000, dist=[stats.norm, stats.norm],kwargs = (dict(loc=-1,scale=.5),dict(loc=1,scale=.5)))
