#!/usr/bin/python
from statsmodels.nonparametric.kernel_density import KDEMultivariate
import numpy as np
from scipy import stats
from statsmodels.distributions.mixture_rvs import mixture_rvs
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mayavi import mlab

class DataInfo:
	COL_LAT = 12
	COL_LONG = 13
	COL_HR = 17	

class DataMiner:

	DATAINFO = [12,13,17]

	def __init__(self):
		self.dataset_path = "./test.csv" #"./With-Weight-Georgia_Institute_of_Technology.csv" 
		self.formated_path = "./transformed.csv"
		self.buildModel()
		
	def buildModel(self):
#		self.convt_data()
		self.data = self._read_formData()
#		self._kde_statsm()
		self._kde_scipy()

#### takes in a time and a bunch of coordinates
#### datapoints format: [[lats][lngs][hrs]]
	def getDesity(self,datapoints):
		return self.kde(datapoints)

	def _kde_scipy(self):
		self.data = self.data.T
#		print "Transposed data: "
#		print(self.data)
#### self.data as formatted below
#### [[lat1,.....]
####  [lng1,.....] 
####  [hr1,......]]
		self.kde = stats.gaussian_kde(self.data)
		density = self.kde(self.data)
		index = density.argsort()
		spec_hrs = np.full(self.data[0].shape,10)
		lat, lng, hr, density = self.data[0][index], self.data[1][index], spec_hrs, density[index]

#		print "lat: "
#		print(lat)
#		print "lng: "
#		print(lng)
#		print "hr: "
#		print(hr)
#		print "density: "
#		print(density)

#		xmin, ymin, zmin = self.data[0].min(), self.data[1].min(), self.data[2].min()
#		xmax, ymax, zmax = self.data[0].max(), self.data[1].max(), self.data[2].max()
#		lat, lng, zi = np.mgrid[xmin:xmax:30j, ymin:ymax:30j, zmin:zmax:30j]
#		hr = np.empty(lat.shape)
#		for i in range(lat.shape[0]):
#    			for j in range(lat.shape[1]):
#        			hr[i,j]=21
#		coords = np.vstack([item.ravel() for item in [lat, lng, hr]]) #zi]])
#		density = self.kde(coords).reshape(lat.shape)
#		print(hr)
#		print(lat)
#		print str(hr.shape)+" "+str(lat.shape)

		figure = mlab.figure('DensityPlot')
		
#		grid = mlab.pipeline.scalar_field(lat, lng, hr, density)#zi, density)
#		min = density.min()
#		max = density.max()
#		mlab.pipeline.volume(grid, vmin=min, vmax=min + .5*(max-min))
		
#### plot location points at 13:00
		pts = mlab.points3d(lat, lng, hr, density, scale_mode='none', scale_factor=0.07)
		figure.scene.disable_render = True
		mask = pts.glyph.mask_points
		mask.maximum_number_of_points = lat.size
		mask.on_ratio = 1
		pts.glyph.mask_input_points = True
		figure.scene.disable_render = False
		mlab.axes()
		mlab.show()

#### matplotlib
#		fig = plt.figure()
#		fig.suptitle('lat,lng,hr')
#		plt.subplot(1,2,1)
#		ax = Axes3D(fig)
#		cset = ax.contour(lat, lng, hr, 16, extend3d=True)
#		ax.clabel(cset, fontsize=9, inline=1)
#		ax.scatter(x, y, z, c=density)
#		plt.show()

	def _read_formData(self):
		form_file = open(self.formated_path,"r")
		record = form_file.readline()
		data = []
		while(record!=""):
			attrs = record[:-1].split(",")
#			print(attrs)
			data.append(attrs)
			record = form_file.readline()
#		print(data)
		form_file.close()
		data = np.array(data,dtype=float)
#		data.astype(np.float)
		return data

#### taking out cols of interest and save them to a new file
	def convt_data(self):
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

