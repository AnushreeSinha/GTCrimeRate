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
	def getDensity(self,datapoints):
		fid = open('kde_out.csv','w')
		out = self.kde(datapoints)
		densities = out.T
		for currentIndex,elem in enumerate(densities):
		  s1 = '%f %f %f %f\n'%(datapoints[0][currentIndex], datapoints[1][currentIndex], datapoints[2][currentIndex], densities[currentIndex] )
		  fid.write(s1)
		fid.close()
		return out

	def _kde_scipy(self):
		self.data = self.data.T
#### plot input data points
#		fig0, ax0 = plt.subplots(subplot_kw=dict(projection='3d'))
#		x,y,z = self.data[0], self.data[1], self.data[2]
#		ax0.scatter(x,y,z)
#		plt.show()

#### self.data as formatted below
#### [[lat1,.....]
####  [lng1,.....] 
####  [hr1,......]]
		self.kde = stats.gaussian_kde(self.data)
		
#		density = self.kde(self.data)
#		index = density.argsort()
#		spec_hrs = np.full(self.data[0].shape,16)
#		lat, lng, hr, density = self.data[0][index], self.data[1][index], spec_hrs, density[index] # mapping of input data points
#		hr = self.data[2][index] #at all hours
		
#### matplotlib to plot all data points with density coloring
#		fig, (ax2) = plt.subplots(1,1,subplot_kw=dict(projection='3d'))
##		ax1.scatter(lat, lng, hr)#, c=density)
#		ax2.scatter(lat, lng, hr, c=density)
#		plt.show()

#### mayavi plotting
#		figure = mlab.figure('DensityPlot')
#		grid = mlab.pipeline.scalar_field(lat, lng, hr, density)#zi, density)
#		min = density.min()
#		max = density.max()
#		mlab.pipeline.volume(grid, vmin=min, vmax=min + .5*(max-min))
		
#### plot location points at a specific time
#		pts = mlab.points3d(lat, lng, hr, density, scale_mode='none', scale_factor=0.07)
#		figure.scene.disable_render = True
#		mask = pts.glyph.mask_points
#		mask.maximum_number_of_points = lat.size
#		mask.on_ratio = 1
#		pts.glyph.mask_input_points = True
#		figure.scene.disable_render = False
#		mlab.axes()
#		mlab.show()

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
#### grid set up, may exclude outliers
#### plot on one plane of the hour
xmin, ymin, zmin = 33.7514, -84.4234, 10#self.data[0].min(), self.data[1].min(), 10#self.data[2].min()
xmax, ymax, zmax = 33.7972, -84.3708, 11#self.data[0].max(), self.data[1].max(), 11#self.data[2].max()
lat, lng = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
#		hr = np.mgrid[zmin:zmax:40j]
hr = np.empty(lat.shape) ##set hr to a specific hour
for i in range(lat.shape[0]):
		for j in range(lat.shape[1]):
			hr[i,j]=20
coords = np.vstack([item.ravel() for item in [lat, lng, hr]]) 
density = dm.getDensity(coords).reshape(lat.shape)
fig3, ax3 = plt.subplots()
ax3.imshow(np.rot90(density), cmap=plt.cm.gist_earth_r,extent=[xmin, xmax, ymin, ymax])
ax3.plot(dm.data[0], dm.data[1], 'k.', markersize=2)
ax3.set_xlim([xmin, xmax])
ax3.set_ylim([ymin, ymax])
plt.show()
#		mlab.contour3d(lat, lng, hr, density, opacity=0.5)
#		mlab.axes()
#		mlab.show()

