#!/usr/bin/python
from statsmodels.nonparametric.kernel_density import KDEMultivariate
import numpy as np,numpy
from scipy import stats
#from statsmodels.distributions.mixture_rvs import mixture_rvs
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
#from mayavi import mlab

class DataInfo:
	COL_LAT = 12
	COL_LONG = 13
	COL_HR = 17	

"""README.md for more details on this class"""
class DataMiner:

	DATAINFO = [12,13,17]
	xmin, ymin = 33.773249, -84.409860#33.7514, -84.4234
	xmax, ymax = 33.786683, -84.391138#33.7972, -84.3708

	def __init__(self):
		self.dataset_path = "./test.csv" #"./With-Weight-Georgia_Institute_of_Technology.csv" 
		self.formated_path = "./transformed.csv"
		self.buildModel()
		
	def buildModel(self):
#		self.convt_data()
		self.data = self._read_formData()
#		self._kde_statsm()
		self._kde_scipy()
		
#### a method to look for the right bandwidth for each dimension
	def selectBdwd(self):
		d = self.data
		x, y, z = d[0],d[1],d[2] ## x=lat, y=lng, z=hr
		xbdwd = 0.00096
		fbdwdkde_x = CustBdwdKDE(x,xbdwd)
		x_grid = np.linspace(x.min(), x.max(), 1000)
		xpdf = fbdwdkde_x(x_grid)
		xpdf_true = fbdwdkde_x(x)
		fig, ax = plt.subplots()
		fig.subplots_adjust(wspace=0)
		ax.plot(x_grid, xpdf, color='blue', alpha=0.5, lw=3,label='xbw={0}'.format(xbdwd))
		ax.plot(x,np.full(x.shape,xpdf.min()),'ro')#, c=pdf_true, alpha=0.4)
		ax.hist(x, 20, fc='gray', histtype='stepfilled', alpha=0.3, normed=True)
		ax.legend(loc='upper left')
#		ax.set_xlim(self.xmin, self.xmax)
		plt.axis([x.min(), x.max(), xpdf.min(), xpdf.max()])
#		plt.show()
		
		ybdwd = 0.00096
		fbdwdkde_y = CustBdwdKDE(y,ybdwd)
		y_grid = np.linspace(y.min(), y.max(), 1000)
		ypdf = fbdwdkde_y(y_grid)
		ypdf_true = fbdwdkde_y(y)
		fig1, ax1 = plt.subplots()
		fig1.subplots_adjust(wspace=0)
		ax1.plot(y_grid, ypdf, color='blue', alpha=0.5, lw=3,label='ybw={0}'.format(ybdwd))
		ax1.plot(y,np.full(y.shape,ypdf.min()),'ro')
		ax1.hist(y, 20, fc='gray', histtype='stepfilled', alpha=0.3,normed=True)
		ax1.legend(loc='upper left')
		ax1.set_xlim(y.min(), y.max())
#		plt.show()
		
		
		zbdwd = 1
		fbdwdkde_z = CustBdwdKDE(z,zbdwd)
		z_grid = np.linspace(z.min(), z.max(), 1000)
		zpdf = fbdwdkde_z(z_grid)
		zpdf_true = fbdwdkde_z(z)
		fig2, ax2 = plt.subplots()
		fig2.subplots_adjust(wspace=0)
		ax2.plot(z_grid, zpdf, color='blue', alpha=0.5, lw=3,label='zbw={0}'.format(zbdwd))
		ax2.plot(z,np.full(z.shape,zpdf.min()),'ro')#, c=pdf_true, alpha=0.4)
		ax2.hist(z, 20, fc='gray', histtype='stepfilled', alpha=0.3, normed=True)
		ax2.legend(loc='upper left')
		ax2.set_xlim(z.min(), z.max())
		plt.show()
	
#### takes in a time and a bunch of coordinates
#### datapoints format: [[lats][lngs][hrs]]
	def getDensity(self,datapoints):
		fid = open('kde_out.csv','w')
		out = self.kde(datapoints)
		densities = out.T
		for currentIndex,elem in enumerate(densities):
		  s1 = '%f,%f,%f,%f\n'%(datapoints[0][currentIndex], datapoints[1][currentIndex], datapoints[2][currentIndex], densities[currentIndex] )
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
##		spec_hrs = np.full(self.data[0].shape,8)
#		lat, lng, hr, density = self.data[0][index], self.data[1][index], spec_hrs, density[index] # mapping of input data points
#		hr = self.data[2][index] #at all hours
		
#### matplotlib to plot all data points with density coloring
#		fig, (ax2) = plt.subplots(1,1,subplot_kw=dict(projection='3d'))
##		ax1.scatter(lat, lng, hr)#, c=density)
##		ax2.imshow(np.rot90(density), cmap=plt.cm.gist_earth_r,extent=[self.xmin, self.xmax, self.ymin, self.ymax])
#		ax2.set_xlim([self.xmin, self.xmax])
#		ax2.set_ylim([self.ymin, self.ymax])
#		ax2.scatter(lat, lng, c=density)
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

"""A Customized class inheriting stats.gaussian_kde to set fixed bandwidth for overwriting scipy's default covariance determination"""
class CustBdwdKDE(stats.gaussian_kde):
    def __init__(self, dataset, bandwidth=1):
        self.dataset = np.atleast_2d(dataset)
        self.d, self.n = self.dataset.shape
        self.set_bandwidth(bandwidth)

    def set_bandwidth(self, bandwidth):
        if np.isscalar(bandwidth):
            covariance = numpy.eye(self.d) * bandwidth**2
        else:
            bandwidth = numpy.asarray(bandwidth)
            if bandwidth.shape == (self.d,): # bandwidth is 1-d array of len(self.d)
                covariance = numpy.diag(bandwidth**2)
            else:
                raise ValueError("'bandwidth' should be a scalar, a 1-d array of length d.")
        self.set_covariance(covariance)

    def set_covariance(self, covariance):
        self.covariance = numpy.asarray(covariance)
        self.inv_cov = numpy.linalg.inv(self.covariance)
        self._norm_factor = numpy.sqrt(numpy.linalg.det(2*numpy.pi*self.covariance)) * self.n
		
dm = DataMiner()
dm.selectBdwd()
#### grid set up, may exclude outliers
#### plot on one plane of the hour
xmin, ymin, zmin = 33.7514, -84.4234, 10#self.data[0].min(), self.data[1].min(), 10#self.data[2].min()
xmax, ymax, zmax = 33.7972, -84.3708, 11#self.data[0].max(), self.data[1].max(), 11#self.data[2].max()
lat, lng = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
#		hr = np.mgrid[zmin:zmax:40j]
speci_hr = 18 ## show data at a specific time
hr = np.empty(lat.shape) ##set hr to a specific hour
for i in range(lat.shape[0]):
		for j in range(lat.shape[1]):
			hr[i,j]=speci_hr
coords = np.vstack([item.ravel() for item in [lat, lng, hr]]) 
density = dm.getDensity(coords).reshape(lat.shape)
fig3, ax3 = plt.subplots()
ax3.imshow(np.rot90(density), cmap=plt.cm.gist_earth_r,extent=[xmin, xmax, ymin, ymax])
ax3.plot(dm.data[0][dm.data[2,:]==speci_hr], dm.data[1][dm.data[2,:]==speci_hr], 'k.', markersize=2,label='hr={0}'.format(speci_hr))
ax3.set_xlim([xmin, xmax])
ax3.set_ylim([ymin, ymax])
ax3.legend(loc='upper left')
plt.show()
#		mlab.contour3d(lat, lng, hr, density, opacity=0.5)
#		mlab.axes()
#		mlab.show()

