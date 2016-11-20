#!/usr/bin/python
from statsmodels.nonparametric.kernel_density import KDEMultivariate
import numpy as np,numpy
from scipy import stats
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
#	xmin, ymin = 33.773249, -84.409860
#	xmax, ymax = 33.786683, -84.391138
	xmin, ymin = 33.739, -84.43
	xmax, ymax = 33.8, -84.36

	def __init__(self):
		self.dataset_path = "./dataset.csv" #"./With-Weight-Georgia_Institute_of_Technology.csv" 
		self.formated_path = "./transformed.csv"
		self.buildModel()
		
	def buildModel(self):
#		self.convt_data()
		self.data = self._read_formData()
		self._kde_scipy()
		
#### a method to look for the right bandwidth for each dimension
	def selectBdwd(self):
		d = self.data
		x, y, z = d[0],d[1],d[2] ## x=lat, y=lng, z=hr
		x_grid = np.linspace(x.min(), x.max(), 10000)
		fig, ax = plt.subplots()
		fig.subplots_adjust(wspace=0)
		for xbdwd in [0.0010, 0.0009, 0.0008, 0.0007,0.0006,0.0005,0.0004]:
			fbdwdkde_x = CustBdwdKDE(x,xbdwd)
			xpdf = fbdwdkde_x(x_grid)
			#xpdf_true = fbdwdkde_x(x)
			ax.plot(x_grid, xpdf, alpha=0.5, lw=3,label='xbw={0}'.format(xbdwd))
#		ax.plot(x,np.full(x.shape,xpdf.min()),'ro')#, c=pdf_true, alpha=0.4)
		ax.scatter(x,np.full(x.shape,(xpdf.max()+xpdf.min())/2), alpha=0.4)
		ax.hist(x, 1000, fc='gray', histtype='stepfilled', alpha=0.3, normed=True)
		ax.legend(loc='upper left')
		plt.axis([x.min(), x.max(), xpdf.min(), xpdf.max()])
		ax.set_xlim(self.xmin, self.xmax)
		plt.show()
		
		y_grid = np.linspace(y.min(), y.max(), 10000)
		fig1, ax1 = plt.subplots()
		fig1.subplots_adjust(wspace=0)
		for ybdwd in [0.002, 0.0015,0.0010, 0.0009, 0.0008, 0.0007,0.0006]:
			fbdwdkde_y = CustBdwdKDE(y,ybdwd)	
			ypdf = fbdwdkde_y(y_grid)
			#ypdf_true = fbdwdkde_y(y)
			ax1.plot(y_grid, ypdf, alpha=0.5, lw=3,label='ybw={0}'.format(ybdwd))
		ax1.scatter(y,np.full(y.shape,(ypdf.max()+ypdf.min())/2), alpha=0.4)
		ax1.hist(y, 1000, fc='gray', histtype='stepfilled', alpha=0.3,normed=True)
		ax1.legend(loc='upper left')
		ax1.set_xlim(self.ymin, self.ymax)
	
		fig2, ax2 = plt.subplots()
		fig2.subplots_adjust(wspace=0)
		z_grid = np.linspace(z.min(), z.max(), 10000)
		for zbdwd in [1.5,1,0.7,0.3]:
			fbdwdkde_z = CustBdwdKDE(z,zbdwd)
			zpdf = fbdwdkde_z(z_grid)
			#zpdf_true = fbdwdkde_z(z)
			ax2.plot(z_grid, zpdf, alpha=0.5, lw=3,label='zbw={0}'.format(zbdwd))
		ax2.scatter(z,np.full(z.shape,zpdf.min()), alpha=0.4)#, c=pdf_true, alpha=0.4)
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
		  s1 = '%f,%f,%f\n'%(datapoints[0][currentIndex], datapoints[1][currentIndex], densities[currentIndex] )
		  fid.write(s1)
		fid.close()
		return out

	def _kde_scipy(self):
		self.data = self.data.T
		'''use default bandwidths'''
#		self.kde = stats.gaussian_kde(self.data)
		'''change bandwidths'''
		self.kde = CustBdwdKDE(self.data,bandwidth=[0.0006,0.001,0.7])

		'''plot out all data points at all hours'''
#		density = self.kde(self.data)
##		print self.data
##		index = density.argsort()
#		spec_hrs = np.full(self.data[0].shape,8)
##		lat, lng, hr, density = self.data[0][index], self.data[1][index], spec_hrs, density[index] # mapping of input data points
#		lat, lng = self.data[0], self.data[1]
#		hr = self.data[2] #at all hours
#		'''matlibplot plotting data points at all hours'''
#		fig, (ax2) = plt.subplots(1,1,subplot_kw=dict(projection='3d'))
#		ax2.set_xlim([self.xmin,self.xmax])
#		ax2.set_ylim([self.ymin,self.ymax])
##		ax2.set_xlim([lat.min(), lat.max()])
##		ax2.set_ylim([lng.min(), lng.max()])
#		ax2.set_zlim([0,24])
#		ax2.scatter(lat, lng, hr,c=density)
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
				if ind!=self.DATAINFO[2]:
					if colstr=="0":
						ndarr = []
						break
				if(colstr==""):
					ndarr = []
					break
				else:
					ndarr = ndarr+(colstr)+","
			#print(ndarr)
			if(ndarr!=[]):
				form_file.write(ndarr[:-1]+"\n")
			record = orig_file.readline()
		orig_file.close()
		form_file.close()

"""A Customized class inheriting stats.gaussian_kde to set fixed bandwidth for overwriting scipy's default covariance determination"""
class CustBdwdKDE(stats.gaussian_kde):
    def __init__(self, dataset, bandwidth=1):
        self.dataset = np.atleast_2d(dataset)
        self.d, self.n = self.dataset.shape
        self.set_bandwidth(bandwidth)

    def set_bandwidth(self, bandwidth):
        if np.isscalar(bandwidth):
            covariance = np.eye(self.d) * bandwidth**2
        else:
            bandwidth = numpy.asarray(bandwidth)
            if bandwidth.shape == (self.d,): # bandwidth is 1-d array of len(self.d)
                covariance = np.diag(bandwidth**2)
            else:
                raise ValueError("'bandwidth' should be a scalar, a 1-d array of length d.")
        self.set_covariance(covariance)

    def set_covariance(self, covariance):
        self.covariance = np.asarray(covariance)
        self.inv_cov = np.linalg.inv(self.covariance)
        self._norm_factor = np.sqrt(numpy.linalg.det(2*numpy.pi*self.covariance)) * self.n
		
dm = DataMiner()
#dm.selectBdwd()
'''plot points and grid densities at a specified hour'''
xmin, ymin = 33.7514, -84.4234#dm.data[0].min(), dm.data[1].min(), 10#
xmax, ymax = 33.7972, -84.3708#dm.data[0].max(), dm.data[1].max(), 11#
lat, lng = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
speci_hr = 20 ## show data at a specific time
hr = np.empty(lat.shape) ##set hr to a specific hour
for i in range(lat.shape[0]):
		for j in range(lat.shape[1]):
			hr[i,j]=speci_hr
hr = np.full(lat.shape,speci_hr)
coords = np.vstack([item.ravel() for item in [lat, lng, hr]]) 
density = dm.getDensity(coords).reshape(lat.shape)
fig3, ax3 = plt.subplots()
ax3.imshow(np.rot90(density), cmap=plt.cm.gist_earth_r,extent=[xmin, xmax, ymin, ymax])
ax3.plot(dm.data[0][dm.data[2,:]==speci_hr], dm.data[1][dm.data[2,:]==speci_hr], 'k.', markersize=2,label='hr={0}'.format(speci_hr))
#ax3.set_xlim([33.739, 33.8])
#ax3.set_ylim([-84.43, -84.36])
ax3.set_xlim([dm.xmin, dm.xmax])
ax3.set_ylim([dm.ymin, dm.ymax])
ax3.legend(loc='upper left')
plt.show()
#		mlab.contour3d(lat, lng, hr, density, opacity=0.5)
#		mlab.axes()
#		mlab.show()

