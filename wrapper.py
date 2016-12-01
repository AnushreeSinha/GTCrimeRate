from kde_model import DataMiner
import datetime
from numpy import array

'''dataMiner = DataMiner()
baselon = -84.409860
baselat = 33.773249
toplon = -84.391138
toplat = 33.786683
lats = list();
longs = list();
hrs = list();
now = datetime.datetime.now()
hour = now.hour
for i in range(0, 10):
    lat_range = baselat+((toplat-baselat)*(i/10))
    for j in range(0, 10):
        lats.append(lat_range)
        lon_range = baselon+((toplon-baselon)*(j/10))
        longs.append(lon_range)
        hrs.append(hour)
datapoints = []
print(len(lats))
print(len(longs))
print(len(hrs))
datapoints.append(lats)
datapoints.append(longs)
datapoints.append(hrs)
#print(datapoints)
print(len(datapoints))
dp = array(datapoints)
dataMiner.getDensity(dp)
'''
class Wrapper:
    
    def callKDEOne(self):
        dataMiner = DataMiner()
        baselon = -84.409860
        baselat = 33.773249
        toplon = -84.391138
        toplat = 33.786683
        lats = list();
        longs = list();
        hrs = list();
        now = datetime.datetime.now()
        hour = now.hour
        for i in range(0, 100):
           lat_range = baselat+((toplat-baselat)*(i/100))
           for j in range(0, 100):
               lats.append(lat_range)
               lon_range = baselon+((toplon-baselon)*(j/100))
               longs.append(lon_range)
               hrs.append(hour)
        datapoints = []
        datapoints.append(lats)
        datapoints.append(longs)
        datapoints.append(hrs)
        dp = array(datapoints)
        dataMiner.getDensity(dp)

        
    def callKDETwo(self, baselon, baselat, toplon, toplat):
        dataMiner = DataMiner()
        lats = list();
        longs = list();
        hrs = list();
        now = datetime.datetime.now()
        hour = now.hour
        for i in range(0, 100):
           lat_range = baselat+((toplat-baselat)*(i/100))
           for j in range(0, 100):
               lats.append(lat_range)
               lon_range = baselon+((toplon-baselon)*(j/100))
               longs.append(lon_range)
               hrs.append(hour)
        datapoints = []
        datapoints.append(lats)
        datapoints.append(longs)
        datapoints.append(hrs)
        dp = array(datapoints)
        dataMiner.getDensity(dp)
