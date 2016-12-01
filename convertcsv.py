import csv
import json

class convertCSV:
    
    def convertToJSON(self):
        csvfile = open('kde_out.csv', 'r')
        jsonfile = open('filejson.json', 'w')
        fieldnames = ("Latitude","Longitude","Score")
        reader = csv.DictReader( csvfile, fieldnames)
        out = json.dumps( [ row for row in reader ] )  
        jsonfile.write(out)
        return out
        