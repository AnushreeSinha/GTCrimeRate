import web
from wrapper import Wrapper
from convertcsv import convertCSV

wrapper = Wrapper()
convert = convertCSV()

urls = (
    "/", "index",
    "/getscore", "getscore"
)

class index:
    def GET(self):
        wrapper.callKDEOne()
        out = convert.convertToJSON()
        return out
        

class getscore:
    def GET(self):
        data = web.input()
        baselon = float(data.baselon)
        baselat = float(data.baselat)
        toplon = float(data.toplon)
        toplat = float(data.toplat)
        wrapper.callKDETwo(baselon, baselat,toplon,toplat)
        out = convert.convertToJSON()
        return out

if __name__ == "__main__":
    app = web.application(urls, globals())
    app.run()
