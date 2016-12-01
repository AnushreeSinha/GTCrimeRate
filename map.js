
            var map, layer;
            function init(){
                map = new OpenLayers.Map ("map", {
                  controls: [
                      new OpenLayers.Control.Attribution(),
                      new OpenLayers.Control.Navigation()
                  ],
                  maxExtent: new OpenLayers.Bounds(-20037508.34,-20037508.34,
                                                   20037508.34,20037508.34),
                  numZoomLevels: 12,
                  maxResolution: 156543.0339,
                  displayProjection: new OpenLayers.Projection("EPSG:4326"),
                  units: 'm',
                  projection: new OpenLayers.Projection("EPSG:4326")
                });
                
                var mapnik = new OpenLayers.Layer.OSM.Mapnik("Mapnik", {
                   displayOutsideMaxExtent: true,
                   wrapDateLine: true
                });
                map.addLayer(mapnik);
                map.setBaseLayer(mapnik);

                //var bounds = OpenLayers.Bounds.fromArray([70.4,15.2,136.2,53.7])
                //        .transform(map.displayProjection, map.getProjectionObject());
                //map.zoomToExtent(bounds);
                map.setCenter(new OpenLayers.LonLat(-84.396746,33.7750).transform(map.displayProjection, map.getProjectionObject()), 16);

                var size = map.getSize();
                if (size.h > 320) {
                    map.addControl(new OpenLayers.Control.PanZoomBar());
                } else {
                    map.addControl(new OpenLayers.Control.PanZoom());
                }
                
                var heatmap = new OpenLayers.Layer.HeatCanvas("Heat Canvas", map, {},
                        {'step':5, 'degree':HeatCanvas.LINEAR, 'opacity':0.7});
                // var data = [[33.7784151,-84.3999697,0], [33.7814969,-84.3947513,14], [33.7779735,-84.4042067,13]];
                var rawFile = new XMLHttpRequest();
                var data;
                rawFile.open("GET", "kde_out.csv", false);
                rawFile.onreadystatechange = function ()
                {
                    if(rawFile.readyState === 4)
                    {
                        if(rawFile.status === 200 || rawFile.status == 0)
                        {
                            var allText = rawFile.responseText;
                            data = $.csv.toArrays(allText);
                    }
                }
                }
                rawFile.send(null);
                console.log(data);
                for(var i=0,l=data.length; i<l; i++) {
                    heatmap.pushData(data[i][0], data[i][1], data[i][2]);
                }
                map.addLayer(heatmap);
    
           var markers = new OpenLayers.Layer.Markers( "Markers" );
           console.log(map);
            map.addLayer(markers);
            
            markers.addMarker(new OpenLayers.Marker(new OpenLayers.LonLat(-84.396746,33.7750).transform(map.displayProjection, map.getProjectionObject())));
    }
            window.map = map;