var map, layer;
function init(){
    map = new OpenLayers.Map('map', {           
        numZoomLevels: 20,
        projection: new OpenLayers.Projection("EPSG:4326"),
        displayProjection: new OpenLayers.Projection("EPSG: 4326"),
        controls: [
            new OpenLayers.Control.Attribution(),
            new OpenLayers.Control.Navigation()
        ],
    });
    
    var mapnik = new OpenLayers.Layer.OSM.Mapnik("Mapnik", {
        displayOutsideMaxExtent: true,
        wrapDateLine: true
    });
    map.addLayer(mapnik);
    map.setBaseLayer(mapnik);
     map.setCenter(new OpenLayers.LonLat(-84.396746,33.7750).transform(
                new OpenLayers.Projection("EPSG:4326"),
                map.getProjectionObject()), 16);

    var size = map.getSize();
    if (size.h > 320) {
        map.addControl(new OpenLayers.Control.PanZoomBar());
    } else {
        map.addControl(new OpenLayers.Control.PanZoom());
    }
    
    var heatmap = new OpenLayers.Layer.HeatCanvas("Heat Canvas", map, {},
            {'step':0.5, 'degree':HeatCanvas.LINEAR, 'opacity':0.7});
    map.addLayer(heatmap);
}
init();
window.map = map;
