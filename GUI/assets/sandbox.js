// Sandbox-mode AoI drawing: Plotly's drawrect/drawclosedpath tools report
// shapes in *paper* coordinates (0-1 fractions of the plot area) on a
// map/mapbox subplot rather than lon/lat, since a screen-space rectangle
// isn't a rectangle in a geographic projection. Convert back to real
// coordinates using the underlying MapLibre GL instance's own unproject(),
// which already accounts for the map's current center/zoom exactly.
window.dash_clientside = Object.assign({}, window.dash_clientside, {
    sandbox: {
        extract_aoi: function (relayoutData) {
            const wrapper = document.getElementById("domain-map");
            const gd = wrapper && wrapper.querySelector(".js-plotly-plot");
            if (!gd || !gd._fullLayout || !gd._fullLayout.map) {
                return window.dash_clientside.no_update;
            }
            const shapes = (gd.layout && gd.layout.shapes) || [];
            if (!shapes.length) {
                return window.dash_clientside.no_update;
            }
            const mapGl = gd._fullLayout.map._subplot.map;
            const rect = gd.getBoundingClientRect();

            function toLonLat(xFrac, yFrac) {
                const px = xFrac * rect.width;
                const py = (1 - yFrac) * rect.height;
                const ll = mapGl.unproject([px, py]);
                return [ll.lng, ll.lat];
            }

            const shape = shapes[shapes.length - 1];
            let paperRing;
            if (shape.type === "rect") {
                paperRing = [
                    [shape.x0, shape.y0],
                    [shape.x1, shape.y0],
                    [shape.x1, shape.y1],
                    [shape.x0, shape.y1],
                ];
            } else if (shape.type === "path" && shape.path) {
                paperRing = shape.path
                    .replace(/^M/, "")
                    .replace(/Z$/, "")
                    .split("L")
                    .map((pair) => pair.split(",").map(Number));
            } else {
                return window.dash_clientside.no_update;
            }
            if (paperRing.length < 3) {
                return window.dash_clientside.no_update;
            }

            // Only one AoI at a time — rect and freeform can otherwise pile
            // up (Plotly's draw tools append, they don't replace), leaving
            // it ambiguous which one export would actually use. The most
            // recently finished shape always wins; anything older is
            // dropped as soon as the new one completes. This relayout call
            // re-fires this same callback, but by then shapes.length is
            // already 1, so it's a no-op on the second pass.
            if (shapes.length > 1 && window.Plotly) {
                window.Plotly.relayout(gd, { shapes: [shape] });
            }

            const ring = paperRing.map(([x, y]) => toLonLat(x, y));
            ring.push(ring[0]);
            return ring;
        },
    },
});
