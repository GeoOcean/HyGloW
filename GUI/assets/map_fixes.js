// Defensive fix for a MapLibre GL rendering race: on a cold page load
// (empty browser cache), the very first swap of the map's figure — e.g. the
// landing/world view to a picked case — can leave the underlying GL canvas
// internally correct (its own getCenter()/getZoom()/isStyleLoaded() all
// report the right values) but visually blank or stuck, as if it never
// painted. Nudging Plotly's own resize/redraw path after the figure changes
// forces the repaint — but a single nudge fired too early (before the GL
// context is actually ready, e.g. still waiting on the base-tile fetch on a
// slow/cold connection) does nothing, and there's no reliable "now it's
// ready" event exposed through Plotly's own API to wait on instead. So
// retry across a window covering "instant" through "slow cold load" rather
// than guessing one delay — each call is cheap and a no-op once the map is
// already showing correctly, so repeating it is harmless.
//
// (A dynamic best-fit zoom for the landing/world view — computed from the
// real container width via map.project(), applied through Plotly.relayout —
// was tried here too, but Plotly's own map-subplot relayout handling
// doesn't reliably keep it: the GL camera would briefly show the corrected
// zoom and then silently revert to the Python-declared value on its own,
// with no exception thrown to explain why. Rather than ship something that
// flickers unpredictably, the world view's zoom is just a fixed, reasonable
// default (map_figure.py) instead.)
window.dash_clientside = Object.assign({}, window.dash_clientside, {
    map_fixes: {
        force_repaint: function () {
            const wrapper = document.getElementById("domain-map");
            const gd = wrapper && wrapper.querySelector(".js-plotly-plot");
            if (!gd || !window.Plotly || !window.Plotly.Plots) {
                return window.dash_clientside.no_update;
            }
            [60, 250, 600, 1200, 2500].forEach((ms) => {
                setTimeout(() => {
                    try {
                        window.Plotly.Plots.resize(gd);
                    } catch (e) {
                        // Nothing useful to do if resize itself throws — the
                        // figure update already landed either way.
                    }
                }, ms);
            });
            return window.dash_clientside.no_update;
        },
    },
});
