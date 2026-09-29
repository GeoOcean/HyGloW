(() => {
  "use strict";

  const DATA = window.DECK_DATA || {};
  const CONFIG = window.DECK_CONFIG || {};
  const byId = Object.fromEntries((DATA.cases || []).map((c) => [c.id, c]));
  // The case the deck walks through (config.js), falling back to the first one.
  const MAIN = byId[CONFIG.mainCase] || (DATA.cases || [])[0];
  // Main case first everywhere cases are listed (charts, cards).
  const CASES = (DATA.cases || []).slice().sort((a, b) => (b === MAIN) - (a === MAIN));
  const SVG_NS = "http://www.w3.org/2000/svg";

  // Fixed identity per case -- colour follows the case, never its rank.
  const CASE_COLOR = {
    NewSouthWales: "var(--series-1)",
    Netherlands: "var(--series-2)",
    Samoa: "var(--teal)",
  };
  const CASE_LABEL = { NewSouthWales: "New South Wales", Netherlands: "Netherlands", Samoa: "Samoa" };
  const label = (id) => CASE_LABEL[id] || id;

  // Variables shown in the switchable charts; all scored as RMSE.
  const VARS = {
    hs: { name: "Hs", long: "wave height", unit: "m" },
    dir: { name: "Dir", long: "mean direction", unit: "°" },
  };
  // "0.12 m" / "19°"; bare=true drops the unit (axis ticks).
  const fmtVal = (x, v, bare = false) =>
    v === "dir" ? `${Math.round(x)}${bare ? "" : "°"}` : `${x.toFixed(2)}${bare ? "" : " m"}`;
  // Round an axis maximum up to a clean value; returns [max, tick step].
  function niceMax(hi) {
    const raw = hi / 4;
    const mag = 10 ** Math.floor(Math.log10(raw));
    const step = [1, 2, 2.5, 5, 10].map((f) => f * mag).find((st) => st >= raw);
    return [Math.ceil(hi / step - 1e-9) * step, step];
  }

  /* ---------------------------------------------------------------------
   * Config-driven text
   * ------------------------------------------------------------------- */
  document.querySelectorAll("[data-config]").forEach((el) => {
    const key = el.getAttribute("data-config");
    if (CONFIG[key] != null) el.textContent = CONFIG[key];
  });

  /* ---------------------------------------------------------------------
   * Totals + case figures (paths come from data.js, i.e. the portal bundles)
   * ------------------------------------------------------------------- */
  const T = DATA.totals;
  if (T) {
    setText("#t-cases", `${T.cases} regions`);
    setText("#t-goals", `${T.goals} goals`);
    setText("#t-obs", `${T.observations} buoys`);
    // setText("#s-runs", T.training_runs.toLocaleString("en-US"));
  }

  document.querySelectorAll("img[data-case-fig]").forEach((img) => {
    const [caseId, stem] = img.getAttribute("data-case-fig").split(":");
    const src = (caseId === "main" ? MAIN : byId[caseId])?.figures?.[stem];
    if (src) img.src = src;
  });

  if (MAIN) {
    const fields = { name: label(MAIN.id), goals: `${MAIN.n_goals} goals` };
    document.querySelectorAll("[data-main]").forEach((el) => {
      const v = fields[el.getAttribute("data-main")];
      if (v != null) el.textContent = v;
    });
  }

  /* ---------------------------------------------------------------------
   * Check 2 -- linear summation vs dynamic SnapWave, per depth contour
   * ------------------------------------------------------------------- */
  (function renderLinearSummation() {
    const ul = document.querySelector("#ls-points");
    if (!ul) return;
    const order = ["5m", "10m", "20m", "50m"];
    const fmt = (pooled, v) =>
      order
        .filter((k) => pooled?.[v]?.[k])
        .map((k) => `${fmtVal(pooled[v][k].rmse, v)} <span style="color:var(--text-muted)">(${k})</span>`)
        .join(" &middot; ");
    const main = MAIN?.linear_summation;
    if (main?.pooled) {
      ul.insertAdjacentHTML(
        "beforeend",
        `<li><strong>Contour-wide H<sub>s</sub> RMSE against the full simulation</strong>, ${label(MAIN.id)},
         ${main.n_sites.toLocaleString("en-US")} sites: ${fmt(main.pooled, "hs")}.</li>`
      );
      if (main.pooled.dir && Object.keys(main.pooled.dir).length) {
        ul.insertAdjacentHTML("beforeend", `<li><strong>Contour-wide mean direction RMSE</strong>: ${fmt(main.pooled, "dir")}.
          Direction holds at every depth, including 5 m.</li>`);
      }
    }
    const others = CASES.filter((c) => c !== MAIN && c.linear_summation?.pooled);
    if (others.length) {
      ul.insertAdjacentHTML(
        "beforeend",
        `<li>Contour-wide H<sub>s</sub> RMSE in other regions: ${others
          .map((c) => `<strong>${label(c.id)}</strong> ${fmt(c.linear_summation.pooled, "hs")}`)
          .join("; ")}.</li>`
      );
    }
    ul.insertAdjacentHTML(
      "beforeend",
      `<li>Errors are largest right at the shoreline (5 m), where breaking matters most and v1 does not model it
       (unit H<sub>s</sub>), and on reef-fringed islands like Samoa. These are contour-wide scores; next, site by site.</li>`
    );
  })();

  /* ---------------------------------------------------------------------
   * Check 2b -- per-site RMSE spread per depth contour vs the contour-wide RMSE,
   * switchable between Hs / Dir
   * ------------------------------------------------------------------- */
  (function renderSiteSpread() {
    const root = document.querySelector("#spread-chart");
    const ps = MAIN?.per_site;
    if (!root || !ps) return;
    const pooledAll = MAIN.linear_summation?.pooled || {};
    const color = CASE_COLOR[MAIN.id] || "var(--series-1)";
    const rowsFor = (v) =>
      [["20m", "20 m contour"], ["10m", "10 m contour"], ["5m", "5 m contour"], ["buoy", "Buoy sites"]].filter(([k]) => ps[v]?.[k]);

    function draw(v) {
      const P = ps[v], pooled = pooledAll[v] || {}, meta = VARS[v];
      const order = rowsFor(v);
      const W = 460, rowH = 58, m = { t: 8, r: 22, b: 36, l: 112 };
      const H = m.t + order.length * rowH + m.b;
      const hi = Math.max(...order.map(([k]) => Math.max(P[k].q[4], pooled[k]?.rmse ?? 0)));
      const [xMax, tick] = niceMax(hi);
      const X = (x) => m.l + Math.max(0, Math.min(1, x / xMax)) * (W - m.l - m.r);

      const svg = svgEl("svg", { viewBox: `0 0 ${W} ${H}`, role: "img" });
      svg.setAttribute("aria-label", `Per-site ${meta.name} RMSE against dynamic SnapWave, per depth contour, with the contour-wide RMSE`);
      const grid = svgEl("g", { class: "grid" });
      const axis = svgEl("g", { class: "axis" });
      for (let x = 0; x <= xMax + 1e-9; x += tick) {
        grid.appendChild(svgEl("line", { x1: X(x), x2: X(x), y1: m.t, y2: H - m.b }));
        axis.appendChild(svgEl("text", { x: X(x), y: H - m.b + 16, "text-anchor": "middle" }, fmtVal(x, v, true)));
      }
      axis.appendChild(svgEl("text", { x: (m.l + W - m.r) / 2, y: H - 2, "text-anchor": "middle" },
        `${meta.name} error per site, RMSE [${meta.unit}]  ·  lower is better`));
      svg.append(grid, axis);

      order.forEach(([k, name], i) => {
        const g = P[k];
        const [p10, p25, p50, p75, p90] = g.q;
        const cy = m.t + i * rowH + rowH / 2;
        svg.appendChild(svgEl("text", { x: 0, y: cy - 2, class: "lab strong" }, name));
        svg.appendChild(svgEl("text", { x: 0, y: cy + 13, class: "lab" }, `${g.n} sites`));
        // 10-90 % whisker
        svg.appendChild(svgEl("line", { x1: X(p10), x2: X(p90), y1: cy, y2: cy, style: `stroke:${color}`, "stroke-width": 2, "stroke-linecap": "round" }));
        // 25-75 % box
        const bx = X(p25), bw = Math.max(X(p75) - bx, 3);
        svg.appendChild(svgEl("rect", { x: bx, y: cy - 9, width: bw, height: 18, rx: 4, style: `fill:${color};fill-opacity:0.3;stroke:${color}`, "stroke-width": 1.5 }));
        // median
        svg.appendChild(svgEl("circle", { cx: X(p50), cy, r: 6, style: `fill:${color};stroke:var(--surface-2)`, "stroke-width": 2 }));
        svg.appendChild(svgEl("text", { x: X(p50), y: cy - 14, "text-anchor": "middle", class: "lab strong" }, fmtVal(p50, v)));
        // contour-wide RMSE (what Check 2 reports)
        const pk = pooled[k];
        if (pk) {
          const px = X(pk.rmse);
          svg.appendChild(svgEl("path", { d: `M${px},${cy + 9} l6,6 l-6,6 l-6,-6 z`, style: "fill:var(--surface-2);stroke:var(--text-primary)", "stroke-width": 1.5 }));
        }
        const hit = svgEl("rect", { x: m.l, y: cy - rowH / 2, width: W - m.l - m.r, height: rowH, class: "hit" });
        hit.dataset.tip = `${name} · ${meta.name} RMSE · median site ${fmtVal(p50, v)} · middle half ${fmtVal(p25, v)}–${fmtVal(p75, v)}` +
          ` · 10–90 %: ${fmtVal(p10, v)}–${fmtVal(p90, v)}` +
          (pk ? ` · contour-wide RMSE ${fmtVal(pk.rmse, v)}, beaten by ${Math.round(g.frac_below_pooled * 100)}% of sites` : "");
        svg.appendChild(hit);
      });

      root.querySelector("svg")?.remove();
      root.appendChild(svg);
    }

    root.innerHTML = `<p class="chart-title">Metamodel vs full SnapWave, per site</p>
      <p class="chart-sub">${label(MAIN.id)}, ${MAIN.linear_summation?.period?.replace(", ", " + ") || ""}, bulk reconstruction</p>
      <div class="chart-legend">
        <span class="key"><span class="dot" style="background:${color}"></span>median site</span>
        <span class="key"><span class="dot" style="background:${color};opacity:0.35;border-radius:3px"></span>middle half of sites (line: 10–90 %)</span>
        <span class="key"><span class="dot" style="background:var(--surface-2);border:1.5px solid var(--text-primary);border-radius:1px;transform:rotate(45deg) scale(0.8)"></span>contour-wide RMSE</span>
      </div>`;
    root.prepend(varToggle(Object.keys(VARS).filter((v) => ps[v]), "hs", draw));
    draw("hs");
    // the "how to read it" panel uses the same colour as the chart
    document.querySelector("#spread-defs")?.style.setProperty("--case", color);
  })();

  /* ---------------------------------------------------------------------
   * Check 1 -- MDA convergence line chart (median holdout Hs R2)
   * ------------------------------------------------------------------- */
  (function renderConvergence() {
    const root = document.querySelector("#conv-chart");
    const series = CASES.filter((c) => c.convergence?.mda?.length);
    if (!root || !series.length) return;

    function draw(v) {
      const W = 460, H = 280, m = { t: 12, r: 110, b: 34, l: 44 };
      const xs = series.flatMap((c) => c.convergence.mda);
      const ys = series.flatMap((c) => c.convergence[v] || []).filter((y) => y != null);
      const x0 = 0, x1 = Math.max(...xs);
      const [y1, step] = niceMax(Math.max(...ys));
      const X = (x) => m.l + ((x - x0) / (x1 - x0)) * (W - m.l - m.r);
      const Y = (y) => m.t + (1 - y / y1) * (H - m.t - m.b);

      const svg = svgEl("svg", { viewBox: `0 0 ${W} ${H}`, role: "img" });
      svg.setAttribute("aria-label", `Median holdout ${VARS[v].name} RMSE versus number of training runs, per case`);
      const grid = svgEl("g", { class: "grid" });
      const axis = svgEl("g", { class: "axis" });
      for (let y = 0; y <= y1 + 1e-9; y += step) {
        grid.appendChild(svgEl("line", { x1: m.l, x2: W - m.r, y1: Y(y), y2: Y(y) }));
        axis.appendChild(svgEl("text", { x: m.l - 6, y: Y(y) + 4, "text-anchor": "end" }, v === "dir" ? Math.round(y) : y.toFixed(2)));
      }
      [...new Set(xs)].sort((a, b) => a - b).forEach((x) => {
        axis.appendChild(svgEl("text", { x: X(x), y: H - m.b + 16, "text-anchor": "middle" }, x));
      });
      axis.appendChild(
        svgEl("text", { x: (m.l + W - m.r) / 2, y: H - 2, "text-anchor": "middle" }, "training runs per goal (MDA size)")
      );
      svg.append(grid, axis);

      // End labels, nudged apart so close finishes don't collide.
      const last = (c) => c.convergence[v][c.convergence[v].length - 1];
      const ends = series.map((c) => ({ c, y: Y(last(c)) })).sort((a, b) => a.y - b.y);
      // two-line labels (name, then value) need ~26 px each
      for (let i = 1; i < ends.length; i++) ends[i].y = Math.max(ends[i].y, ends[i - 1].y + 26);
      // keep the stack inside the plot: shift up if the last one runs off the bottom
      const over = ends.length ? ends[ends.length - 1].y + 16 - (H - m.b) : 0;
      if (over > 0) ends.forEach((e) => (e.y -= over));

      series.forEach((c) => {
        const color = CASE_COLOR[c.id] || "var(--text-muted)";
        const pts = c.convergence.mda.map((n, i) => [X(n), Y(c.convergence[v][i])]);
        svg.appendChild(
          svgEl("polyline", {
            points: pts.map((p) => p.join(",")).join(" "),
            fill: "none",
            style: `stroke:${color}`,
            "stroke-width": 2,
            "stroke-linejoin": "round",
          })
        );
        pts.forEach(([px, py], i) => {
          svg.appendChild(
            svgEl("circle", { cx: px, cy: py, r: 4, style: `fill:${color};stroke:var(--surface-2)`, "stroke-width": 2 })
          );
          const hit = svgEl("circle", { cx: px, cy: py, r: 12, class: "hit" });
          hit.dataset.tip = `${label(c.id)} · ${c.convergence.mda[i]} runs · ${VARS[v].name} RMSE ${fmtVal(c.convergence[v][i], v)}`;
          svg.appendChild(hit);
        });
        const end = ends.find((e) => e.c === c);
        const lp = pts[pts.length - 1];
        svg.appendChild(svgEl("text", { x: lp[0] + 10, y: end.y + 1, class: "lab strong" }, label(c.id)));
        svg.appendChild(svgEl("text", { x: lp[0] + 10, y: end.y + 14, class: "lab" }, fmtVal(last(c), v)));
      });

      root.querySelector("svg")?.remove();
      root.appendChild(svg);
      const sub = root.querySelector(".chart-sub");
      if (sub) sub.textContent = `Median ${VARS[v].name} RMSE [${VARS[v].unit}] across goals & depth contours · lower is better`;
    }

    root.innerHTML = `<p class="chart-title">Metamodel vs SnapWave, unseen cases</p>
      <p class="chart-sub"></p>
      ${legendHtml(series.map((c) => [label(c.id), CASE_COLOR[c.id]]))}`;
    root.prepend(varToggle(Object.keys(VARS).filter((v) => series.some((c) => c.convergence[v])), "hs", draw));
    draw("hs");
  })();

  /* ---------------------------------------------------------------------
   * Check 3 -- buoys: dumbbell of median Hs RMSE, hindcast -> HyWaves
   * ------------------------------------------------------------------- */
  (function renderBuoyChart() {
    const root = document.querySelector("#buoy-chart");
    const rows = CASES.filter((c) => c.buoys?.hs?.hindcast != null && c.buoys?.hs?.hyglow != null);
    if (!root || !rows.length) return;

    const W = 460, rowH = 62, m = { t: 8, r: 20, b: 34, l: 130 };
    const H = m.t + rows.length * rowH + m.b;
    const xMax = Math.ceil(Math.max(...rows.flatMap((c) => [c.buoys.hs.hindcast, c.buoys.hs.hyglow])) * 4) / 4;
    const X = (v) => m.l + (v / xMax) * (W - m.l - m.r);

    const svg = svgEl("svg", { viewBox: `0 0 ${W} ${H}`, role: "img" });
    svg.setAttribute("aria-label", "Median Hs RMSE at buoys: offshore hindcast versus HyWaves, per case");
    const grid = svgEl("g", { class: "grid" });
    const axis = svgEl("g", { class: "axis" });
    for (let v = 0; v <= xMax + 1e-9; v += 0.25) {
      grid.appendChild(svgEl("line", { x1: X(v), x2: X(v), y1: m.t, y2: H - m.b }));
      axis.appendChild(svgEl("text", { x: X(v), y: H - m.b + 16, "text-anchor": "middle" }, v.toFixed(2)));
    }
    axis.appendChild(svgEl("text", { x: (m.l + W - m.r) / 2, y: H - 2, "text-anchor": "middle" }, "median Hs error, RMSE [m]  ·  lower is better"));
    svg.append(grid, axis);

    rows.forEach((c, i) => {
      const cy = m.t + i * rowH + rowH / 2;
      const a = c.buoys.hs.hindcast, b = c.buoys.hs.hyglow;
      svg.appendChild(svgEl("text", { x: 0, y: cy - 2, class: "lab strong" }, label(c.id)));
      svg.appendChild(
        svgEl("text", { x: 0, y: cy + 13, class: "lab" }, `${c.buoys.n_buoys} ${c.buoys.kind} buoy${c.buoys.n_buoys > 1 ? "s" : ""}`)
      );
      svg.appendChild(
        svgEl("line", { x1: X(a), x2: X(b), y1: cy, y2: cy, style: "stroke:var(--border)", "stroke-width": 4, "stroke-linecap": "round" })
      );
      svg.appendChild(
        svgEl("circle", { cx: X(a), cy, r: 6, style: "fill:var(--surface-2);stroke:var(--text-muted)", "stroke-width": 2 })
      );
      svg.appendChild(
        svgEl("circle", { cx: X(b), cy, r: 7, style: "fill:var(--series-1);stroke:var(--surface-2)", "stroke-width": 2 })
      );
      const bLeft = b < a;
      svg.appendChild(
        svgEl("text", { x: X(b) + (bLeft ? -12 : 12), y: cy - 11, "text-anchor": bLeft ? "end" : "start", class: "lab strong" }, b.toFixed(2))
      );
      svg.appendChild(
        svgEl("text", { x: X(a) + (bLeft ? 12 : -12), y: cy - 11, "text-anchor": bLeft ? "start" : "end", class: "lab" }, a.toFixed(2))
      );
      const hit = svgEl("rect", { x: m.l, y: cy - rowH / 2, width: W - m.l - m.r, height: rowH, class: "hit" });
      hit.dataset.tip = `${label(c.id)} (${c.buoys.period}) · hindcast ${a.toFixed(2)} m → HyWaves ${b.toFixed(2)} m`;
      svg.appendChild(hit);
    });

    root.innerHTML = `<p class="chart-title">Significant wave height error at buoys</p>
      <p class="chart-sub">Median across buoys, hourly data</p>
      <div class="chart-legend">
        <span class="key"><span class="dot" style="background:var(--surface-2);border:2px solid var(--text-muted)"></span>Offshore hindcast, nearest point</span>
        <span class="key"><span class="dot" style="background:var(--series-1)"></span>HyWaves, spectral metamodel</span>
      </div>`;
    root.appendChild(svg);

    // Main-case headline tiles
    const mb = MAIN?.buoys;
    const tiles = document.querySelector("#main-stats");
    if (mb && tiles) {
      setText("#main-head", `${label(MAIN.id)} · ${mb.n_buoys} ${mb.kind} buoys · ${mb.period}`);
      const pairs = mb.per_buoy.filter((r) => r.hindcast != null && r.hyglow != null);
      const better = pairs.filter((r) => r.hyglow < r.hindcast).length;
      tiles.innerHTML =
        stat(`<span class="from">${mb.hs.hindcast.toFixed(2)}</span>${mb.hs.hyglow.toFixed(2)} m`,
          `median Hs error, ${Math.round((1 - mb.hs.hyglow / mb.hs.hindcast) * 100)}% lower than the offshore hindcast`) +
        stat(`${better} / ${pairs.length}`, `buoys where HyWaves beats the offshore hindcast on Hs`) +
        (mb.dir?.hindcast != null && mb.dir?.hyglow != null
          ? stat(`<span class="from">${mb.dir.hindcast.toFixed(0)}&deg;</span>${mb.dir.hyglow.toFixed(0)}&deg;`,
            `median direction error: ${mb.dir.hyglow < mb.dir.hindcast ? "lower" : "no gain yet"}` +
            ` (better at ${mb.per_buoy.filter((r) => r.dir_hyglow != null && r.dir_hindcast != null && r.dir_hyglow < r.dir_hindcast).length}` +
            ` of ${mb.per_buoy.filter((r) => r.dir_hyglow != null && r.dir_hindcast != null).length} buoys)`)
          : "");
    }
  })();

  /* ---------------------------------------------------------------------
   * One site up close -- Bengello
   * ------------------------------------------------------------------- */
  (function renderSite() {
    const el = document.querySelector("#site-stats");
    const b = MAIN?.buoys?.per_buoy?.find((r) => r.buoy === CONFIG.featuredBuoy);
    if (!el || !b) return;
    const fig = document.querySelector("#site-fig");
    if (fig && b.figure) fig.src = b.figure;
    const tiles = [];
    if (b.hindcast != null && b.hyglow != null) {
      tiles.push(
        stat(`<span class="from">${b.hindcast.toFixed(2)}</span>${b.hyglow.toFixed(2)} m`,
          "Hs error here: offshore hindcast (struck through) vs HyWaves"),
        stat(`${Math.round((1 - b.hyglow / b.hindcast) * 100)}%`, "lower error than using the offshore data directly")
      );
    }
    if (b.dir_hindcast != null && b.dir_hyglow != null) {
      tiles.push(stat(`<span class="from">${b.dir_hindcast.toFixed(0)}&deg;</span>${b.dir_hyglow.toFixed(0)}&deg;`,
        "mean direction error: offshore hindcast vs HyWaves"));
    }
    el.innerHTML = tiles.join("");
  })();

  /* ---------------------------------------------------------------------
   * Case cards
   * ------------------------------------------------------------------- */
  (function renderCases() {
    const root = document.querySelector("#case-cards");
    if (!root) return;
    root.innerHTML = CASES.map((c) => {
      const fig = c.figures?.fig01_domain;
      const b = c.buoys;
      const pills = [
        `${c.n_goals} goals`,
        `${c.n_observations} buoys`,
        c.n_training_runs ? `${c.n_training_runs.toLocaleString("en-US")} runs` : null,
        c.linear_summation?.n_sites ? `${c.linear_summation.n_sites.toLocaleString("en-US")} sites` : null,
      ].filter(Boolean);
      const headline = b
        ? `Hs error at ${b.kind} buoys: <strong>${b.hs.hindcast.toFixed(2)} &rarr; ${b.hs.hyglow.toFixed(2)} m</strong>`
        : "";
      return `<div class="card case-card" style="border-top:3px solid ${CASE_COLOR[c.id] || "var(--accent)"}">
        ${fig ? `<img src="${fig}" alt="${escapeHtml(c.id)} domain map" data-img="${fig}" data-alt="${escapeHtml(c.id)} domain" />` : ""}
        <div class="case-body">
          <h3>${escapeHtml(label(c.id))}</h3>
          <p class="region">${escapeHtml(c.region)}</p>
          <div class="facts">${pills.map((p) => `<span class="pill">${p}</span>`).join("")}</div>
          <p style="font-size:0.88rem;margin:0.7rem 0 0;">${headline}</p>
        </div>
      </div>`;
    }).join("");
  })();

  /* ---------------------------------------------------------------------
   * GUI screenshot switcher
   * ------------------------------------------------------------------- */
  // Screenshots live per case under assets/gui/<Case>/ -- follow mainCase.
  const shotImg = document.querySelector("#shot-img");
  const shotSrc = (tab) => `assets/gui/${MAIN?.id}/${tab.getAttribute("data-shot")}`;
  document.querySelectorAll(".shot-tab").forEach((tab) => {
    tab.addEventListener("click", () => {
      document.querySelectorAll(".shot-tab").forEach((t) => t.classList.toggle("active", t === tab));
      if (shotImg) shotImg.src = shotSrc(tab);
    });
  });
  const firstShot = document.querySelector(".shot-tab.active");
  if (shotImg && firstShot) shotImg.src = shotSrc(firstShot);

  /* ---------------------------------------------------------------------
   * Chart tooltips (any [data-tip] inside a chart)
   * ------------------------------------------------------------------- */
  const tip = document.querySelector("#chart-tip");
  document.addEventListener("mousemove", (e) => {
    const t = e.target.closest?.("[data-tip]");
    if (!tip) return;
    if (!t) return tip.classList.remove("show");
    tip.textContent = t.dataset.tip;
    tip.style.left = `${e.clientX + 14}px`;
    tip.style.top = `${e.clientY - 30}px`;
    tip.classList.add("show");
  });

  const genNote = document.querySelector("#gen-note");
  if (genNote && DATA.generated_at) {
    genNote.textContent = `data.js ${DATA.generated_at} · rerun presentation/data/build_data.py to refresh`;
  }

  /* ---------------------------------------------------------------------
   * Slide engine, with short/full mode
   * ------------------------------------------------------------------- */
  const allSlides = Array.from(document.querySelectorAll(".slide"));
  let mode = readStored("hyglow-deck-mode") || CONFIG.defaultMode || "full";
  let slides = [];
  let current = 0;

  const counterEl = document.querySelector("#slide-counter");
  const progressEl = document.querySelector("#progress-fill");
  const modeBtn = document.querySelector("#mode-btn");

  function applyMode(keepSlide) {
    document.body.classList.toggle("mode-short", mode === "short");
    slides = allSlides.filter((s) => mode === "full" || s.getAttribute("data-depth") !== "deep");
    if (modeBtn) {
      modeBtn.innerHTML = mode === "short" ? "<strong>Short</strong> deck" : "<strong>Full</strong> deck";
    }
    // Stay on the same slide if it survives the switch, else the nearest one before it.
    let idx = slides.indexOf(keepSlide);
    if (idx < 0 && keepSlide) {
      const pos = allSlides.indexOf(keepSlide);
      idx = slides.filter((s) => allSlides.indexOf(s) < pos).length - 1;
    }
    show(Math.max(0, idx));
  }

  function toggleMode() {
    const keep = slides[current];
    mode = mode === "short" ? "full" : "short";
    writeStored("hyglow-deck-mode", mode);
    applyMode(keep);
  }

  function show(index) {
    current = Math.max(0, Math.min(slides.length - 1, index));
    allSlides.forEach((s) => s.classList.toggle("active", s === slides[current]));
    if (counterEl) counterEl.textContent = `${current + 1} / ${slides.length}`;
    if (progressEl) progressEl.style.width = `${((current + 1) / slides.length) * 100}%`;
    history.replaceState(null, "", `#${current + 1}`);
  }

  const next = () => show(current + 1);
  const prev = () => show(current - 1);

  document.querySelector("#next-btn")?.addEventListener("click", next);
  document.querySelector("#prev-btn")?.addEventListener("click", prev);
  modeBtn?.addEventListener("click", toggleMode);

  window.addEventListener("keydown", (e) => {
    if (lightboxOpen()) {
      if (e.key === "Escape") closeLightbox();
      return;
    }
    if (["ArrowRight", "ArrowDown", "PageDown", " "].includes(e.key)) {
      e.preventDefault();
      next();
    } else if (["ArrowLeft", "ArrowUp", "PageUp"].includes(e.key)) {
      e.preventDefault();
      prev();
    } else if (e.key === "Home") {
      show(0);
    } else if (e.key === "End") {
      show(slides.length - 1);
    } else if (e.key === "f" || e.key === "F") {
      toggleFullscreen();
    } else if (e.key === "s" || e.key === "S") {
      toggleMode();
    }
  });

  let touchStartX = null;
  document.addEventListener("touchstart", (e) => (touchStartX = e.touches[0].clientX));
  document.addEventListener("touchend", (e) => {
    if (touchStartX == null) return;
    const dx = e.changedTouches[0].clientX - touchStartX;
    if (Math.abs(dx) > 50) (dx < 0 ? next : prev)();
    touchStartX = null;
  });

  // Skip the fade-in on first paint -- only animate slide-to-slide moves.
  document.body.classList.add("preload");
  const startIndex = (parseInt(location.hash.replace("#", ""), 10) || 1) - 1;
  applyMode(null);
  show(startIndex);
  requestAnimationFrame(() => requestAnimationFrame(() => document.body.classList.remove("preload")));

  /* ---------------------------------------------------------------------
   * Lightbox
   * ------------------------------------------------------------------- */
  const lightbox = document.querySelector("#lightbox");
  const lightboxImg = lightbox?.querySelector("img");

  function openLightbox(src, alt) {
    if (!lightbox || !lightboxImg) return;
    lightboxImg.src = src;
    lightboxImg.alt = alt || "";
    lightbox.classList.add("open");
  }
  function closeLightbox() {
    lightbox?.classList.remove("open");
  }
  function lightboxOpen() {
    return !!lightbox?.classList.contains("open");
  }

  document.querySelectorAll(".fig:not([data-no-zoom]) img").forEach((img) => {
    img.addEventListener("click", (e) => {
      e.stopPropagation();
      openLightbox(img.src, img.alt);
    });
  });
  document.querySelectorAll(".fig[data-no-zoom]").forEach((f) => (f.style.cursor = "default"));
  lightbox?.addEventListener("click", closeLightbox);

  // Everything else clickable-to-zoom (cards, case thumbnails) -- runs after
  // the renderers above so data.js-built elements are covered too.
  document.querySelectorAll("[data-img]").forEach((el) => {
    el.addEventListener("click", (e) => {
      e.stopPropagation();
      openLightbox(el.getAttribute("data-img"), el.getAttribute("data-alt"));
    });
  });

  /* ---------------------------------------------------------------------
   * Theme toggle (per-viewer convenience only)
   * ------------------------------------------------------------------- */
  const themeBtn = document.querySelector("#theme-btn");
  function updateThemeLabel() {
    if (!themeBtn) return;
    const cur = document.documentElement.getAttribute("data-theme");
    themeBtn.textContent = cur === "dark" ? "☀ Light" : cur === "light" ? "◑ Auto" : "☽ Dark";
  }
  themeBtn?.addEventListener("click", () => {
    const cur = document.documentElement.getAttribute("data-theme");
    const nextTheme = cur === "dark" ? "light" : cur === "light" ? null : "dark";
    if (nextTheme) document.documentElement.setAttribute("data-theme", nextTheme);
    else document.documentElement.removeAttribute("data-theme");
    writeStored("deck-theme", nextTheme);
    updateThemeLabel();
  });
  const storedTheme = readStored("deck-theme");
  if (storedTheme === "light" || storedTheme === "dark") {
    document.documentElement.setAttribute("data-theme", storedTheme);
  }
  updateThemeLabel();

  function toggleFullscreen() {
    if (!document.fullscreenElement) document.documentElement.requestFullscreen?.();
    else document.exitFullscreen?.();
  }

  /* ---------------------------------------------------------------------
   * Helpers
   * ------------------------------------------------------------------- */
  function svgEl(tag, attrs = {}, text) {
    const el = document.createElementNS(SVG_NS, tag);
    Object.entries(attrs).forEach(([k, v]) => el.setAttribute(k, v));
    if (text != null) el.textContent = text;
    return el;
  }

  // Segmented Hs / Dir / Spr switch; calls onPick(var) on click.
  function varToggle(vars, active, onPick) {
    const wrap = document.createElement("div");
    wrap.className = "var-toggle";
    wrap.setAttribute("role", "tablist");
    vars.forEach((v) => {
      const b = document.createElement("button");
      b.type = "button";
      b.textContent = VARS[v]?.name || v;
      b.classList.toggle("active", v === active);
      b.addEventListener("click", (e) => {
        e.stopPropagation();
        wrap.querySelectorAll("button").forEach((x) => x.classList.toggle("active", x === b));
        b.blur(); // keep Space / arrows driving the deck
        onPick(v);
      });
      wrap.appendChild(b);
    });
    return wrap;
  }

  function legendHtml(items) {
    return `<div class="chart-legend">${items
      .map(([name, color]) => `<span class="key"><span class="dot" style="background:${color}"></span>${escapeHtml(name)}</span>`)
      .join("")}</div>`;
  }

  function stat(num, lbl) {
    return `<div class="stat"><div class="num">${num}</div><div class="lbl">${lbl}</div></div>`;
  }

  function setText(sel, txt) {
    const el = document.querySelector(sel);
    if (el) el.textContent = txt;
  }

  function escapeHtml(str) {
    return String(str ?? "").replace(/[&<>"']/g, (c) =>
      ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c])
    );
  }

  function readStored(key) {
    try {
      return localStorage.getItem(key);
    } catch (_) {
      return null; // private mode / blocked storage
    }
  }

  function writeStored(key, value) {
    try {
      if (value == null) localStorage.removeItem(key);
      else localStorage.setItem(key, value);
    } catch (_) {
      /* ignore */
    }
  }
})();
