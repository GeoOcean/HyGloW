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
    setText("#t-obs", `${T.buoys} buoys`);
  }

  document.querySelectorAll("img[data-case-fig]").forEach((img) => {
    const [caseId, stem] = img.getAttribute("data-case-fig").split(":");
    const src = (caseId === "main" ? MAIN : byId[caseId])?.figures?.[stem];
    if (src) img.src = src;
  });

  if (MAIN) {
    const mb = MAIN.buoys;
    const num = (n) => n?.toLocaleString("en-US");
    const fields = {
      name: label(MAIN.id),
      goals: `${MAIN.n_goals} goals`,
      runs: num(MAIN.runs_per_goal),
      "runs-total": num(MAIN.n_training_runs),
      "buoy-period": mb?.period,
      reconstruct: mb?.reconstruct_seconds
        ? `~${Math.max(1, Math.round(mb.reconstruct_seconds / 60))} min for ${mb.period} at every buoy`
        : null,
    };
    document.querySelectorAll("[data-main]").forEach((el) => {
      const v = fields[el.getAttribute("data-main")];
      if (v != null) el.textContent = v;
    });
  }

  /* ---------------------------------------------------------------------
   * Check 1 -- MDA convergence line chart (median holdout RMSE)
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

    // Headline: the range across cases at each one's largest MDA size.
    const range = (v, f) => {
      const xs = series.map((c) => f(c.convergence[v][c.convergence[v].length - 1]));
      const lo = Math.min(...xs), hi = Math.max(...xs);
      return lo === hi ? `${lo}` : `${lo}&ndash;${hi}`;
    };
    const sizes = series.map((c) => c.convergence.mda[c.convergence.mda.length - 1]);
    const runs = Math.min(...sizes) === Math.max(...sizes) ? `${sizes[0]}` : `${Math.min(...sizes)}&ndash;${Math.max(...sizes)}`;
    document.querySelector("#conv-headline")?.insertAdjacentHTML("afterbegin",
      `With <strong>${runs} training runs</strong> per goal, ${series.length > 1 ? `all ${series.length} regions reproduce` : "the metamodel reproduces"}
       the physics model to within <strong>${range("hs", (x) => Math.round(x * 100))} cm</strong> of H<sub>s</sub>
       and <strong>${range("dir", Math.round)}&deg;</strong> of mean direction on unseen cases.`);
  })();

  /* ---------------------------------------------------------------------
   * One site up close -- config.featuredBuoy
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
   * The whole flow -- stages revealed by the slide's steps, boxes explain
   * themselves in the detail bar.
   * ------------------------------------------------------------------- */
  (function renderFlow() {
    const root = document.querySelector("#hf");
    const detail = document.querySelector("#hf-detail");
    if (!root || !detail) return;
    const slide = root.closest(".slide");
    const stages = root.querySelectorAll(".hf-stage");
    const boxes = root.querySelectorAll(".hf-box");
    const hint = detail.innerHTML;

    slide.addEventListener("step", (e) => {
      stages.forEach((st) => st.classList.toggle("hidden", parseInt(st.dataset.step, 10) > e.detail));
      boxes.forEach((b) => b.classList.remove("on"));
      detail.innerHTML = hint;
    });
    boxes.forEach((b) => {
      b.addEventListener("click", () => {
        boxes.forEach((o) => o.classList.toggle("on", o === b));
        const name = b.querySelector("span").firstChild.textContent.trim();
        detail.innerHTML = `<strong>${escapeHtml(name)}</strong> &mdash; ${escapeHtml(b.dataset.text)}`;
      });
    });
  })();

  /* ---------------------------------------------------------------------
   * Each goal alone, then add -- two JONSWAP spectra and their sum
   * (illustrative shapes, not case data).
   * ------------------------------------------------------------------- */
  (function renderSum() {
    const root = document.querySelector("#sum-sketch");
    if (!root) return;
    const W = 200, H = 110, N = 120, f0 = 0.03, f1 = 0.3;
    const fs = Array.from({ length: N }, (_, i) => f0 + ((f1 - f0) * i) / (N - 1));
    const jonswap = (fp, scale) =>
      fs.map((f) => {
        const s = f <= fp ? 0.07 : 0.09;
        const r = Math.exp(-((f - fp) ** 2) / (2 * s * s * fp * fp));
        return scale * f ** -5 * Math.exp(-1.25 * (fp / f) ** 4) * 3.3 ** r * fp ** 4;
      });
    const swell = jonswap(0.075, 1.6);
    const sea = jonswap(0.17, 0.5);
    const sum = swell.map((v, i) => v + sea[i]);
    const top = Math.max(...sum) * 1.08;
    const pts = (e) => e.map((v, i) => `${((fs[i] - f0) / (f1 - f0)) * W},${H - (v / top) * (H - 6)}`);

    function panel(series, caption) {
      const svg = svgEl("svg", { viewBox: `0 0 ${W} ${H + 4}`, role: "img" });
      svg.setAttribute("aria-label", caption);
      svg.appendChild(svgEl("line", { class: "axis-line", x1: 0, x2: W, y1: H, y2: H }));
      series.forEach(([e, color]) => {
        const p = pts(e);
        svg.appendChild(svgEl("polygon", { class: "area", points: `0,${H} ${p.join(" ")} ${W},${H}`, style: `fill:${color}` }));
        svg.appendChild(svgEl("polyline", { class: "curve", points: p.join(" "), style: `stroke:${color}` }));
      });
      const fig = document.createElement("figure");
      fig.appendChild(svg);
      fig.insertAdjacentHTML("beforeend", `<figcaption>${caption}</figcaption>`);
      return fig;
    }
    const op = (t) => Object.assign(document.createElement("span"), { className: "op", textContent: t });
    root.append(
      panel([[swell, "var(--series-1)"]], "Goal A: long swell"),
      op("+"),
      panel([[sea, "var(--series-2)"]], "Goal B: wind sea"),
      op("="),
      panel([[sum, "var(--violet)"]], "At the coast: both"),
    );
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
        b ? `${b.n_buoys} buoys validated` : null,
        c.n_training_runs ? `${c.n_training_runs.toLocaleString("en-US")} runs` : null,
        `${c.n_sites.toLocaleString("en-US")} sites`,
      ].filter(Boolean);
      const headline = b
        ? `Hs error at ${b.kind} buoys: <strong>${b.hs.hindcast.toFixed(2)} &rarr; ${b.hs.hyglow.toFixed(2)} m</strong>`
        : "";
      return `<div class="card case-card" style="border-top:3px solid ${CASE_COLOR[c.id] || "var(--accent)"}">
        ${fig ? `<img loading="lazy" decoding="async" src="${fig}" alt="${escapeHtml(c.id)} domain map" data-img="${fig}" data-alt="${escapeHtml(c.id)} domain" />` : ""}
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
  // Screenshots of the main case (deck/assets/gui/<Case>/, listed in data.js).
  const shotImg = document.querySelector("#shot-img");
  const showShot = (tab) => {
    const src = MAIN?.gui?.[tab.getAttribute("data-shot")];
    if (shotImg && src) shotImg.src = src;
  };
  document.querySelectorAll(".shot-tab").forEach((tab) => {
    tab.addEventListener("click", () => {
      document.querySelectorAll(".shot-tab").forEach((t) => t.classList.toggle("active", t === tab));
      showShot(tab);
    });
  });
  const firstShot = document.querySelector(".shot-tab.active");
  if (firstShot) showShot(firstShot);

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
    genNote.textContent = `Numbers from the published case portals, built ${DATA.generated_at}`;
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

  // Slides with data-steps="N" reveal N more steps (next/prev) before moving
  // on; each step change fires a "step" event on the slide.
  const stepsOf = (s) => parseInt(s?.dataset.steps || "0", 10);
  const stepOf = (s) => parseInt(s?.dataset.step || "0", 10);
  function setStep(slide, k) {
    slide.dataset.step = k;
    slide.dispatchEvent(new CustomEvent("step", { detail: k }));
  }

  // atEnd: enter the slide fully revealed (when stepping back into it).
  function show(index, atEnd = false) {
    current = Math.max(0, Math.min(slides.length - 1, index));
    allSlides.forEach((s) => s.classList.toggle("active", s === slides[current]));
    const s = slides[current];
    if (stepsOf(s)) setStep(s, atEnd ? stepsOf(s) : 0);
    if (counterEl) counterEl.textContent = `${current + 1} / ${slides.length}`;
    if (progressEl) progressEl.style.width = `${((current + 1) / slides.length) * 100}%`;
    history.replaceState(null, "", `#${current + 1}`);
  }

  function next() {
    const s = slides[current];
    if (stepOf(s) < stepsOf(s)) setStep(s, stepOf(s) + 1);
    else show(current + 1);
  }
  function prev() {
    const s = slides[current];
    if (stepOf(s) > 0) setStep(s, stepOf(s) - 1);
    else show(current - 1, true);
  }

  document.querySelector("#next-btn")?.addEventListener("click", next);
  document.querySelector("#prev-btn")?.addEventListener("click", prev);
  modeBtn?.addEventListener("click", toggleMode);

  window.addEventListener("keydown", (e) => {
    if (codeModalOpen()) {
      if (e.key === "Escape") closeCodeModal();
      return;
    }
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
   * Code modal -- clickable cards with data-code="<template id>" (the
   * bluemath-tk toolkit slide) show that <template>'s snippet in a popup.
   * ------------------------------------------------------------------- */
  const codeModal = document.querySelector("#code-modal");
  const codeModalCode = document.querySelector("#code-modal-code");
  const codeModalTitle = document.querySelector("#code-modal-title");

  function openCodeModal(templateId, title) {
    const tpl = document.getElementById(templateId);
    if (!codeModal || !codeModalCode || !tpl) return;
    // Snippets live in <template><pre>: the <pre> keeps HTML formatters from
    // re-indenting the code.
    const pre = tpl.content.querySelector("pre");
    codeModalCode.innerHTML = pre ? pre.innerHTML : tpl.innerHTML;
    codeModalCode.scrollTop = 0;
    if (codeModalTitle) codeModalTitle.textContent = title || "";
    codeModal.classList.add("open");
  }
  function closeCodeModal() {
    codeModal?.classList.remove("open");
  }
  function codeModalOpen() {
    return !!codeModal?.classList.contains("open");
  }

  document.querySelectorAll("[data-code]").forEach((el) => {
    el.addEventListener("click", (e) => {
      if (e.target.closest("a")) return;
      openCodeModal(el.getAttribute("data-code"), el.getAttribute("data-title"));
    });
  });
  document.querySelector("#code-modal-close")?.addEventListener("click", closeCodeModal);
  codeModal?.addEventListener("click", (e) => {
    if (e.target === codeModal) closeCodeModal();
  });

  /* ---------------------------------------------------------------------
   * Wrappers, live -- step-by-step animation of the BinWaves propagation
   * (templates -> build_cases -> run_cases -> postprocess_cases). Tile
   * names are a real, evenly spaced sample of BinWaves/Paradip_CASES/.
   * ------------------------------------------------------------------- */
  (function wrapperAnimation() {
    const root = document.querySelector("#wa");
    if (!root) return;

    const TOTAL = 1260;
    const NAMES = [
      "dm5.0fp0.035", "dm5.0fp0.281", "dm15.0fp0.074", "dm15.0fp0.603", "dm25.0fp0.175",
      "dm35.0fp0.051", "dm35.0fp0.374", "dm45.0fp0.108", "dm45.0fp0.883", "dm55.0fp0.232",
      "dm65.0fp0.067", "dm65.0fp0.548", "dm75.0fp0.144", "dm85.0fp0.042", "dm85.0fp0.340",
      "dm95.0fp0.099", "dm95.0fp0.729", "dm105.0fp0.211", "dm115.0fp0.061", "dm115.0fp0.453",
      "dm125.0fp0.131", "dm135.0fp0.038", "dm135.0fp0.309", "dm145.0fp0.081", "dm145.0fp0.663",
      "dm155.0fp0.192", "dm165.0fp0.051", "dm165.0fp0.412", "dm175.0fp0.119", "dm185.0fp0.035",
      "dm185.0fp0.256", "dm195.0fp0.074", "dm195.0fp0.603", "dm205.0fp0.159", "dm215.0fp0.046",
      "dm215.0fp0.374", "dm225.0fp0.099", "dm225.0fp0.802", "dm235.0fp0.232", "dm245.0fp0.067",
      "dm245.0fp0.498", "dm255.0fp0.144", "dm265.0fp0.042", "dm265.0fp0.309", "dm275.0fp0.090",
      "dm275.0fp0.729", "dm285.0fp0.211", "dm295.0fp0.056", "dm295.0fp0.453", "dm305.0fp0.131",
      "dm315.0fp0.035", "dm315.0fp0.281", "dm325.0fp0.081", "dm325.0fp0.603", "dm335.0fp0.175",
      "dm345.0fp0.051", "dm345.0fp0.412", "dm355.0fp0.108", "dm355.0fp0.883",
    ];
    const LAUNCHERS = {
      serial: {
        workers: 10,
        code: 'wrapper.run_cases(launcher="serial", num_workers=10)',
        note: "10 SWAN processes on this laptop",
      },
      docker: {
        workers: 10,
        code: 'wrapper.run_cases_in_background(launcher="docker_serial", num_workers=10)',
        note: "10 geoocean/rocky8 containers, notebook stays free",
      },
      slurm: {
        workers: NAMES.length,
        code: 'wrapper.run_cases_bulk(launcher="sbatch --array=1-1260 sbatch_example.sh")',
        note: "one Slurm array job, every case on its own core",
      },
    };
    const CODE = [
      "swan_cases  # {'dm': 36 directions, 'fp': 35 frequencies}",
      "wrapper.build_cases(mode=\"all_combinations\", cases_name_format=lambda ctx: f\"dm{ctx['dm']:.1f}fp{ctx['fp']:.3f}\")",
      null, // launcher-dependent
      "ds = wrapper.postprocess_cases()  # -> output_postprocessed.nc",
    ];
    // step = index of the next call to make (0, "templates", is the resting state)
    const BUTTON = {
      1: "build_cases() &#9654;",
      2: "run_cases() &#9654;",
      3: "postprocess_cases() &#9654;",
      4: "Replay &#8634;",
    };
    const BUSY = { 1: "building&hellip;", 2: "running&hellip;", 3: "merging&hellip;" };

    const reduce = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    const $ = (sel) => root.querySelector(sel);
    const grid = $("#wa-grid");
    const btn = $("#wa-next");
    const codeEl = $("#wa-code");
    const countEl = $("#wa-count");
    const caseNumEl = $("#wa-casenum");
    const bar = $("#wa-bar");
    const pct = $("#wa-pct");
    const note = $("#wa-note");
    const nc = $("#wa-nc");
    const ncSize = $("#wa-nc-size");
    const repr = $("#wa-repr");
    const link1 = $("#wa-link1");
    const link2 = $("#wa-link2");
    const chips = root.querySelectorAll(".wa-chip");
    const pills = root.querySelectorAll(".wa-steps li");

    const tiles = NAMES.map((name) => {
      const [dm, fp] = name.replace("dm", "").split("fp");
      const t = document.createElement("div");
      t.className = "wa-tile";
      t.title = name;
      t.innerHTML = `<span>dm${dm}</span><span>fp${fp}</span>`;
      grid.appendChild(t);
      return t;
    });
    const more = document.createElement("div");
    more.className = "wa-tile wa-more";
    more.innerHTML = `<span>+${(TOTAL - NAMES.length).toLocaleString("en")}</span><span>more</span>`;
    grid.appendChild(more);

    let step = 0;
    let busy = false;
    let launcher = "serial";
    let timers = [];
    let typer = null;

    const fmt = (n) => Math.round(n).toLocaleString("en");
    function later(fn, ms) {
      timers.push(setTimeout(fn, reduce ? 0 : ms));
    }

    function typeCode(text) {
      clearInterval(typer);
      if (reduce) {
        codeEl.textContent = text;
        return;
      }
      let i = 0;
      codeEl.textContent = "";
      typer = setInterval(() => {
        i += 2;
        codeEl.textContent = text.slice(0, i);
        if (i >= text.length) clearInterval(typer);
      }, 14);
    }

    function render() {
      pills.forEach((p, i) => {
        p.classList.toggle("done", i < step);
        p.classList.toggle("current", i === step);
      });
      btn.innerHTML = busy ? BUSY[step] : BUTTON[step];
      btn.disabled = busy;
      chips.forEach((c) => (c.disabled = busy || step > 2));
      note.textContent = LAUNCHERS[launcher].note;
    }

    function setProgress(fraction) {
      bar.style.width = `${fraction * 100}%`;
      pct.innerHTML = `monitor_cases()&nbsp;&nbsp;${Math.round(fraction * 100)}%`;
    }

    function finish(nextStep) {
      busy = false;
      step = nextStep;
      render();
    }

    function build() {
      busy = true;
      render();
      typeCode(CODE[1]);
      link1.classList.add("active");
      const stagger = 32;
      const total = tiles.length * stagger;
      [...tiles, more].forEach((t, i) => later(() => t.classList.add("built"), 250 + i * stagger));
      // Jinja rendering the {{ case_num }} placeholder, case after case
      let shown = 0;
      const ticks = 30;
      for (let k = 1; k <= ticks; k++) {
        later(() => {
          shown = Math.round((k / ticks) * (TOTAL - 1));
          caseNumEl.textContent = String(shown);
          countEl.textContent = `${fmt(((k / ticks) * TOTAL))} cases`;
        }, 250 + (k / ticks) * total);
      }
      later(() => {
        link1.classList.remove("active");
        caseNumEl.textContent = "{{ case_num }}";
        countEl.textContent = `${fmt(TOTAL)} cases`;
        finish(2);
        typeCode(LAUNCHERS[launcher].code);
      }, 450 + total);
    }

    function run() {
      busy = true;
      render();
      typeCode(LAUNCHERS[launcher].code);
      root.classList.add("running");
      const workers = LAUNCHERS[launcher].workers;
      const queue = [...tiles];
      let active = 0;
      let done = 0;
      const pump = () => {
        while (active < workers && queue.length) {
          const t = queue.shift();
          active++;
          t.classList.add("running");
          later(() => {
            t.classList.remove("running");
            t.classList.add("done");
            active--;
            done++;
            setProgress(done / tiles.length);
            if (done === tiles.length) {
              more.classList.add("done");
              root.classList.remove("running");
              typeCode(CODE[3]);
              finish(3);
            } else {
              pump();
            }
          }, 350 + Math.random() * 650);
        }
      };
      pump();
    }

    function postprocess() {
      busy = true;
      render();
      link2.classList.add("active");
      const target = nc.querySelector(".wa-doc").getBoundingClientRect();
      const tx = target.left + target.width / 2;
      const ty = target.top + target.height / 2;
      const stagger = 22;
      [...tiles, more].forEach((t, i) => {
        const r = t.getBoundingClientRect();
        const dx = tx - (r.left + r.width / 2);
        const dy = ty - (r.top + r.height / 2);
        t.style.transitionDelay = reduce ? "0ms" : `${i * stagger}ms`;
        t.classList.add("flying");
        t.style.transform = `translate(${dx}px, ${dy}px) scale(0.15)`;
      });
      const total = (tiles.length + 1) * stagger + 650;
      later(() => nc.classList.add("filling"), total * 0.4);
      later(() => {
        link2.classList.remove("active");
        nc.classList.remove("filling");
        nc.classList.add("filled");
        ncSize.textContent = "1 file, 191 MB";
        repr.classList.add("show");
        countEl.textContent = "merged";
        typeCode("ds  # one xarray.Dataset with every case -> BinWaves reconstruction");
        finish(4);
      }, total);
    }

    function reset() {
      timers.forEach(clearTimeout);
      timers = [];
      clearInterval(typer);
      root.classList.add("wa-instant");
      root.classList.remove("running");
      [...tiles, more].forEach((t) => {
        t.className = t === more ? "wa-tile wa-more" : "wa-tile";
        t.style.transform = "";
        t.style.transitionDelay = "";
      });
      link1.classList.remove("active");
      link2.classList.remove("active");
      nc.classList.remove("filled", "filling");
      repr.classList.remove("show");
      ncSize.innerHTML = "waiting&hellip;";
      caseNumEl.textContent = "{{ case_num }}";
      countEl.textContent = "empty";
      setProgress(0);
      void root.offsetWidth; // flush styles before re-enabling transitions
      root.classList.remove("wa-instant");
      busy = false;
      step = 1;
      codeEl.textContent = CODE[0];
      render();
    }

    btn.addEventListener("click", () => {
      if (busy) return;
      if (step === 1) build();
      else if (step === 2) run();
      else if (step === 3) postprocess();
      else reset();
    });
    $("#wa-reset").addEventListener("click", reset);
    chips.forEach((chip) =>
      chip.addEventListener("click", () => {
        if (busy || step > 2) return;
        launcher = chip.getAttribute("data-launcher");
        chips.forEach((c) => c.classList.toggle("active", c === chip));
        if (step === 2) typeCode(LAUNCHERS[launcher].code);
        render();
      })
    );

    reset();
  })();

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
