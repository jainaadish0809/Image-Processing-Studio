"use strict";

/* app.js - the page: file upload, controls, drawing results. All maths lives in cv.js. */
(() => {
  const MAX_BYTES = 5 * 1024 * 1024;
  const MAX_SIDE = 1000;                       // larger photos are scaled down to stay fast
  const ALLOWED = ["png", "jpg", "jpeg", "bmp", "webp"];
  const HIST_COLORS = { blue: "#2563eb", green: "#16a34a", red: "#dc2626", gray: "#334155" };

  const state = {
    selected: null,    // operation chosen in the list
    original: null,    // image loaded from the file
    source: null,      // image the next operation will run on
    result: null,      // image produced by the last run
    pendingStep: null, // label of the last run, added to the steps list on "Use result as input"
    pipeline: [],
    busy: false,
  };

  const $ = (id) => document.getElementById(id);
  const el = {
    file: $("file"), fileLabel: $("file-label"), opList: $("op-list"),
    description: $("op-description"), params: $("params"), apply: $("apply"),
    status: $("status"), imgInput: $("img-input"), imgResult: $("img-result"),
    histInput: $("hist-input"), histResult: $("hist-result"),
    commit: $("commit"), download: $("download"), reset: $("reset"),
    pipeline: $("pipeline"), stats: $("stats"), log: $("log"),
  };

  /** Build a DOM node without innerHTML. */
  function h(tag, attrs = {}, ...children) {
    const node = document.createElement(tag);
    for (const [name, value] of Object.entries(attrs)) node.setAttribute(name, value);
    node.append(...children);
    return node;
  }

  function setStatus(message, isError = false) {
    el.status.textContent = message;
    el.status.classList.toggle("error", isError);
  }

  function addLog(message) {
    if (el.log.firstChild && el.log.firstChild.classList.contains("muted")) el.log.replaceChildren();
    const time = h("time", {}, new Date().toLocaleTimeString());
    el.log.prepend(h("li", {}, time, message));
    while (el.log.children.length > 8) el.log.lastChild.remove();
    console.log(message);
  }

  function refreshButtons() {
    el.apply.disabled = state.busy || !state.source || !state.selected;
    el.commit.disabled = state.busy || !state.result;
    el.reset.disabled = state.busy || !state.original;
    el.download.setAttribute("aria-disabled", state.result ? "false" : "true");
  }

  // ---------- operation list and settings ----------
  function renderOperationList() {
    const groups = new Map();
    for (const op of CV.operations) {
      if (!groups.has(op.module)) groups.set(op.module, []);
      groups.get(op.module).push(op);
    }
    for (const [module, ops] of groups) {
      const box = h("div", { class: "op-group" }, h("h3", {}, CV.moduleLabels[module]));
      for (const op of ops) {
        const radio = h("input", { type: "radio", name: "operation", value: op.key });
        radio.addEventListener("change", () => selectOperation(op));
        box.append(h("label", {}, radio, h("span", {}, op.label)));
      }
      el.opList.append(box);
    }
  }

  function selectOperation(op) {
    state.selected = op;
    el.description.textContent = op.description;
    el.params.replaceChildren(...op.params.map(paramControl));
    refreshButtons();
  }

  function paramControl(p) {
    const id = "param-" + p.name;
    const field = h("div", { class: "field" });
    let input;
    if (p.kind === "choice") {
      input = h("select", { id });
      p.choices.forEach((choice) => input.append(h("option", { value: choice }, choice)));
      input.value = p.default;
      field.append(h("div", { class: "row" }, h("label", { for: id }, p.label)));
    } else {
      input = h("input", { id, type: "range", min: p.min, max: p.max, step: p.step, value: p.default });
      const output = h("output", { for: id }, String(p.default));
      input.addEventListener("input", () => { output.textContent = input.value; });
      field.append(h("div", { class: "row" }, h("label", { for: id }, p.label), output));
    }
    input.dataset.name = p.name;
    field.append(input);
    return field;
  }

  function readParams() {
    const values = {};
    el.params.querySelectorAll("[data-name]").forEach((input) => { values[input.dataset.name] = input.value; });
    return values;
  }

  // ---------- drawing ----------
  function paint(canvas, img) {
    canvas.width = img.width;
    canvas.height = img.height;
    canvas.getContext("2d").putImageData(new ImageData(img.data, img.width, img.height), 0, 0);
    canvas.hidden = false;
  }

  function drawHistogram(canvas, histogram) {
    const ctx = canvas.getContext("2d");
    const { width, height } = canvas;
    ctx.clearRect(0, 0, width, height);
    const peak = Math.max(1, ...Object.values(histogram).flat());
    ctx.lineWidth = 1.5;
    ctx.globalAlpha = 0.85;
    for (const [channel, bins] of Object.entries(histogram)) {
      ctx.strokeStyle = HIST_COLORS[channel];
      ctx.beginPath();
      bins.forEach((count, i) => {
        const x = (i / 255) * (width - 2) + 1;
        const y = height - 2 - (count / peak) * (height - 6);
        if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
      });
      ctx.stroke();
    }
    ctx.globalAlpha = 1;
  }

  function renderStats(stats, info) {
    const rows = [
      ["Size", `${stats.width} x ${stats.height} px`],
      ["Channels", stats.channels === 1 ? "1 (gray)" : "3 (colour)"],
      ["Average brightness", stats.mean_brightness],
      ["Contrast (std. dev.)", stats.contrast],
      ["Edge pixels", `${stats.edge_density_percent}%`],
      ["Similarity to input (PSNR)", stats.psnr_db === null ? "Not comparable" : `${stats.psnr_db} dB`],
    ];
    for (const [key, value] of Object.entries(info)) {
      rows.push([key.replace(/_/g, " ").replace(/^./, (c) => c.toUpperCase()), value]);
    }
    el.stats.replaceChildren(...rows.flatMap(([name, value]) => [h("dt", {}, name), h("dd", {}, String(value))]));
  }

  function renderPipeline() {
    const steps = state.pipeline.length
      ? state.pipeline.map((label) => h("li", {}, label))
      : [h("li", { class: "muted" }, "None yet")];
    el.pipeline.replaceChildren(...steps);
  }

  function clearResult() {
    state.result = null;
    state.pendingStep = null;
    el.imgResult.hidden = true;
    el.download.removeAttribute("href");
    el.stats.replaceChildren();
    for (const canvas of [el.histInput, el.histResult]) canvas.getContext("2d").clearRect(0, 0, canvas.width, canvas.height);
  }

  function resetToOriginal() {
    state.source = state.original;
    state.pipeline = [];
    clearResult();
    renderPipeline();
    paint(el.imgInput, state.source);
    refreshButtons();
  }

  // ---------- loading an image ----------
  function loadFile(file) {
    const extension = file.name.split(".").pop().toLowerCase();
    if (!ALLOWED.includes(extension)) return fail("That file type is not supported. Use PNG, JPG, BMP or WebP.");
    if (file.size > MAX_BYTES) return fail("That file is larger than 5 MB. Choose a smaller image.");

    const url = URL.createObjectURL(file);
    const image = new Image();
    image.onload = () => {
      URL.revokeObjectURL(url);
      const scale = Math.min(1, MAX_SIDE / Math.max(image.naturalWidth, image.naturalHeight));
      const w = Math.max(1, Math.round(image.naturalWidth * scale));
      const h = Math.max(1, Math.round(image.naturalHeight * scale));
      const canvas = document.createElement("canvas");
      canvas.width = w;
      canvas.height = h;
      const ctx = canvas.getContext("2d", { willReadFrequently: true });
      ctx.fillStyle = "#ffffff";                // transparent areas become white
      ctx.fillRect(0, 0, w, h);
      ctx.drawImage(image, 0, 0, w, h);
      const pixels = ctx.getImageData(0, 0, w, h);
      state.original = { width: w, height: h, data: pixels.data };
      el.fileLabel.textContent = file.name;
      resetToOriginal();
      setStatus(`Image loaded (${w} x ${h} px). Choose an operation and apply it.`);
      addLog(`Loaded ${file.name} (${w} x ${h})`);
    };
    image.onerror = () => {
      URL.revokeObjectURL(url);
      fail("The file could not be read as an image.");
    };
    image.src = url;
  }

  function fail(message) {
    el.file.value = "";
    setStatus(message, true);
    addLog("Rejected: " + message);
  }

  el.file.addEventListener("change", () => {
    if (el.file.files[0]) loadFile(el.file.files[0]);
  });

  // ---------- applying an operation ----------
  el.apply.addEventListener("click", () => {
    if (state.busy || !state.source || !state.selected) return;
    state.busy = true;
    refreshButtons();
    setStatus("Processing...");
    // Let the browser show "Processing..." before the (synchronous) work starts.
    setTimeout(() => {
      try {
        const started = performance.now();
        const { image, info } = CV.runOperation(state.selected.key, state.source, readParams());
        const stats = CV.imageStats(state.source, image);
        const ms = Math.round(performance.now() - started);

        state.result = image;
        state.pendingStep = state.selected.label;
        paint(el.imgResult, image);
        el.download.href = el.imgResult.toDataURL("image/png");
        drawHistogram(el.histInput, CV.histogram(state.source));
        drawHistogram(el.histResult, CV.histogram(image));
        renderStats(stats, info);
        setStatus(`${state.selected.label} finished in ${ms} ms.`);
        addLog(`${state.selected.label} on ${state.source.width}x${state.source.height}: ${ms} ms`);
      } catch (error) {
        if (error instanceof CV.ValidationError) {
          setStatus(error.message, true);
          addLog("Rejected: " + error.message);
        } else {
          console.error(error);
          setStatus("Something went wrong while processing the image.", true);
          addLog("Error: " + error.message);
        }
      } finally {
        state.busy = false;
        refreshButtons();
      }
    }, 30);
  });

  // ---------- chaining and reset ----------
  el.commit.addEventListener("click", () => {
    if (!state.result) return;
    state.pipeline.push(state.pendingStep);
    state.source = state.result;
    paint(el.imgInput, state.source);
    clearResult();
    renderPipeline();
    refreshButtons();
    setStatus("The result is now the input. Choose the next operation.");
  });

  el.reset.addEventListener("click", () => {
    resetToOriginal();
    setStatus("Back to the original image.");
  });

  renderOperationList();
  refreshButtons();
})();
