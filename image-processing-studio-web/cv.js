"use strict";

/*
 * cv.js - image-processing algorithms written in plain JavaScript.
 *
 * Images are plain objects: { width, height, data } where `data` is an RGBA
 * Uint8ClampedArray (the same layout as a canvas ImageData). Nothing here touches
 * the page, so the file can be tested with Node and reused anywhere.
 *
 * Sections:  1. helpers   2. preprocessing   3. feature extraction
 *            4. analysis  5. operation registry (used to build the web controls)
 */
const CV = (() => {
  class ValidationError extends Error {}

  // ======================================================================
  // 1. Helpers
  // ======================================================================
  const clamp = (v, lo, hi) => (v < lo ? lo : v > hi ? hi : v);
  const makeImage = (width, height) => ({ width, height, data: new Uint8ClampedArray(width * height * 4) });
  const cloneImage = (img) => ({ width: img.width, height: img.height, data: new Uint8ClampedArray(img.data) });
  const makeOdd = (n) => (n % 2 === 1 ? n : n + 1);

  /** Luminance (0-255) of every pixel as a flat array. */
  function toGray(img) {
    const n = img.width * img.height;
    const gray = new Uint8ClampedArray(n);
    const d = img.data;
    for (let i = 0, j = 0; i < n; i++, j += 4) gray[i] = 0.299 * d[j] + 0.587 * d[j + 1] + 0.114 * d[j + 2];
    return gray;
  }

  function grayToImage(gray, width, height) {
    const out = makeImage(width, height);
    const d = out.data;
    for (let i = 0, j = 0; i < gray.length; i++, j += 4) {
      d[j] = d[j + 1] = d[j + 2] = gray[i];
      d[j + 3] = 255;
    }
    return out;
  }

  /** True when every pixel has R = G = B. */
  function isGray(img) {
    const d = img.data;
    for (let j = 0; j < d.length; j += 4) if (d[j] !== d[j + 1] || d[j] !== d[j + 2]) return false;
    return true;
  }

  function splitPlanes(img) {
    const n = img.width * img.height;
    const r = new Float32Array(n), g = new Float32Array(n), b = new Float32Array(n);
    const d = img.data;
    for (let i = 0, j = 0; i < n; i++, j += 4) {
      r[i] = d[j]; g[i] = d[j + 1]; b[i] = d[j + 2];
    }
    return [r, g, b];
  }

  function joinPlanes(r, g, b, width, height) {
    const out = makeImage(width, height);
    const d = out.data;
    for (let i = 0, j = 0; i < r.length; i++, j += 4) {
      d[j] = r[i]; d[j + 1] = g[i]; d[j + 2] = b[i]; d[j + 3] = 255; // typed array rounds and clamps
    }
    return out;
  }

  /** Apply a function that maps one plane to one plane on R, G and B separately. */
  function mapPlanes(img, fn) {
    const [r, g, b] = splitPlanes(img);
    return joinPlanes(fn(r), fn(g), fn(b), img.width, img.height);
  }

  function gaussianKernel(sigma, size) {
    const radius = (size - 1) / 2;
    const kernel = new Float32Array(size);
    let sum = 0;
    for (let i = 0; i < size; i++) {
      const x = i - radius;
      kernel[i] = Math.exp(-(x * x) / (2 * sigma * sigma));
      sum += kernel[i];
    }
    return kernel.map((v) => v / sum);
  }

  /** Convolve a plane with a 1-D kernel along x, then along y (edge pixels are repeated). */
  function convolveSeparable(plane, w, h, kernel) {
    const radius = (kernel.length - 1) / 2;
    const tmp = new Float32Array(w * h);
    const out = new Float32Array(w * h);
    for (let y = 0; y < h; y++) {
      for (let x = 0; x < w; x++) {
        let sum = 0;
        for (let k = -radius; k <= radius; k++) sum += kernel[k + radius] * plane[y * w + clamp(x + k, 0, w - 1)];
        tmp[y * w + x] = sum;
      }
    }
    for (let y = 0; y < h; y++) {
      for (let x = 0; x < w; x++) {
        let sum = 0;
        for (let k = -radius; k <= radius; k++) sum += kernel[k + radius] * tmp[clamp(y + k, 0, h - 1) * w + x];
        out[y * w + x] = sum;
      }
    }
    return out;
  }

  /** Gaussian blur of one plane. Sigma follows OpenCV's rule when not given. */
  function blurPlane(plane, w, h, ksize, sigma) {
    if (ksize <= 1) return Float32Array.from(plane);
    const s = sigma || 0.3 * ((ksize - 1) * 0.5 - 1) + 0.8;
    return convolveSeparable(plane, w, h, gaussianKernel(s, ksize));
  }

  /** 3x3 Sobel gradients of a plane. */
  function sobelPlanes(plane, w, h) {
    const gx = new Float32Array(w * h), gy = new Float32Array(w * h);
    for (let y = 0; y < h; y++) {
      const ym = clamp(y - 1, 0, h - 1) * w, y0 = y * w, yp = clamp(y + 1, 0, h - 1) * w;
      for (let x = 0; x < w; x++) {
        const xm = clamp(x - 1, 0, w - 1), xp = clamp(x + 1, 0, w - 1);
        const a = plane[ym + xm], b = plane[ym + x], c = plane[ym + xp];
        const d = plane[y0 + xm], f = plane[y0 + xp];
        const g = plane[yp + xm], hh = plane[yp + x], i = plane[yp + xp];
        gx[y0 + x] = (c + 2 * f + i) - (a + 2 * d + g);
        gy[y0 + x] = (g + 2 * hh + i) - (a + 2 * b + c);
      }
    }
    return { gx, gy };
  }

  function otsuThreshold(gray) {
    const hist = new Float64Array(256);
    for (let i = 0; i < gray.length; i++) hist[gray[i]]++;
    const total = gray.length;
    let sumAll = 0;
    for (let t = 0; t < 256; t++) sumAll += t * hist[t];
    let sumB = 0, wB = 0, best = -1, threshold = 0;
    for (let t = 0; t < 256; t++) {
      wB += hist[t];
      if (wB === 0) continue;
      const wF = total - wB;
      if (wF === 0) break;
      sumB += t * hist[t];
      const diff = sumB / wB - (sumAll - sumB) / wF;
      const between = wB * wF * diff * diff;
      if (between > best) { best = between; threshold = t; }
    }
    return threshold;
  }

  // ======================================================================
  // 2. Preprocessing
  // ======================================================================
  const grayscale = (img) => grayToImage(toGray(img), img.width, img.height);

  function resize(img, { scale_percent }) {
    const w = img.width, h = img.height;
    const nw = Math.max(1, Math.floor((w * scale_percent) / 100));
    const nh = Math.max(1, Math.floor((h * scale_percent) / 100));
    if (Math.max(nw, nh) > 3200) throw new ValidationError("The result would be larger than 3200 px. Choose a smaller scale.");
    const out = makeImage(nw, nh);
    const src = img.data, dst = out.data;
    for (let y = 0; y < nh; y++) {
      const sy = clamp(((y + 0.5) * h) / nh - 0.5, 0, h - 1);
      const y0 = Math.floor(sy), y1 = Math.min(y0 + 1, h - 1), fy = sy - y0;
      for (let x = 0; x < nw; x++) {
        const sx = clamp(((x + 0.5) * w) / nw - 0.5, 0, w - 1);
        const x0 = Math.floor(sx), x1 = Math.min(x0 + 1, w - 1), fx = sx - x0;
        const o = (y * nw + x) * 4;
        for (let c = 0; c < 3; c++) {
          const top = src[(y0 * w + x0) * 4 + c] * (1 - fx) + src[(y0 * w + x1) * 4 + c] * fx;
          const bottom = src[(y1 * w + x0) * 4 + c] * (1 - fx) + src[(y1 * w + x1) * 4 + c] * fx;
          dst[o + c] = top * (1 - fy) + bottom * fy;
        }
        dst[o + 3] = 255;
      }
    }
    return out;
  }

  const gaussianBlur = (img, { ksize }) =>
    mapPlanes(img, (plane) => blurPlane(plane, img.width, img.height, makeOdd(ksize)));

  /** Median filter using a sliding histogram (fast even for large kernels). */
  function medianPlane(plane, w, h, k) {
    const r = (k - 1) / 2;
    const half = (k * k + 1) >> 1;
    const out = new Uint8ClampedArray(w * h);
    const hist = new Int32Array(256);
    for (let y = 0; y < h; y++) {
      hist.fill(0);
      for (let dy = -r; dy <= r; dy++) {
        const row = clamp(y + dy, 0, h - 1) * w;
        for (let dx = -r; dx <= r; dx++) hist[plane[row + clamp(dx, 0, w - 1)]]++;
      }
      for (let x = 0; x < w; x++) {
        if (x > 0) {
          const xo = clamp(x - 1 - r, 0, w - 1), xn = clamp(x + r, 0, w - 1);
          for (let dy = -r; dy <= r; dy++) {
            const row = clamp(y + dy, 0, h - 1) * w;
            hist[plane[row + xo]]--;
            hist[plane[row + xn]]++;
          }
        }
        let acc = 0, m = 0;
        while ((acc += hist[m]) < half) m++;
        out[y * w + x] = m;
      }
    }
    return out;
  }

  const medianBlur = (img, { ksize }) =>
    mapPlanes(img, (plane) => medianPlane(plane, img.width, img.height, makeOdd(ksize)));

  /** Equalise the luminance only, so colours are not distorted. */
  function equalizeHistogram(img) {
    const n = img.width * img.height;
    const d = img.data;
    const luma = new Float32Array(n), cb = new Float32Array(n), cr = new Float32Array(n);
    const hist = new Int32Array(256);
    for (let i = 0, j = 0; i < n; i++, j += 4) {
      const y = 0.299 * d[j] + 0.587 * d[j + 1] + 0.114 * d[j + 2];
      luma[i] = y; cr[i] = d[j] - y; cb[i] = d[j + 2] - y;
      hist[Math.round(y)]++;
    }
    const lut = new Float32Array(256);
    let cdf = 0, cdfMin = 0;
    for (let v = 0; v < 256; v++) if (hist[v] > 0) { cdfMin = hist[v]; break; }
    for (let v = 0; v < 256; v++) {
      cdf += hist[v];
      lut[v] = n === cdfMin ? v : ((cdf - cdfMin) / (n - cdfMin)) * 255;
    }
    const out = makeImage(img.width, img.height);
    const o = out.data;
    for (let i = 0, j = 0; i < n; i++, j += 4) {
      const y = lut[Math.round(luma[i])];
      const r = y + cr[i], b = y + cb[i];
      o[j] = r; o[j + 2] = b;
      o[j + 1] = (y - 0.299 * clamp(r, 0, 255) - 0.114 * clamp(b, 0, 255)) / 0.587;
      o[j + 3] = 255;
    }
    return out;
  }

  function brightnessContrast(img, { brightness, contrast }) {
    const out = cloneImage(img);
    const d = out.data;
    for (let j = 0; j < d.length; j += 4) {
      d[j] = contrast * d[j] + brightness;
      d[j + 1] = contrast * d[j + 1] + brightness;
      d[j + 2] = contrast * d[j + 2] + brightness;
    }
    return out;
  }

  function threshold(img, { mode, value }) {
    const { width: w, height: h } = img;
    const gray = toGray(img);
    const out = new Uint8ClampedArray(w * h);
    if (mode === "adaptive") {
      // Compare each pixel with the mean of its 11x11 neighbourhood minus 2.
      const stride = w + 1;
      const integral = new Float64Array(stride * (h + 1));
      for (let y = 0; y < h; y++) {
        let rowSum = 0;
        for (let x = 0; x < w; x++) {
          rowSum += gray[y * w + x];
          integral[(y + 1) * stride + x + 1] = integral[y * stride + x + 1] + rowSum;
        }
      }
      for (let y = 0; y < h; y++) {
        const y0 = Math.max(0, y - 5), y1 = Math.min(h, y + 6);
        for (let x = 0; x < w; x++) {
          const x0 = Math.max(0, x - 5), x1 = Math.min(w, x + 6);
          const sum = integral[y1 * stride + x1] - integral[y0 * stride + x1] - integral[y1 * stride + x0] + integral[y0 * stride + x0];
          const mean = sum / ((x1 - x0) * (y1 - y0));
          out[y * w + x] = gray[y * w + x] > mean - 2 ? 255 : 0;
        }
      }
    } else {
      const level = mode === "otsu" ? otsuThreshold(gray) : value;
      for (let i = 0; i < out.length; i++) out[i] = gray[i] > level ? 255 : 0;
    }
    return grayToImage(out, w, h);
  }

  /** Unsharp mask: image + amount * (image - blurred image). */
  function sharpen(img, { amount }) {
    const { width: w, height: h } = img;
    const kernel = gaussianKernel(3, 19);
    return mapPlanes(img, (plane) => {
      const blurred = convolveSeparable(plane, w, h, kernel);
      const out = new Float32Array(plane.length);
      for (let i = 0; i < out.length; i++) out[i] = (1 + amount) * plane[i] - amount * blurred[i];
      return out;
    });
  }

  function negative(img) {
    const out = cloneImage(img);
    const d = out.data;
    for (let j = 0; j < d.length; j += 4) { d[j] = 255 - d[j]; d[j + 1] = 255 - d[j + 1]; d[j + 2] = 255 - d[j + 2]; }
    return out;
  }

  /** Square-kernel min (erode) or max (dilate), done as two 1-D passes. */
  function rankFilter(plane, w, h, k, useMax) {
    const r = (k - 1) / 2;
    const pick = useMax ? Math.max : Math.min;
    const tmp = new Float32Array(w * h), out = new Float32Array(w * h);
    for (let y = 0; y < h; y++) {
      for (let x = 0; x < w; x++) {
        let v = plane[y * w + x];
        for (let dx = -r; dx <= r; dx++) v = pick(v, plane[y * w + clamp(x + dx, 0, w - 1)]);
        tmp[y * w + x] = v;
      }
    }
    for (let y = 0; y < h; y++) {
      for (let x = 0; x < w; x++) {
        let v = tmp[y * w + x];
        for (let dy = -r; dy <= r; dy++) v = pick(v, tmp[clamp(y + dy, 0, h - 1) * w + x]);
        out[y * w + x] = v;
      }
    }
    return out;
  }

  function morphology(img, { operation, ksize }) {
    const { width: w, height: h } = img;
    const k = makeOdd(ksize);
    const erode = (p) => rankFilter(p, w, h, k, false);
    const dilate = (p) => rankFilter(p, w, h, k, true);
    const steps = {
      erode, dilate,
      open: (p) => dilate(erode(p)),
      close: (p) => erode(dilate(p)),
    };
    return mapPlanes(img, steps[operation]);
  }

  // ======================================================================
  // 3. Feature extraction
  // ======================================================================
  /** Canny: blur -> Sobel gradient -> non-maximum suppression -> hysteresis. Returns 0/255 per pixel. */
  function cannyPlane(gray, w, h, low, high) {
    const blurred = blurPlane(gray, w, h, 5);
    const { gx, gy } = sobelPlanes(blurred, w, h);
    const mag = new Float32Array(w * h);
    for (let i = 0; i < mag.length; i++) mag[i] = Math.abs(gx[i]) + Math.abs(gy[i]);

    const thin = new Float32Array(w * h);
    for (let y = 1; y < h - 1; y++) {
      for (let x = 1; x < w - 1; x++) {
        const i = y * w + x, m = mag[i];
        if (m < low) continue;
        let deg = (Math.atan2(gy[i], gx[i]) * 180) / Math.PI;
        if (deg < 0) deg += 180;
        let a, b;
        if (deg < 22.5 || deg >= 157.5) { a = mag[i - 1]; b = mag[i + 1]; }
        else if (deg < 67.5) { a = mag[i - w - 1]; b = mag[i + w + 1]; }
        else if (deg < 112.5) { a = mag[i - w]; b = mag[i + w]; }
        else { a = mag[i - w + 1]; b = mag[i + w - 1]; }
        if (m >= a && m >= b) thin[i] = m;
      }
    }

    const edges = new Uint8ClampedArray(w * h);
    const stack = new Int32Array(w * h);
    let sp = 0;
    for (let i = 0; i < thin.length; i++) {
      if (thin[i] > 0 && thin[i] >= high) { edges[i] = 255; stack[sp++] = i; }
    }
    while (sp > 0) {
      const p = stack[--sp];
      const px = p % w, py = (p - px) / w;
      for (let dy = -1; dy <= 1; dy++) {
        for (let dx = -1; dx <= 1; dx++) {
          const nx = px + dx, ny = py + dy;
          if (nx < 0 || ny < 0 || nx >= w || ny >= h) continue;
          const q = ny * w + nx;
          if (!edges[q] && thin[q] > 0 && thin[q] >= low) { edges[q] = 255; stack[sp++] = q; }
        }
      }
    }
    return edges;
  }

  function canny(img, { low, high }) {
    if (low >= high) throw new ValidationError("The low threshold must be smaller than the high threshold.");
    return grayToImage(cannyPlane(toGray(img), img.width, img.height, low, high), img.width, img.height);
  }

  function sobel(img) {
    const { width: w, height: h } = img;
    const { gx, gy } = sobelPlanes(Float32Array.from(toGray(img)), w, h);
    const mag = new Float32Array(w * h);
    let peak = 0;
    for (let i = 0; i < mag.length; i++) { mag[i] = Math.hypot(gx[i], gy[i]); if (mag[i] > peak) peak = mag[i]; }
    const out = new Uint8ClampedArray(w * h);
    if (peak > 0) for (let i = 0; i < out.length; i++) out[i] = (mag[i] / peak) * 255;
    return grayToImage(out, w, h);
  }

  function laplacian(img) {
    const { width: w, height: h } = img;
    const gray = blurPlane(Float32Array.from(toGray(img)), w, h, 3);
    const out = new Uint8ClampedArray(w * h);
    for (let y = 0; y < h; y++) {
      const ym = clamp(y - 1, 0, h - 1) * w, y0 = y * w, yp = clamp(y + 1, 0, h - 1) * w;
      for (let x = 0; x < w; x++) {
        const xm = clamp(x - 1, 0, w - 1), xp = clamp(x + 1, 0, w - 1);
        out[y0 + x] = Math.abs(gray[ym + x] + gray[yp + x] + gray[y0 + xm] + gray[y0 + xp] - 4 * gray[y0 + x]);
      }
    }
    return grayToImage(out, w, h);
  }

  /**
   * Outline shapes: Otsu threshold -> connected-component labelling -> draw the border
   * of every component whose area is at least `min_area`. The smaller of the two
   * classes (dark / bright) is treated as the shapes, so a plain background is ignored.
   */
  function contours(img, { min_area }) {
    const { width: w, height: h } = img;
    const gray = toGray(img);
    const cut = otsuThreshold(gray);
    const fg = new Uint8Array(w * h);
    let bright = 0;
    for (let i = 0; i < fg.length; i++) if (gray[i] > cut) { fg[i] = 1; bright++; }
    if (bright > fg.length - bright) for (let i = 0; i < fg.length; i++) fg[i] ^= 1;

    const labels = new Int32Array(w * h);
    const stack = new Int32Array(w * h);
    const areas = [0];
    for (let start = 0; start < fg.length; start++) {
      if (!fg[start] || labels[start]) continue;
      const id = areas.length;
      let sp = 0, area = 0;
      stack[sp++] = start; labels[start] = id;
      while (sp > 0) {
        const p = stack[--sp];
        area++;
        const x = p % w;
        if (x > 0 && fg[p - 1] && !labels[p - 1]) { labels[p - 1] = id; stack[sp++] = p - 1; }
        if (x < w - 1 && fg[p + 1] && !labels[p + 1]) { labels[p + 1] = id; stack[sp++] = p + 1; }
        if (p >= w && fg[p - w] && !labels[p - w]) { labels[p - w] = id; stack[sp++] = p - w; }
        if (p < fg.length - w && fg[p + w] && !labels[p + w]) { labels[p + w] = id; stack[sp++] = p + w; }
      }
      areas.push(area);
    }

    const out = cloneImage(img);
    const d = out.data;
    for (let y = 0; y < h; y++) {
      for (let x = 0; x < w; x++) {
        const p = y * w + x, id = labels[p];
        if (!id || areas[id] < min_area) continue;
        const border = x === 0 || y === 0 || x === w - 1 || y === h - 1 ||
          labels[p - 1] !== id || labels[p + 1] !== id || labels[p - w] !== id || labels[p + w] !== id;
        if (border) { d[p * 4] = 0; d[p * 4 + 1] = 200; d[p * 4 + 2] = 0; }
      }
    }
    const found = areas.slice(1).filter((a) => a >= min_area).length;
    return { image: out, info: { contours_found: found } };
  }

  function drawCircle(img, cx, cy, radius, rgb) {
    const { width: w, height: h, data } = img;
    for (let step = 0; step < 48; step++) {
      const angle = (step / 48) * Math.PI * 2;
      for (const r of [radius, radius + 1]) {
        const x = Math.round(cx + Math.cos(angle) * r), y = Math.round(cy + Math.sin(angle) * r);
        if (x < 0 || y < 0 || x >= w || y >= h) continue;
        const o = (y * w + x) * 4;
        data[o] = rgb[0]; data[o + 1] = rgb[1]; data[o + 2] = rgb[2];
      }
    }
  }

  /** Harris corners: structure tensor from Sobel gradients, response R = det - 0.04 * trace^2. */
  function harris(img, { sensitivity }) {
    const { width: w, height: h } = img;
    const { gx, gy } = sobelPlanes(Float32Array.from(toGray(img)), w, h);
    const xx = new Float32Array(w * h), yy = new Float32Array(w * h), xy = new Float32Array(w * h);
    for (let i = 0; i < xx.length; i++) { xx[i] = gx[i] * gx[i]; yy[i] = gy[i] * gy[i]; xy[i] = gx[i] * gy[i]; }
    const box = new Float32Array([1, 1, 1]);
    const sxx = convolveSeparable(xx, w, h, box), syy = convolveSeparable(yy, w, h, box), sxy = convolveSeparable(xy, w, h, box);
    const response = new Float64Array(w * h);
    let peak = 0;
    for (let i = 0; i < response.length; i++) {
      const det = sxx[i] * syy[i] - sxy[i] * sxy[i];
      const trace = sxx[i] + syy[i];
      response[i] = det - 0.04 * trace * trace;
      if (response[i] > peak) peak = response[i];
    }
    const out = cloneImage(img);
    let found = 0;
    if (peak > 0) {
      const cut = sensitivity * peak, r = 2;
      for (let y = r; y < h - r; y++) {
        for (let x = r; x < w - r; x++) {
          const value = response[y * w + x];
          if (value <= cut) continue;
          let isPeak = true;
          for (let dy = -r; dy <= r && isPeak; dy++) {
            for (let dx = -r; dx <= r; dx++) {
              if (dx === 0 && dy === 0) continue;
              const other = response[(y + dy) * w + x + dx];
              const earlier = dy < 0 || (dy === 0 && dx < 0);   // tie-break so flat peaks count once
              if (earlier ? other >= value : other > value) { isPeak = false; break; }
            }
          }
          if (isPeak) { found++; drawCircle(out, x, y, 4, [255, 0, 0]); }
        }
      }
    }
    return { image: out, info: { corners_found: found } };
  }

  // ======================================================================
  // 4. Analysis
  // ======================================================================
  function histogram(img) {
    const gray = isGray(img);
    const hist = gray ? { gray: new Array(256).fill(0) } : { red: new Array(256).fill(0), green: new Array(256).fill(0), blue: new Array(256).fill(0) };
    const d = img.data;
    for (let j = 0; j < d.length; j += 4) {
      if (gray) hist.gray[d[j]]++;
      else { hist.red[d[j]]++; hist.green[d[j + 1]]++; hist.blue[d[j + 2]]++; }
    }
    return hist;
  }

  /** Measurements of `processed`, plus PSNR against `original` when the sizes match. */
  function imageStats(original, processed) {
    const { width: w, height: h } = processed;
    const gray = toGray(processed);
    let sum = 0;
    for (let i = 0; i < gray.length; i++) sum += gray[i];
    const mean = sum / gray.length;
    let variance = 0;
    for (let i = 0; i < gray.length; i++) variance += (gray[i] - mean) ** 2;

    const edges = cannyPlane(gray, w, h, 100, 200);
    let edgePixels = 0;
    for (let i = 0; i < edges.length; i++) if (edges[i]) edgePixels++;

    let psnr = null;
    if (original.width === w && original.height === h) {
      const a = original.data, b = processed.data;
      const flat = isGray(processed);
      let mse = 0;
      for (let j = 0; j < a.length; j += 4) {
        const ref = flat ? [0.299 * a[j] + 0.587 * a[j + 1] + 0.114 * a[j + 2]] : [a[j], a[j + 1], a[j + 2]];
        if (flat) mse += (Math.round(ref[0]) - b[j]) ** 2;
        else for (let c = 0; c < 3; c++) mse += (ref[c] - b[j + c]) ** 2;
      }
      mse /= w * h * (flat ? 1 : 3);
      if (mse > 0) psnr = Math.round(10 * Math.log10((255 * 255) / mse) * 100) / 100;
    }
    const round2 = (v) => Math.round(v * 100) / 100;
    return {
      width: w, height: h, channels: isGray(processed) ? 1 : 3,
      mean_brightness: round2(mean),
      contrast: round2(Math.sqrt(variance / gray.length)),
      edge_density_percent: round2((edgePixels / edges.length) * 100),
      psnr_db: psnr,
    };
  }

  // ======================================================================
  // 5. Operation registry - the page builds its controls from this list
  // ======================================================================
  const P = (name, label, kind, def, min, max, step, choices) => ({ name, label, kind, default: def, min, max, step: step || 1, choices: choices || [] });

  const operations = [
    { key: "grayscale", label: "Grayscale", module: "preprocessing", description: "Remove colour and keep only brightness.", params: [], run: grayscale },
    { key: "resize", label: "Resize", module: "preprocessing", description: "Make the image smaller or larger.", params: [P("scale_percent", "Scale (%)", "int", 50, 10, 200, 5)], run: resize },
    { key: "gaussian_blur", label: "Gaussian blur", module: "preprocessing", description: "Smooth the image to reduce fine noise.", params: [P("ksize", "Kernel size", "int", 5, 1, 31, 2)], run: gaussianBlur },
    { key: "median_blur", label: "Median blur", module: "preprocessing", description: "Remove salt-and-pepper noise.", params: [P("ksize", "Kernel size", "int", 5, 3, 15, 2)], run: medianBlur },
    { key: "equalize_histogram", label: "Histogram equalization", module: "preprocessing", description: "Improve contrast by spreading out intensities.", params: [], run: equalizeHistogram },
    { key: "brightness_contrast", label: "Brightness and contrast", module: "preprocessing", description: "Make the image lighter, darker, flatter or punchier.", params: [P("brightness", "Brightness", "int", 0, -100, 100, 5), P("contrast", "Contrast", "float", 1.0, 0.5, 3.0, 0.1)], run: brightnessContrast },
    { key: "threshold", label: "Threshold", module: "preprocessing", description: "Convert to pure black and white.", params: [P("mode", "Method", "choice", "otsu", null, null, 1, ["binary", "otsu", "adaptive"]), P("value", "Level (binary only)", "int", 127, 0, 255, 1)], run: threshold },
    { key: "sharpen", label: "Sharpen", module: "preprocessing", description: "Make edges and details crisper.", params: [P("amount", "Amount", "float", 1.0, 0.1, 3.0, 0.1)], run: sharpen },
    { key: "negative", label: "Negative", module: "preprocessing", description: "Invert all pixel values.", params: [], run: negative },
    { key: "morphology", label: "Morphology", module: "preprocessing", description: "Grow, shrink, open or close shapes.", params: [P("operation", "Operation", "choice", "dilate", null, null, 1, ["erode", "dilate", "open", "close"]), P("ksize", "Kernel size", "int", 3, 1, 15, 2)], run: morphology },
    { key: "canny", label: "Canny edges", module: "features", description: "Detect thin, clean edges.", params: [P("low", "Low threshold", "int", 50, 0, 255, 1), P("high", "High threshold", "int", 150, 0, 255, 1)], run: canny },
    { key: "sobel", label: "Sobel edges", module: "features", description: "Show how quickly brightness changes.", params: [], run: sobel },
    { key: "laplacian", label: "Laplacian edges", module: "features", description: "Highlight regions of rapid intensity change.", params: [], run: laplacian },
    { key: "contours", label: "Contours", module: "features", description: "Outline shapes found by thresholding and count them.", params: [P("min_area", "Minimum area (px)", "int", 100, 0, 10000, 50)], run: contours },
    { key: "harris", label: "Harris corners", module: "features", description: "Find and circle corners.", params: [P("sensitivity", "Threshold", "float", 0.05, 0.01, 0.2, 0.01)], run: harris },
  ];

  const moduleLabels = { preprocessing: "Preprocessing", features: "Feature extraction" };

  function parseParam(param, raw) {
    if (param.kind === "choice") {
      const value = String(raw);
      if (!param.choices.includes(value)) throw new ValidationError(`'${param.label}' must be one of: ${param.choices.join(", ")}.`);
      return value;
    }
    const number = Number(raw);
    if (raw === "" || raw === null || raw === undefined || Number.isNaN(number)) throw new ValidationError(`'${param.label}' must be a number.`);
    if (number < param.min || number > param.max) throw new ValidationError(`'${param.label}' must be between ${param.min} and ${param.max}.`);
    return param.kind === "int" ? Math.round(number) : number;
  }

  /** Validate settings, run one operation and return { image, info }. */
  function runOperation(key, img, rawParams = {}) {
    const op = operations.find((o) => o.key === key);
    if (!op) throw new ValidationError("Unknown operation. Choose one from the list.");
    const params = {};
    for (const p of op.params) params[p.name] = parseParam(p, rawParams[p.name] !== undefined ? rawParams[p.name] : p.default);
    const result = op.run(img, params);
    return result.data ? { image: result, info: {} } : { image: result.image, info: result.info || {} };
  }

  return { ValidationError, operations, moduleLabels, runOperation, histogram, imageStats, toGray, isGray, makeImage };
})();

if (typeof module !== "undefined" && module.exports) module.exports = CV;
