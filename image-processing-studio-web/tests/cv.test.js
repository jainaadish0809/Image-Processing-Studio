// Run with:  node tests/cv.test.js     (needs Node.js 18+; uses only built-in modules)
const assert = require("node:assert/strict");
const test = require("node:test");
const CV = require("../cv.js");

/** Black 160x120 image with a white rectangle and a red circle. */
function sample() {
  const img = CV.makeImage(160, 120);
  const d = img.data;
  for (let y = 0; y < 120; y++) {
    for (let x = 0; x < 160; x++) {
      const o = (y * 160 + x) * 4;
      let rgb = [0, 0, 0];
      if (x >= 40 && x < 120 && y >= 30 && y < 90) rgb = [255, 255, 255];
      if ((x - 80) ** 2 + (y - 60) ** 2 <= 225) rgb = [255, 0, 0];
      d[o] = rgb[0]; d[o + 1] = rgb[1]; d[o + 2] = rgb[2]; d[o + 3] = 255;
    }
  }
  return img;
}
const count = (arr, v) => arr.reduce((n, x) => n + (x === v ? 1 : 0), 0);

test("every operation runs with default settings", () => {
  for (const op of CV.operations) {
    const { image, info } = CV.runOperation(op.key, sample(), {});
    assert.equal(image.data.length, image.width * image.height * 4, op.key);
    assert.equal(typeof info, "object");
  }
});

test("grayscale gives equal R, G and B", () => {
  const { image } = CV.runOperation("grayscale", sample());
  assert.ok(CV.isGray(image));
});

test("resize halves the dimensions", () => {
  const { image } = CV.runOperation("resize", sample(), { scale_percent: 50 });
  assert.deepEqual([image.width, image.height], [80, 60]);
});

test("negative applied twice restores the image", () => {
  const once = CV.runOperation("negative", sample()).image;
  const twice = CV.runOperation("negative", once).image;
  assert.deepEqual(Array.from(twice.data), Array.from(sample().data));
});

test("even kernel sizes are accepted", () => {
  assert.doesNotThrow(() => CV.runOperation("gaussian_blur", sample(), { ksize: 4 }));
});

test("median blur removes a single noisy pixel", () => {
  const img = sample();
  const o = (10 * 160 + 10) * 4;
  img.data[o] = img.data[o + 1] = img.data[o + 2] = 255;
  const { image } = CV.runOperation("median_blur", img, { ksize: 3 });
  assert.equal(image.data[o], 0);
});

test("otsu threshold output is only black and white", () => {
  const { image } = CV.runOperation("threshold", sample(), { mode: "otsu" });
  const values = new Set();
  for (let j = 0; j < image.data.length; j += 4) values.add(image.data[j]);
  assert.deepEqual([...values].sort((a, b) => a - b), [0, 255]);
});

test("canny finds the rectangle edges", () => {
  const { image } = CV.runOperation("canny", sample(), { low: 50, high: 150 });
  const edgePixels = count(Array.from(image.data).filter((_, i) => i % 4 === 0), 255);
  assert.ok(edgePixels > 100);
});

test("contours finds at least one shape", () => {
  const { info } = CV.runOperation("contours", sample(), { min_area: 100 });
  assert.ok(info.contours_found >= 1);
});

test("harris finds the rectangle corners", () => {
  const { info } = CV.runOperation("harris", sample(), { sensitivity: 0.05 });
  assert.ok(info.corners_found >= 4);
});

test("erode shrinks and dilate grows the white area", () => {
  const white = (img) => count(Array.from(img.data).filter((_, i) => i % 4 === 1), 255);
  const base = white(sample());
  assert.ok(white(CV.runOperation("morphology", sample(), { operation: "erode", ksize: 5 }).image) < base);
  assert.ok(white(CV.runOperation("morphology", sample(), { operation: "dilate", ksize: 5 }).image) > base);
});

test("histogram counts every pixel", () => {
  const hist = CV.histogram(sample());
  assert.equal(hist.red.length, 256);
  assert.equal(hist.red.reduce((a, b) => a + b, 0), 160 * 120);
  assert.deepEqual(Object.keys(CV.histogram(CV.runOperation("grayscale", sample()).image)), ["gray"]);
});

test("PSNR is null for identical images and a number after blurring", () => {
  const img = sample();
  assert.equal(CV.imageStats(img, img).psnr_db, null);
  const blurred = CV.runOperation("gaussian_blur", img, { ksize: 9 }).image;
  assert.ok(CV.imageStats(img, blurred).psnr_db > 0);
});

test("invalid input is rejected with a ValidationError", () => {
  const bad = [
    ["threshold", { mode: "bogus" }],
    ["gaussian_blur", { ksize: 99 }],
    ["gaussian_blur", { ksize: "abc" }],
    ["canny", { low: 200, high: 100 }],
    ["not_an_operation", {}],
  ];
  for (const [key, params] of bad) {
    assert.throws(() => CV.runOperation(key, sample(), params), CV.ValidationError, key);
  }
});
