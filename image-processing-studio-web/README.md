# Image Processing Studio (HTML / CSS / JavaScript)

A browser app for exploring classical computer-vision operations. Open an image, apply an operation, and compare the result with the input, including histograms and measurements. Built with plain HTML, CSS and JavaScript for the Computer Vision (CV) course project. No server, framework or internet connection is needed.

## Overview
The page (`index.html`, `style.css`, `app.js`) handles uploading, controls and drawing. The algorithms are implemented from scratch in `cv.js` and work on RGBA pixel arrays. The list of operations and their settings comes from one registry in `cv.js`, so the page builds its controls automatically.

## Features
- **Preprocessing module:** grayscale, resize, Gaussian blur, median blur, histogram equalization, brightness/contrast, thresholding (binary, Otsu, adaptive), sharpen, negative, morphology (erode, dilate, open, close)
- **Feature extraction module:** Canny edges (with non-maximum suppression and hysteresis), Sobel, Laplacian, contours (Otsu + connected components, with count), Harris corners (with count)
- **Analysis module:** RGB/gray histograms, size, average brightness, contrast, edge density, PSNR against the input
- Chain several operations ("Use result as input"), reset, download the result
- Validation (file type, 5 MB limit, image decoding, setting ranges), clear error messages, activity log, large photos scaled to 1000 px for speed

## Technologies
HTML5, CSS3, JavaScript (ES2015+), Canvas API, Node.js (optional, for tests only), Git

## Project Structure
```
image-processing-studio-web/
├── index.html      # Page layout
├── style.css       # Styling
├── app.js          # UI: upload, controls, drawing, log
├── cv.js           # Algorithms, analysis and operation registry
├── tests/cv.test.js
├── statement.md
└── README.md
```

## Run
Double-click `index.html` (or drag it into Chrome, Edge or Firefox). That is all.

## Testing
Requires Node.js 18 or newer:
```bash
node tests/cv.test.js
```
The tests cover every operation with default settings, specific results (grayscale, resize, negative, median noise removal, thresholding, Canny, contours, Harris, morphology), histogram and PSNR calculations, and rejection of invalid settings.

## Adding a New Operation
1. Write a function in `cv.js` that takes an image and its settings and returns an image.
2. Add one entry to the `operations` list in `cv.js`.

The page picks it up automatically.

## Screenshots
Add screenshots here (loaded image, a result with histograms, a chained pipeline).
