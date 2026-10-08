# Problem Statement

## Problem
Students and beginners who want to learn computer vision usually have to write code or install heavy tools before they can see what an operation such as blurring, thresholding or edge detection actually does to an image. This makes it hard to build intuition about how the techniques behave and how their settings change the result.

## Scope
Image Processing Studio is a browser-based application that lets a user open one image, apply classical (non-deep-learning) computer-vision operations, and compare the result with the input. It covers preprocessing, feature extraction and analysis. All processing happens on the user's device in plain JavaScript, so nothing is uploaded and no server or installation is needed. It does not cover model training or video.

## Target Users
- Students learning computer vision
- Teachers who want a quick live demonstration of image-processing operations
- Beginners who want to experiment with image-processing concepts without writing code

## High-Level Features
1. **Preprocessing:** grayscale, resize, Gaussian blur, median blur, histogram equalization, brightness and contrast, thresholding (binary, Otsu, adaptive), sharpen, negative, morphology
2. **Feature extraction:** Canny, Sobel and Laplacian edges, contour outlining and counting, Harris corner detection
3. **Analysis:** side-by-side input and result, RGB / gray histograms, measurements (brightness, contrast, edge density, PSNR)
4. **Workflow:** chain operations by using a result as the next input, reset to the original, download the result
5. **Reliability:** input validation, clear error messages and an on-page activity log
