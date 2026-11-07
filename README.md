# CSST106 — Computer Vision Laboratory Portfolio

A single Web UI for the course laboratory activities. Each implemented lab has a labeled navigation entry, a dedicated Python module, and an API endpoint. **LAB 01: Image Manipulation**, **LAB 02: Edge Detection**, and **LAB 03: Histogram Equalization and Contrast Adjustment** are implemented. **LAB 04: SURF Feature Detection** adds SIFT/ORB matching and SURF/U-SURF support when the OpenCV build permits it; add future labs to the same application as they are completed.

## Run locally

Use Python 3.13 (the environment verified for this project). From the repository root:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r backend/requirements.txt
.\.venv\Scripts\python.exe -m backend.app.main
```

If using the already verified global environment instead, run `python -m backend.app.main`. Open **http://127.0.0.1:5000**. Stop with Ctrl+C. Node.js is not required; Flask serves the HTML, CSS, JavaScript, and API together. In VS Code, select the same interpreter used to launch the server.

Dependencies: Flask, NumPy, Matplotlib (histogram plots), and the GUI-capable OpenCV contrib distribution already present in the checked environment. Do not install multiple OpenCV distributions into the same environment. Browser previews replace `cv2.imshow`; uploads are decoded with `cv2.imdecode` rather than reading a fixed file path.

## Lab Activity 1

1. Upload a color PNG, JPEG, BMP, or WebP image (up to 10 MB; at most 12 million decoded pixels).
2. Set the pixel coordinate and replacement BGR values, crop rectangle, and resize dimensions.
3. Run the experiment. Inspect original image dimensions, array shape, data type, pixel values, and channel order.
4. Compare the original, grayscale, modified pixel, magnified pixel neighborhood, cropped, resized, and 90° clockwise rotated outputs. Each operation starts from the original, rather than chaining transformations.
5. Download PNG outputs and capture screenshots for your independently written report. Save your PDF as `reports/lab_01.pdf` (or inside `reports/lab_01/`), then open the **Reports** tab and click **Refresh reports**. Read it in the embedded viewer or download it. The Lab 1 page also links to its PDFs. Print your report for hard-copy submission.

Coordinates use `x` for columns and `y` for rows; array access is `image[y, x]`. Cropping uses exclusive end indices. Resizing width and height independently can change aspect ratio. Grayscale uploads are decoded into BGR too, so three equal channels are expected in the original color representation. Transparent uploads are converted to three-channel color. Use an ordinary color photo for the clearest comparison.

## Organization

```text
backend/
  requirements.txt
  app/
    main.py          # Flask app factory and local entry point
    api/routes.py    # Lab registry and upload endpoints
    labs/lab_01.py   # Image manipulation, independent of Flask
    labs/lab_02.py   # Sobel and Canny edge detection
    labs/lab_03.py   # Histogram equalization, CLAHE, stretching, plots
    labs/lab_04.py   # Feature detection, capabilities, descriptor matching
  tests/test_lab_01.py
frontend/
  public/index.html  # Shared lab workspace
  src/app.js         # Lab 1, shared navigation, PDF library
  src/lab_02.js      # Lab 2 uploads, controls, comparison
  src/lab_03.js      # Lab 3 images, histograms, mode and CLAHE controls
  src/lab_04.js      # Lab 4 paired uploads, detector comparison, matches
  src/styles.css     # Responsive layout and PDF viewer
reports/
  README.md          # PDF naming guide and report checklist
  lab_01.pdf         # Your own report (add when ready)
```

For future labs, add a processing module, register a route and navigation entry, and implement the corresponding UI. Label activities with uppercase, two-digit numbering: **LAB 01**, **LAB 02**, etc., throughout navigation, headings, and report labels. In each lab introduction, put the lab number and week information above the main heading, and use the actual lab name as the main heading (for example, **Image Manipulation**). Do not repeat the lab name in a badge or replace it with a slogan. Keep reports aligned with the implemented techniques and actual outputs. Only implemented labs are advertised as available.

## Verification

```powershell
python -m unittest discover -s backend/tests -v
```

Tests use generated image fixtures and Flask's test client; they do not need a display or sample photo.

## GitHub submission

Maintain this organized source tree in your GitHub repository. Commit source, requirements, and documentation. Exclude virtual environments, secrets, and temporary uploads. PDF reports in `reports/` are included by Git so they can be reviewed with the application. Check their contents before committing. No GitHub upload or deployment has been performed by this implementation.

This is a local development application. Uploaded bytes and results are processed in memory; there is no upload archive, database, authentication, or public hosting configuration. The Stairs n Slabs font is served locally from `frontend/src/fonts/` and works offline.

## PDF report library

Write and export your own reports outside the app. Place PDFs in `reports/`; the website discovers them automatically on page load or after **Refresh reports**. Use `lab_01.pdf`, `lab_02.pdf`, or a subfolder such as `lab_01/my-report.pdf` to associate a report with a lab. Multiple reports per lab are supported. Other PDF names appear as general reports. The library supports future lab numbers without adding placeholder activities. Reports are displayed as submitted; the app does not generate answers or verify that a PDF matches an experiment.

## LAB 02 - Edge Detection (weeks 3-4)

The supplied document filename identifies Experiment 02 while its body says Activity 3. This application follows the requested sequence and labels the module **LAB 02**.

1. Open LAB 02 and upload the instructor's `input.jpg` if available, or another sample image.
2. Start with Canny thresholds 100 and 200, matching the handout. Lower must be less than upper, and both must be within 0-255.
3. Run edge detection. Both detectors use a grayscale image preprocessed with a 3-by-3 Gaussian blur and automatically derived sigma.
4. Compare the grayscale original, combined Sobel magnitude, and Canny binary edges side by side (stacked on narrow screens). Expand the preprocessing panel to inspect the blurred image. Download any output as PNG.
5. Change thresholds and rerun to explore their effect on Canny. Results identify the parameters used for the successful run; changing the form alone does not change existing results.

Sobel uses X and Y derivatives in `CV_64F`, kernel size 3, combined with `cv2.magnitude`. The magnitude is normalized to 0-255 for PNG display, matching the handout's per-image matplotlib display scaling. The unscaled maximum magnitude is shown separately. Canny uses OpenCV's default 3-by-3 aperture and L1 gradient norm, as in the handout. No additional libraries are required; the browser replaces matplotlib's display windows.

Prepare your own report PDF and place it at `reports/lab_02.pdf` or inside `reports/lab_02/`. No report answers are generated by the app.

## LAB 03 - Histogram Equalization and Contrast Adjustment (weeks 4-5)

1. Open **LAB 03** and upload a sample image, preferably with low contrast.
2. Select **Grayscale** or **Color (YUV luminance)**. Grayscale converts the input to one intensity channel. Color modifies only the Y channel, retaining U and V before reconstruction; reconstructed RGB colors may clip or round.
3. Choose the CLAHE clip limit (0.1-40, default 2.0) and tiles per axis (1-32, default 8). The grid counts regions, not pixels per tile.
4. Run the experiment to compare the original, globally equalized, CLAHE, and linearly stretched images. All methods operate independently on the same original channel.
5. Compare the 256-bin histograms, intensity ranges, means, and standard deviations. Histograms share a pixel-count scale. In color mode, histograms describe Y before color reconstruction. Download the output images and histogram PNGs as needed.

Global equalization uses `cv2.equalizeHist`. CLAHE uses `cv2.createCLAHE`. Linear stretching maps the input minimum and maximum to 0 and 255, rounding to the nearest integer. A constant channel remains unchanged when stretching because it has no range to expand. Matplotlib renders the histogram plots without a GUI.

Prepare your own observations, answers, and conclusion in your report. Place the completed PDF at `reports/lab_03.pdf` or inside `reports/lab_03/`, then refresh the Reports tab. The application does not fill in your report.

## LAB 04 - SURF Feature Detection (week 6)

**Environment limitation:** the installed OpenCV contrib 4.11 build exposes `SURF_create` but raises an error because nonfree functionality is disabled. Installing contrib alone does not enable SURF in this environment. LAB 04 reports this explicitly; SIFT and ORB are working alternatives allowed by the handout. SURF and U-SURF results are unavailable until Python uses an OpenCV contrib build compiled with `OPENCV_ENABLE_NONFREE=ON`. The app will detect support automatically after restart. No custom OpenCV build or environment replacement has been performed.

1. Open LAB 04 and select two overlapping views of an object or scene (up to 4 million pixels each, combined files under 10 MB). Use textured images; blank regions may have no keypoints.
2. Start with Hessian threshold 400 and nearest-neighbor ratio 0.75. The Hessian threshold affects only SURF and U-SURF. Increase it to explore changes in detected keypoint counts when SURF is available.
3. Run the comparison. Each available detector processes the same grayscale pair. View keypoint overlays with scale/orientation, descriptor shapes and data types, detection timings, and candidate matches.
4. Matching uses brute-force L2 distance for floating-point SURF/SIFT descriptors and Hamming distance for binary ORB descriptors. For each descriptor in A, find the two nearest in B, keeping the best only if `distance_1 < ratio * distance_2`. There must be at least two descriptors in B. Matches are not checked by homography or another geometric verification step.
5. At most the strongest 1,000 keypoints/descriptors per image are used for drawing and matching. Detected counts are shown separately. SIFT and ORB are configured with a 1,000-feature target. The best 50 accepted candidate matches are drawn for readability.
6. When SURF support is available, both orientation-invariant SURF and upright U-SURF run with the same Hessian threshold. Try rotated views to compare them. U-SURF is created using `upright=True`; it does not estimate keypoint orientations.

Timings measure a single detection-and-description run per image, excluding decoding and drawing. Matching time is shown separately. They are observations for the current inputs and hardware, not a controlled speed benchmark. Empty descriptor sets and insufficient matches produce zero matches with guidance rather than crashing.

Download visualizations as PNGs. Prepare your own report and place it at `reports/lab_04.pdf` or inside `reports/lab_04/`. Report SURF/U-SURF as unavailable unless you have actually run them using a compatible build; SIFT/ORB output is never relabeled as SURF output.
