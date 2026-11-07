// Independent upload state preserves each lab's inputs and results when switching tabs.
(() => {
  let revision = 0;
  let ready = false;
  let busy = false;
  function status(text, error = false) {
    $('edge-message').textContent = text;
    $('edge-message').classList.toggle('error', error);
  }
  $('edge-image').addEventListener('change', async () => {
    const current = ++revision;
    const file = $('edge-image').files[0];
    ready = false;
    $('edge-run').disabled = true;
    $('edge-results').hidden = true;
    $('edge-empty').hidden = false;
    $('edge-file-info').textContent = 'Use input.jpg if supplied by your instructor, or choose your own image.';
    if (!file) return status('Choose an image to begin.');
    if (file.size > 10 * 1024 * 1024) return status('Choose an image smaller than 10 MB.', true);
    const url = URL.createObjectURL(file);
    try {
      const image = new Image();
      image.src = url;
      await image.decode();
      if (current !== revision) return;
      if (image.naturalWidth * image.naturalHeight > 12000000) throw new Error('Choose an image with at most 12 million pixels.');
      $('edge-file-info').textContent = `${file.name} / ${image.naturalWidth} x ${image.naturalHeight} pixels`;
      ready = true;
      $('edge-run').disabled = busy;
      status('Ready. Run with the handout defaults or adjust the thresholds.');
    } catch (error) {
      if (current === revision) status(error.message.startsWith('Choose') ? error.message : 'Cannot open this image. Choose a valid PNG, JPEG, BMP or WebP.', true);
    } finally { URL.revokeObjectURL(url); }
  });

  $('edge-form').addEventListener('submit', async event => {
    event.preventDefault();
    if (!ready || busy) return;
    const lower = Number($('lower-threshold').value);
    const upper = Number($('upper-threshold').value);
    if (lower >= upper) return status('Lower threshold must be less than upper threshold.', true);
    const current = revision;
    const file = $('edge-image').files[0];
    const body = new FormData();
    body.append('image', file);
    body.append('options', JSON.stringify({lower_threshold: lower, upper_threshold: upper}));
    busy = true;
    $('edge-run').disabled = true;
    status('Detecting edges...');
    try {
      const response = await fetch('/api/labs/2/process', {method: 'POST', body});
      const data = await response.json();
      if (!response.ok) throw new Error(data.error || 'Edge detection failed.');
      if (current !== revision) return;
      renderEdges(data, file.name);
      status('Experiment complete. Results show the parameters used for this run.');
    } catch (error) {
      if (current === revision) status(`${error.message} Any visible results belong to the previous successful run.`, true);
    } finally {
      busy = false;
      $('edge-run').disabled = !ready;
    }
  });

  function outputCard(name, title, description, output, filename) {
    const figure = element('figure', '', 'output-card');
    const stage = element('div', '', 'image-stage');
    const image = document.createElement('img');
    image.src = output.url;
    image.alt = `${title} from ${filename}`;
    stage.append(image);
    const caption = document.createElement('figcaption');
    const download = element('a', 'Download PNG');
    download.href = output.url;
    download.download = `lab02-${name}.png`;
    caption.append(element('h3', title), element('p', description), download);
    figure.append(stage, caption);
    return figure;
  }

  function renderEdges(data, filename) {
    const p = data.properties, params = data.parameters;
    $('edge-properties').replaceChildren();
    const facts = [['Resolution', `${p.width} x ${p.height}`], ['Canny edge pixels', p.canny_edge_pixels.toLocaleString()], ['Canny coverage', `${p.canny_edge_percent}%`], ['Max Sobel magnitude', String(p.sobel_max_magnitude)]];
    for (const [label, value] of facts) {
      const card = element('div', '', 'property');
      card.append(element('small', label), element('strong', value));
      $('edge-properties').append(card);
    }
    $('edge-run-details').textContent = `${filename} / Grayscale shape (${p.shape.join(', ')}). Gaussian kernel: 3 x 3. Sobel kernel: 3 x 3, CV_64F. Canny thresholds: ${params.lower_threshold}, ${params.upper_threshold}.`;
    $('edge-output-grid').replaceChildren(
      outputCard('original', 'Original (grayscale)', 'Decoded directly to grayscale, as in the handout.', data.images.original, filename),
      outputCard('sobel', 'Sobel edges', 'Combined X/Y gradient magnitude, normalized to 0-255 for display.', data.images.sobel, filename),
      outputCard('canny', 'Canny edges', 'Binary edge map after non-maximum suppression and hysteresis.', data.images.canny, filename)
    );
    $('edge-blur-output').replaceChildren(outputCard('blurred', 'Gaussian-blurred input', 'The same noise-reduced grayscale input is used by both detectors.', data.images.blurred, filename));
    $('edge-empty').hidden = true;
    $('edge-results').hidden = false;
  }
})();
