const seg = (id) => document.getElementById(id);
let segReady = false;
let segBusy = false;

function segCard(grid, key, title, note, output, downloadName) {
  const card = document.createElement('figure');
  card.className = 'output-card';
  const stage = document.createElement('div');
  stage.className = 'image-stage';
  const image = document.createElement('img');
  image.src = output.url;
  image.alt = title;
  stage.append(image);
  const caption = document.createElement('figcaption');
  const heading = document.createElement('h3');
  heading.textContent = title;
  const description = document.createElement('p');
  description.textContent = `${note} Output shape: ${output.shape.join(' × ')}.`;
  const download = document.createElement('a');
  download.textContent = 'Download PNG ↓';
  download.href = output.url;
  download.download = `lab07-${downloadName || key}.png`;
  caption.append(heading, description, download);
  card.append(stage, caption);
  grid.append(card);
}

seg('seg-image').addEventListener('change', async () => {
  const file = seg('seg-image').files[0];
  segReady = false;
  seg('seg-run').disabled = true;
  seg('seg-semantic-run').disabled = true;
  seg('seg-results').hidden = true;
  seg('seg-empty').hidden = false;
  if (!file) return;
  if (file.size > 10 * 1024 * 1024) {
    seg('seg-message').textContent = 'Choose an image smaller than 10 MB.';
    seg('seg-message').classList.add('error');
    return;
  }
  const url = URL.createObjectURL(file);
  const image = new Image();
  try {
    image.src = url;
    await image.decode();
    const {naturalWidth: w, naturalHeight: h} = image;
    if (w * h > 12000000) throw new Error('Choose an image with at most 12 million pixels.');
    seg('seg-seed-x').max = w - 1;
    seg('seg-seed-y').max = h - 1;
    seg('seg-seed-x').value = Math.floor(w / 2);
    seg('seg-seed-y').value = Math.floor(h / 2);
    seg('seg-file-info').textContent = `${file.name} · ${w} × ${h} pixels`;
    segReady = true;
    seg('seg-run').disabled = false;
    seg('seg-semantic-run').disabled = false;
    seg('seg-message').classList.remove('error');
    seg('seg-message').textContent = 'Ready. Set the region seed or run with the defaults.';
  } catch (error) {
    seg('seg-message').textContent = error.message;
    seg('seg-message').classList.add('error');
  } finally { URL.revokeObjectURL(url); }
});

seg('segmentation-form').addEventListener('submit', async (event) => {
  event.preventDefault();
  if (!segReady || segBusy) return;
  const image = seg('seg-image').files[0];
  const fields = {
    polarity: seg('seg-polarity').value,
    block_size: Number(seg('seg-block').value),
    adaptive_c: Number(seg('seg-adaptive-c').value),
    seed_x: Number(seg('seg-seed-x').value), seed_y: Number(seg('seg-seed-y').value),
    tolerance: Number(seg('seg-tolerance').value),
    canny_low: Number(seg('seg-canny-low').value), canny_high: Number(seg('seg-canny-high').value),
    foreground_fraction: Number(seg('seg-foreground').value)
  };
  const body = new FormData();
  body.append('image', image);
  body.append('options', JSON.stringify(fields));
  const truth = seg('seg-ground-truth').files[0];
  if (truth) body.append('ground_truth', truth);
  segBusy = true;
  seg('seg-run').disabled = true;
  seg('seg-message').textContent = 'Processing segmentation methods…';
  seg('seg-message').classList.remove('error');
  try {
    const response = await fetch('/api/labs/7/process', {method: 'POST', body});
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || 'Segmentation failed.');
    seg('seg-summary').textContent = `${data.properties.width} × ${data.properties.height} · Otsu threshold ${data.properties.otsu_threshold} · ${data.properties.contours} contours · ${data.properties.watershed_regions} watershed region(s)`;
    seg('seg-details').textContent = `Region seed (${data.properties.seed.x}, ${data.properties.seed.y}) selected ${data.properties.region_pixels.toLocaleString()} pixels. ${truth ? 'IoU compares each output mask with the uploaded reference mask.' : 'Upload a same-size ground-truth mask to calculate IoU.'}`;
    const metrics = document.createElement('table');
    metrics.className = 'feature-table';
    metrics.innerHTML = '<caption>Foreground area and optional overlap with ground truth</caption><thead><tr><th>Method</th><th>Foreground pixels</th><th>IoU</th></tr></thead>';
    const tbody = document.createElement('tbody');
    for (const item of data.metrics) {
      const row = document.createElement('tr');
      for (const value of [item.method, item.foreground_pixels.toLocaleString(), item.iou === null ? 'Upload a mask to calculate' : item.iou.toFixed(4)]) {
        const cell = document.createElement('td'); cell.textContent = value; row.append(cell);
      }
      tbody.append(row);
    }
    metrics.append(tbody);
    seg('seg-metrics').replaceChildren(metrics);
    const labels = {
      original: ['Original image', 'Input image used by every method.'],
      otsu: ['Otsu global threshold', 'One automatically selected grayscale threshold.'],
      adaptive: ['Adaptive Gaussian threshold', `Local threshold with block size ${fields.block_size} and C=${fields.adaptive_c}.`],
      region: ['Seed-based region growing', 'Colored region grown from the selected seed using fixed-range color tolerance.'],
      region_mask: ['Region mask', 'White pixels belong to the grown region.'],
      canny: ['Canny edges', `Gaussian-smoothed edges with thresholds ${fields.canny_low}/${fields.canny_high}.`],
      contours: ['Closed contours', 'External contours detected from the Canny edge map.'],
      watershed: ['Watershed labels', 'Colored regions are separated by red watershed boundaries.']
    };
    const grid = seg('seg-output-grid'); grid.replaceChildren();
    for (const [key, output] of Object.entries(data.images)) segCard(grid, key, labels[key][0], labels[key][1], output);
    seg('seg-results').hidden = false;
    seg('seg-empty').hidden = true;
    seg('seg-message').textContent = 'Segmentation complete. Download the outputs for your report.';
  } catch (error) {
    seg('seg-message').textContent = error.message;
    seg('seg-message').classList.add('error');
  } finally { segBusy = false; seg('seg-run').disabled = !segReady; }
});

seg('seg-semantic-run').addEventListener('click', async () => {
  if (!segReady) return;
  const button = seg('seg-semantic-run');
  button.disabled = true;
  seg('seg-semantic-status').textContent = 'Running DeepLabV3. First use may download pretrained model weights.';
  try {
    const body = new FormData(); body.append('image', seg('seg-image').files[0]);
    const response = await fetch('/api/labs/7/semantic', {method: 'POST', body});
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || 'Semantic segmentation failed.');
    seg('seg-semantic-classes').textContent = `Predicted classes and pixel counts: ${data.classes.map(item => `${item.name}: ${item.pixels.toLocaleString()}`).join(' · ')}`;
    const grid = seg('seg-semantic-grid'); grid.replaceChildren();
    segCard(grid, 'semantic', 'DeepLabV3 semantic overlay', 'Pretrained per-pixel class predictions blended with the input image.', data.images.semantic);
    segCard(grid, 'semantic-mask', 'Semantic class mask', 'Each predicted class has a distinct display color.', data.images.semantic_mask);
    seg('seg-semantic-results').hidden = false;
    seg('seg-semantic-status').textContent = 'Semantic prediction complete.';
  } catch (error) { seg('seg-semantic-status').textContent = error.message; }
  finally { button.disabled = !segReady; }
});

fetch('/api/labs/7/semantic-capabilities').then(response => response.json()).then(data => {
  seg('seg-semantic-status').textContent = data.message;
}).catch(() => { seg('seg-semantic-status').textContent = 'Could not check semantic segmentation dependencies.'; });
