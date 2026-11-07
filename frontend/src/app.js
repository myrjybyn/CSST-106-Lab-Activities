const $ = (id) => document.getElementById(id);
const optionKeys = ['pixel_x', 'pixel_y', 'blue', 'green', 'red', 'crop_x', 'crop_y', 'crop_width', 'crop_height', 'resize_width', 'resize_height'];
let generation = 0;
let ready = false;
let busy = false;

// Keep lab discovery separate so additional modules can extend the application.
async function loadLabs() {
  try {
    const response = await fetch('/api/labs');
    if (!response.ok) throw new Error('Could not load the laboratory navigation. Reload this page.');
    const data = await response.json();
    for (const lab of data.labs) {
      const link = document.createElement('a');
      link.href = `#${lab.id}`;
      link.dataset.view = lab.id;
      link.textContent = `LAB ${String(lab.number).padStart(2, '0')}`;
      const subtitle = document.createElement('small');
      subtitle.textContent = lab.title;
      link.append(subtitle);
      $('lab-nav').append(link);
    }
    const reportsLink = element('a', 'Reports');
    reportsLink.href = '#reports';
    reportsLink.dataset.view = 'reports';
    $('lab-nav').append(reportsLink);
    selectView();
  } catch (error) { message(error.message, true); }
}

function message(text, error = false) {
  $('message').textContent = text;
  $('message').classList.toggle('error', error);
}

$('image').addEventListener('change', async () => {
  const current = ++generation;
  const file = $('image').files[0];
  ready = false;
  $('run').disabled = true;
  $('results').hidden = true;
  $('empty-state').hidden = false;
  $('file-info').textContent = 'A color image works best for comparing BGR and grayscale.';
  if (!file) return message('Choose an image to begin.');
  if (file.size > 10 * 1024 * 1024) return message('Choose an image smaller than 10 MB.', true);
  const url = URL.createObjectURL(file);
  const image = new Image();
  try {
    image.src = url;
    await image.decode();
    if (current !== generation) return;
    const w = image.naturalWidth, h = image.naturalHeight;
    if (w * h > 12000000) throw new Error('Choose an image with at most 12 million pixels.');
    for (const key of ['pixel_x', 'crop_x']) { $(key).max = w - 1; $(key).value = 0; }
    for (const key of ['pixel_y', 'crop_y']) { $(key).max = h - 1; $(key).value = 0; }
    $('crop_width').max = w; $('crop_height').max = h;
    $('crop_width').value = Math.max(1, Math.floor(w / 2));
    $('crop_height').value = Math.max(1, Math.floor(h / 2));
    $('resize_width').value = Math.min(4096, Math.max(1, Math.floor(w / 2)));
    $('resize_height').value = Math.min(4096, Math.max(1, Math.floor(h / 2)));
    $('file-info').textContent = `${file.name} · ${w} × ${h} pixels · ${(file.size / 1024).toFixed(0)} KB`;
    ready = true;
    $('run').disabled = busy;
    message('Ready. Adjust the controls or run with the default settings.');
  } catch (error) {
    if (current === generation) message(error.message.startsWith('Choose') ? error.message : 'Cannot open this image. Choose a PNG, JPEG, BMP or WebP image.', true);
  } finally { URL.revokeObjectURL(url); }
});

$('lab-form').addEventListener('submit', async (event) => {
  event.preventDefault();
  if (!ready || busy) return;
  const current = generation;
  const file = $('image').files[0];
  const options = Object.fromEntries(optionKeys.map(key => [key, Number($(key).value)]));
  const body = new FormData();
  body.append('image', file);
  body.append('options', JSON.stringify(options));
  busy = true;
  $('run').disabled = true;
  message('Processing your image…');
  try {
    const response = await fetch('/api/labs/1/process', {method: 'POST', body});
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || 'Image processing failed.');
    if (current !== generation) return;
    render(data, file.name);
    message('Experiment complete. Download the outputs for your report.');
  } catch (error) {
    if (current === generation) message(`${error.message} Any visible results are from the previous successful run.`, true);
  } finally {
    busy = false;
    $('run').disabled = !ready;
  }
});

function element(tag, text, className) {
  const node = document.createElement(tag);
  node.textContent = text;
  if (className) node.className = className;
  return node;
}

function render(data, filename) {
  const p = data.properties, pixel = data.pixel, ops = data.operations;
  $('properties').replaceChildren();
  const properties = [['Resolution (W × H)', `${p.width} × ${p.height}`], ['Color shape (H, W, C)', p.shape.join(' × ')], ['Data type / range', `${p.dtype} / 0–255`], ['Channel order', 'B → G → R']];
  for (const [label, value] of properties) {
    const card = element('div', '', 'property');
    card.append(element('small', label), element('strong', value));
    $('properties').append(card);
  }
  $('pixel-details').textContent = `Pixel (x=${pixel.x}, y=${pixel.y}) → array[y, x]. Original BGR: [${pixel.before_bgr.join(', ')}]; changed BGR: [${pixel.after_bgr.join(', ')}]. Original grayscale intensity: ${pixel.grayscale_intensity}. Grayscale shape: (${p.grayscale_shape.join(', ')}). Total pixels: ${p.pixels.toLocaleString()}; decoded color array: ${p.decoded_bytes.toLocaleString()} bytes.`;
  const captions = {
    original: ['Original image', 'Decoded with cv2.imdecode into a three-channel BGR array.'],
    grayscale: ['Grayscale', 'cv2.cvtColor with COLOR_BGR2GRAY converts BGR into one intensity channel.'],
    pixel_modified: ['One pixel changed', `Only pixel (${pixel.x}, ${pixel.y}) changes. The full-size difference can be subtle.`],
    pixel_zoom: ['Pixel neighborhood · magnified', 'Up to 9 × 9 pixels around the changed pixel, enlarged with nearest-neighbor interpolation. Near an edge, the neighborhood is clipped.'],
    cropped: ['Cropped region', `Rows ${ops.crop.y}:${ops.crop.y + ops.crop.height}, columns ${ops.crop.x}:${ops.crop.x + ops.crop.width}. End indices are excluded.`],
    resized: ['Resized image', `cv2.resize → ${ops.resize.width} × ${ops.resize.height} pixels. Independent width and height can change the aspect ratio.`],
    rotated: ['Rotated 90° clockwise', 'cv2.rotate with ROTATE_90_CLOCKWISE swaps the height and width.']
  };
  $('output-grid').replaceChildren();
  for (const [name, output] of Object.entries(data.images)) {
    const [title, description] = captions[name];
    const card = element('figure', '', 'output-card');
    const stage = element('div', '', 'image-stage');
    const image = document.createElement('img');
    image.src = output.url;
    image.alt = `${title} from ${filename}`;
    stage.append(image);
    const caption = document.createElement('figcaption');
    const download = element('a', 'Download PNG ↓');
    download.href = output.url;
    download.download = `lab01-${name}.png`;
    caption.append(element('h3', title), element('p', `${description} Shape: (${output.shape.join(', ')}).`), download);
    card.append(stage, caption);
    $('output-grid').append(card);
  }
  $('empty-state').hidden = true;
  $('results').hidden = false;
}

function selectView() {
  const reports = location.hash === '#reports';
  const views = {'lab-01': 'lab-view', 'lab-02': 'lab-02-view', 'lab-03': 'lab-03-view', 'lab-04': 'lab-04-view'};
  const selected = Object.hasOwn(views, location.hash.slice(1)) ? location.hash.slice(1) : 'lab-01';
  for (const [lab, view] of Object.entries(views)) $(view).hidden = reports || lab !== selected;
  $('reports-view').hidden = !reports;
  $('breadcrumb').textContent = reports ? 'Laboratory workspace / Reports' : `Laboratory workspace / LAB ${selected.slice(-2)}`;
  document.querySelectorAll('#lab-nav a').forEach(link => {
    if (link.dataset.view === (reports ? 'reports' : selected)) link.setAttribute('aria-current', 'page');
    else link.removeAttribute('aria-current');
  });
}

function reportCard(report) {
  const card = element('article', '', 'report-card');
  const label = report.lab_number === null ? 'General report' : `LAB ${String(report.lab_number).padStart(2, '0')}`;
  card.append(element('small', label), element('h3', report.filename));
  const actions = element('div', '', 'report-actions');
  const preview = element('button', 'View PDF', 'secondary');
  preview.type = 'button';
  preview.addEventListener('click', () => {
    location.hash = 'reports';
    selectView();
    $('pdf-frame').src = report.url;
    $('pdf-title').textContent = `${label} / ${report.filename}`;
    $('pdf-open').href = report.url;
    $('pdf-viewer').hidden = false;
    $('pdf-viewer').scrollIntoView({behavior: 'smooth', block: 'start'});
  });
  const download = element('a', 'Download PDF', 'text-link');
  download.href = report.url;
  download.download = report.filename;
  actions.append(preview, download);
  card.append(actions);
  return card;
}

async function loadReports() {
  $('refresh-reports').disabled = true;
  $('reports-status').textContent = 'Loading reports...';
  try {
    const response = await fetch('/api/reports', {cache: 'no-store'});
    if (!response.ok) throw new Error('Could not load reports. Try refreshing.');
    const data = await response.json();
    $('report-list').replaceChildren();
    $('lab-reports').replaceChildren();
    $('lab-02-reports').replaceChildren();
    $('lab-03-reports').replaceChildren();
    $('lab-04-reports').replaceChildren();
    $('pdf-viewer').hidden = true;
    $('pdf-frame').removeAttribute('src');
    for (const report of data.reports) {
      $('report-list').append(reportCard(report));
      if (report.lab_number === 1) $('lab-reports').append(reportCard(report));
      if (report.lab_number === 2) $('lab-02-reports').append(reportCard(report));
      if (report.lab_number === 3) $('lab-03-reports').append(reportCard(report));
      if (report.lab_number === 4) $('lab-04-reports').append(reportCard(report));
    }
    if (!$('lab-reports').childElementCount) {
      $('lab-reports').append(element('p', 'No LAB 01 PDF yet. Add reports/lab_01.pdf, then refresh the Reports tab.', 'muted'));
    }
    if (!$('lab-02-reports').childElementCount) {
      $('lab-02-reports').append(element('p', 'No LAB 02 PDF yet. Add reports/lab_02.pdf, then refresh the Reports tab.', 'muted'));
    }
    if (!$('lab-03-reports').childElementCount) {
      $('lab-03-reports').append(element('p', 'No LAB 03 PDF yet. Add reports/lab_03.pdf, then refresh the Reports tab.', 'muted'));
    }
    if (!$('lab-04-reports').childElementCount) {
      $('lab-04-reports').append(element('p', 'No LAB 04 PDF yet. Add reports/lab_04.pdf, then refresh the Reports tab.', 'muted'));
    }
    $('reports-status').textContent = data.reports.length ? `${data.reports.length} PDF report(s) available.` : 'No reports yet. Add your PDFs to reports/ (for example, lab_01.pdf), then click Refresh reports.';
  } catch (error) {
    $('reports-status').textContent = error.message;
    $('lab-reports').replaceChildren(element('p', error.message, 'muted'));
    $('lab-02-reports').replaceChildren(element('p', error.message, 'muted'));
    $('lab-03-reports').replaceChildren(element('p', error.message, 'muted'));
    $('lab-04-reports').replaceChildren(element('p', error.message, 'muted'));
  } finally { $('refresh-reports').disabled = false; }
}
window.addEventListener('hashchange', selectView);
$('refresh-reports').addEventListener('click', loadReports);
selectView();
loadLabs();
loadReports();
