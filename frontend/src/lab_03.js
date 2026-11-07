// LAB 03 retains its upload and last successful result when changing tabs.
(() => {
  let revision = 0;
  let ready = false;
  let busy = false;
  function status(text, error = false) {
    $('hist-message').textContent = text;
    $('hist-message').classList.toggle('error', error);
  }
  $('hist-image').addEventListener('change', async () => {
    const current = ++revision;
    const file = $('hist-image').files[0];
    ready = false;
    $('hist-run').disabled = true;
    $('hist-results').hidden = true;
    $('hist-empty').hidden = false;
    $('hist-file-info').textContent = 'Try an image with low contrast, then compare with a well-lit image.';
    if (!file) return status('Choose an image to begin.');
    if (file.size > 10 * 1024 * 1024) return status('Choose an image smaller than 10 MB.', true);
    const url = URL.createObjectURL(file);
    try {
      const image = new Image();
      image.src = url;
      await image.decode();
      if (current !== revision) return;
      if (image.naturalWidth * image.naturalHeight > 12000000) throw new Error('Choose an image with at most 12 million pixels.');
      $('hist-file-info').textContent = `${file.name} / ${image.naturalWidth} x ${image.naturalHeight} pixels`;
      ready = true;
      $('hist-run').disabled = busy;
      status('Ready. Choose a processing mode and CLAHE settings.');
    } catch (error) {
      if (current === revision) status(error.message.startsWith('Choose') ? error.message : 'Cannot open this image. Choose a valid PNG, JPEG, BMP or WebP.', true);
    } finally { URL.revokeObjectURL(url); }
  });

  $('hist-form').addEventListener('submit', async event => {
    event.preventDefault();
    if (!ready || busy) return;
    const current = revision;
    const file = $('hist-image').files[0];
    const options = {mode: $('hist-mode').value, clip_limit: Number($('hist-clip').value), tile_grid: Number($('hist-grid').value)};
    const body = new FormData();
    body.append('image', file);
    body.append('options', JSON.stringify(options));
    busy = true;
    $('hist-run').disabled = true;
    status('Processing images and plotting histograms...');
    try {
      const response = await fetch('/api/labs/3/process', {method: 'POST', body});
      const data = await response.json();
      if (!response.ok) throw new Error(data.error || 'Histogram processing failed.');
      if (current !== revision) return;
      renderHistograms(data, file.name);
      status('Experiment complete. Images and histograms reflect this successful run.');
    } catch (error) {
      if (current === revision) status(`${error.message} Any visible results belong to the previous successful run.`, true);
    } finally {
      busy = false;
      $('hist-run').disabled = !ready;
    }
  });

  function download(url, filename, text) {
    const link = element('a', text);
    link.href = url;
    link.download = filename;
    return link;
  }

  function renderHistograms(data, filename) {
    const p = data.parameters;
    $('hist-run-details').textContent = `${filename} / ${data.properties.width} x ${data.properties.height} pixels / ${data.properties.channel}. CLAHE: clip limit ${p.clip_limit}, ${p.tile_grid} x ${p.tile_grid} tiles. Stretch input range: ${p.stretch_min}-${p.stretch_max}${p.stretch_min === p.stretch_max ? ' (constant channel; stretching leaves it unchanged)' : ' mapped to 0-255'}.`;
    const captions = {
      original: ['Original', 'Unmodified grayscale or color input for this mode.'],
      equalized: ['Global equalization', 'cv2.equalizeHist redistributes the intensity channel globally.'],
      clahe: ['CLAHE', 'Contrast-limited adaptive equalization processes local regions.'],
      stretched: ['Linear contrast stretching', 'A linear transform expands the original minimum-to-maximum range.']
    };
    $('hist-output-grid').replaceChildren();
    for (const [name, output] of Object.entries(data.images)) {
      const [title, description] = captions[name];
      const card = element('figure', '', 'output-card');
      const stage = element('div', '', 'image-stage');
      const image = document.createElement('img');
      image.src = output.url;
      image.alt = `${title} from ${filename}`;
      stage.append(image);
      const caption = document.createElement('figcaption');
      const stats = output.statistics;
      caption.append(element('h3', title), element('p', description),
        element('p', `${data.properties.channel}: range ${stats.min}-${stats.max}; mean ${stats.mean}; standard deviation ${stats.std}.`),
        download(output.url, `lab03-${name}.png`, 'Download image PNG'));
      const histogram = document.createElement('img');
      histogram.className = 'histogram-plot';
      histogram.src = output.histogram_url;
      histogram.alt = `${title}: 256-bin ${data.properties.channel} histogram. Intensity range ${stats.min}-${stats.max}, mean ${stats.mean}, standard deviation ${stats.std}.`;
      const chartDownload = download(output.histogram_url, `lab03-${name}-histogram.png`, 'Download histogram PNG');
      chartDownload.className = 'hist-download';
      caption.append(histogram, chartDownload);
      card.append(stage, caption);
      $('hist-output-grid').append(card);
    }
    $('hist-empty').hidden = true;
    $('hist-results').hidden = false;
  }
})();
