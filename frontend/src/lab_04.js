(() => {
  let revision = 0;
  let ready = false;
  let busy = false;
  function status(text, error = false) {
    $('feature-message').textContent = text;
    $('feature-message').classList.toggle('error', error);
  }
  async function availability() {
    try {
      const response = await fetch('/api/labs/4/capabilities');
      if (!response.ok) throw new Error('Could not check detectors. Try reloading the page.');
      const data = await response.json();
      $('feature-capabilities').replaceChildren();
      for (const item of data.algorithms) {
        $('feature-capabilities').append(element('p', `${item.name}: ${item.available ? 'Available' : item.reason}`));
      }
      if (data.algorithms.some(item => !item.available && item.name === 'SURF')) {
        $('feature-capabilities').append(element('p', 'SIFT and ORB results remain labeled as SIFT and ORB. Full SURF / U-SURF comparisons require a compatible OpenCV build; those outputs are unavailable in this environment.'));
      }
    } catch (error) { $('feature-capabilities').textContent = error.message; }
  }
  async function chooseImages() {
    const current = ++revision;
    const files = [$('feature-image-a').files[0], $('feature-image-b').files[0]];
    ready = false;
    $('feature-run').disabled = true;
    $('feature-results').hidden = true;
    $('feature-empty').hidden = false;
    $('feature-file-info').textContent = 'Each image: up to 4 million pixels. Combined files must be under 10 MB.';
    if (files.some(file => !file)) return status('Choose both images to begin.');
    if (files.reduce((sum, file) => sum + file.size, 0) > 10 * 1024 * 1024 - 8192) return status('Combined files must be under 10 MB.', true);
    try {
      const dimensions = [];
      for (const file of files) {
        const url = URL.createObjectURL(file);
        try {
          const image = new Image();
          image.src = url;
          await image.decode();
          if (image.naturalWidth * image.naturalHeight > 4000000) throw new Error('Each image must have at most 4 million pixels.');
          dimensions.push(`${file.name}: ${image.naturalWidth} x ${image.naturalHeight}`);
        } finally { URL.revokeObjectURL(url); }
      }
      if (current !== revision) return;
      $('feature-file-info').textContent = dimensions.join(' / ');
      ready = true;
      $('feature-run').disabled = busy;
      status('Ready. The same image pair will be used for each available detector.');
    } catch (error) {
      if (current === revision) status(error.message.startsWith('Each') ? error.message : 'Cannot decode one of the images. Choose valid image files.', true);
    }
  }
  $('feature-image-a').addEventListener('change', chooseImages);
  $('feature-image-b').addEventListener('change', chooseImages);
  $('feature-form').addEventListener('submit', async event => {
    event.preventDefault();
    if (!ready || busy) return;
    const current = revision;
    const files = [$('feature-image-a').files[0], $('feature-image-b').files[0]];
    const body = new FormData();
    body.append('image_a', files[0]);
    body.append('image_b', files[1]);
    body.append('options', JSON.stringify({hessian_threshold: Number($('feature-hessian').value), ratio: Number($('feature-ratio').value)}));
    busy = true;
    $('feature-run').disabled = true;
    status('Detecting features and matching descriptors...');
    try {
      const response = await fetch('/api/labs/4/process', {method: 'POST', body});
      const data = await response.json();
      if (!response.ok) throw new Error(data.error || 'Feature processing failed.');
      if (current !== revision) return;
      renderFeatures(data, files.map(file => file.name));
      const missing = data.algorithms.filter(item => !item.available).map(item => item.name);
      status(missing.length ? `Available detector results ready. Unavailable: ${missing.join(', ')}.` : 'Feature comparison complete.');
    } catch (error) {
      if (current === revision) status(`${error.message} Any visible results belong to the previous successful run.`, true);
    } finally { busy = false; $('feature-run').disabled = !ready; }
  });
  function figure(title, url, filename) {
    const card = element('figure', '', 'output-card');
    const stage = element('div', '', 'image-stage');
    const image = document.createElement('img');
    image.src = url;
    image.alt = title;
    stage.append(image);
    const caption = document.createElement('figcaption');
    const link = element('a', 'Download PNG');
    link.href = url;
    link.download = `lab04-${filename}.png`;
    caption.append(element('h3', title), link);
    card.append(stage, caption);
    return card;
  }
  function renderFeatures(data, filenames) {
    const p = data.parameters;
    $('feature-run-details').textContent = `A: ${filenames[0]} / B: ${filenames[1]}. Grayscale inputs. Hessian threshold: ${p.hessian_threshold} (SURF/U-SURF only). Ratio: ${p.ratio}. Descriptor cap: ${p.descriptor_cap} per image.`;
    $('feature-originals').replaceChildren(figure('Original A (grayscale)', data.originals.a, 'original-a'), figure('Original B (grayscale)', data.originals.b, 'original-b'));
    $('feature-summary').replaceChildren();
    $('feature-algorithms').replaceChildren();
    for (const item of data.algorithms) {
      const row = document.createElement('tr');
      const heading = element('th', item.name);
      heading.scope = 'row';
      row.append(heading);
      if (!item.available) {
        const cell = element('td', item.reason);
        cell.colSpan = 3;
        row.append(cell);
      } else {
        row.append(element('td', `${item.keypoints_a} / ${item.keypoints_b}`), element('td', `${item.detect_ms_a} / ${item.detect_ms_b}`), element('td', String(item.matches)));
        const section = element('section', '', 'report');
        section.append(element('h2', item.name));
        const shape = value => value ? value.join(' x ') : 'none';
        section.append(element('p', `Descriptors A: ${shape(item.descriptor_a)}; B: ${shape(item.descriptor_b)}; type: ${item.descriptor_dtype || 'none'}. Used keypoints: ${item.used_a} / ${item.used_b}. ${item.distance} distance, matching time ${item.match_ms} ms. Showing ${item.displayed_matches} of ${item.matches} ratio-test matches.`, 'muted'));
        if (!item.matches) section.append(element('p', 'No matches passed the ratio test, or there were too few descriptors. Try a more textured object, overlapping views, or a less strict ratio.', 'muted'));
        const grid = element('div', '', 'output-grid');
        grid.append(figure(`${item.name} keypoints A`, item.images.keypoints_a, `${item.name}-keypoints-a`), figure(`${item.name} keypoints B`, item.images.keypoints_b, `${item.name}-keypoints-b`));
        section.append(grid, figure(`${item.name} candidate matches`, item.images.matches, `${item.name}-matches`));
        $('feature-algorithms').append(section);
      }
      $('feature-summary').append(row);
    }
    $('feature-empty').hidden = true;
    $('feature-results').hidden = false;
  }
  availability();
})();
