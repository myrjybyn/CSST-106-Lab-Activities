(() => {
  let cameraStream = null;
  let selectedImage = null;
  let trainedId = null;

  async function capability() {
    try {
      const response = await fetch('/api/labs/6/capabilities');
      const data = await response.json();
      $('yolo-capability').textContent = data.available
        ? `Ultralytics ${data.version} is ready. ${data.message}`
        : data.message;
      $('yolo-detect').disabled = !data.available;
      $('yolo-train').disabled = !data.available;
      $('yolo-camera-start').disabled = !data.available;
    } catch {
      $('yolo-capability').textContent = 'Could not check the Ultralytics installation. Restart Flask and reload this page.';
    }
  }

  function setStatus(id, text, error = false) {
    $(id).textContent = text;
    $(id).classList.toggle('error', error);
  }

  $('yolo-image').addEventListener('change', () => {
    selectedImage = $('yolo-image').files[0] || null;
    $('yolo-detect').disabled = !selectedImage;
    if (selectedImage) setStatus('yolo-message', `${selectedImage.name} ready for detection.`);
  });

  $('yolo-camera-start').addEventListener('click', async () => {
    try {
      cameraStream = await navigator.mediaDevices.getUserMedia({video: {facingMode: 'environment'}, audio: false});
      $('yolo-video').srcObject = cameraStream;
      $('yolo-video').hidden = false;
      $('yolo-camera-capture').disabled = false;
      $('yolo-camera-stop').disabled = false;
      $('yolo-camera-start').disabled = true;
      setStatus('yolo-message', 'Camera ready. Capture a frame to run detection.');
    } catch (error) {
      setStatus('yolo-message', `Could not start webcam: ${error.message}. Allow camera access and use localhost or HTTPS.`, true);
    }
  });
  $('yolo-camera-stop').addEventListener('click', stopCamera);
  function stopCamera() {
    if (cameraStream) cameraStream.getTracks().forEach(track => track.stop());
    cameraStream = null;
    $('yolo-video').srcObject = null;
    $('yolo-video').hidden = true;
    $('yolo-camera-capture').disabled = true;
    $('yolo-camera-stop').disabled = true;
    $('yolo-camera-start').disabled = false;
  }
  window.addEventListener('pagehide', stopCamera);

  $('yolo-camera-capture').addEventListener('click', () => {
    const video = $('yolo-video');
    if (!video.videoWidth) return setStatus('yolo-message', 'Wait for the webcam image to appear.', true);
    const canvas = $('yolo-canvas');
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    canvas.getContext('2d').drawImage(video, 0, 0);
    canvas.toBlob(blob => {
      if (!blob) return setStatus('yolo-message', 'Could not capture a webcam frame.', true);
      selectedImage = new File([blob], 'webcam-snapshot.jpg', {type: 'image/jpeg'});
      $('yolo-detect').disabled = false;
      detectSelected();
    }, 'image/jpeg', 0.9);
  });

  $('yolo-detect-form').addEventListener('submit', event => {
    event.preventDefault();
    detectSelected();
  });
  async function detectSelected() {
    if (!selectedImage) return setStatus('yolo-message', 'Upload an image or capture a webcam frame first.', true);
    const body = new FormData();
    body.append('image', selectedImage);
    body.append('confidence', $('yolo-confidence').value);
    body.append('model_id', $('yolo-model').value);
    $('yolo-detect').disabled = true;
    $('yolo-camera-capture').disabled = true;
    setStatus('yolo-message', 'Running object detection...');
    try {
      const response = await fetch('/api/labs/6/detect', {method: 'POST', body});
      const result = await response.json();
      if (!response.ok) throw new Error(result.error || 'Detection failed.');
      $('yolo-annotated').src = result.annotated;
      $('yolo-download').href = result.annotated;
      $('yolo-summary').textContent = `${result.count} detection(s) / ${result.inference_ms} ms / ${result.width} x ${result.height} / ${result.model}`;
      $('yolo-detections').replaceChildren();
      for (const detection of result.detections) {
        const row = document.createElement('tr');
        row.append(element('td', `${detection.label} (${detection.class_id})`), element('td', `${(detection.confidence * 100).toFixed(1)}%`), element('td', detection.xyxy.join(', ')));
        $('yolo-detections').append(row);
      }
      if (!result.detections.length) {
        const row = document.createElement('tr');
        const cell = element('td', 'No objects passed the confidence threshold. Try lowering it.');
        cell.colSpan = 3; row.append(cell); $('yolo-detections').append(row);
      }
      $('yolo-results').hidden = false;
      setStatus('yolo-message', 'Detection complete. Capture another webcam frame or upload a different image.');
    } catch (error) { setStatus('yolo-message', error.message, true); }
    finally {
      $('yolo-detect').disabled = !selectedImage;
      $('yolo-camera-capture').disabled = !cameraStream;
    }
  }

  $('yolo-train-form').addEventListener('submit', async event => {
    event.preventDefault();
    const dataset = $('yolo-dataset').files[0];
    if (!dataset) return setStatus('yolo-train-message', 'Choose a dataset ZIP first.', true);
    if (dataset.size > 120 * 1024 * 1024) return setStatus('yolo-train-message', 'Dataset ZIP must be smaller than 120 MB.', true);
    $('yolo-train').disabled = true;
    $('yolo-evaluate').disabled = true;
    setStatus('yolo-train-message', 'Checking dataset and training the model. This can take a long time; keep Flask running...');
    const body = new FormData();
    body.append('dataset', dataset);
    body.append('epochs', $('yolo-epochs').value);
    body.append('image_size', $('yolo-image-size').value);
    try {
      const response = await fetch('/api/labs/6/train', {method: 'POST', body});
      const result = await response.json();
      if (!response.ok) throw new Error(result.error || 'Training failed.');
      trainedId = result.model_id;
      const option = element('option', `Custom model ${trainedId.slice(0, 8)}`);
      option.value = trainedId;
      $('yolo-model').append(option);
      $('yolo-model').value = trainedId;
      $('yolo-checkpoint').href = result.checkpoint_url;
      $('yolo-training-summary').textContent = `${result.message} Epochs: ${result.epochs}; image size: ${result.image_size}; classes: ${Object.values(result.classes).join(', ')}; image entries: ${JSON.stringify(result.images)}.`;
      showMetrics(result.metrics);
      $('yolo-training-results').hidden = false;
      $('yolo-evaluate').disabled = false;
      setStatus('yolo-train-message', 'Training completed. The custom checkpoint is available for inference above.');
    } catch (error) { setStatus('yolo-train-message', error.message, true); }
    finally { $('yolo-train').disabled = false; }
  });

  $('yolo-evaluate').addEventListener('click', async () => {
    if (!trainedId) return;
    $('yolo-evaluate').disabled = true;
    setStatus('yolo-train-message', 'Evaluating the trained model on the validation split...');
    try {
      const response = await fetch('/api/labs/6/evaluate', {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({model_id: trainedId})});
      const result = await response.json();
      if (!response.ok) throw new Error(result.error || 'Evaluation failed.');
      showMetrics(result.metrics);
      setStatus('yolo-train-message', 'Validation complete. Review precision, recall, and mAP metrics above.');
    } catch (error) { setStatus('yolo-train-message', error.message, true); }
    finally { $('yolo-evaluate').disabled = !trainedId; }
  });

  function showMetrics(metrics) {
    $('yolo-metrics').replaceChildren();
    const entries = Object.entries(metrics || {});
    if (!entries.length) entries.push(['Metrics', 'No scalar metrics were returned. See the Flask terminal output.']);
    for (const [key, value] of entries) {
      const card = element('div', '', 'property');
      card.append(element('small', key.replaceAll('_', ' ')), element('strong', typeof value === 'number' ? value.toFixed(4) : String(value)));
      $('yolo-metrics').append(card);
    }
  }
  capability();
})();
