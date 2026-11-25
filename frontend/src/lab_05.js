(() => {
  let busy = false;
  function status(text, error = false) {
    $('pca-message').textContent = text;
    $('pca-message').classList.toggle('error', error);
  }
  $('pca-run').addEventListener('click', async () => {
    if (busy) return;
    busy = true;
    $('pca-run').disabled = true;
    status('Fetching the LFW face dataset and fitting the PCA model. First run may take a few minutes...');
    try {
      const response = await fetch('/api/labs/5/run', {method: 'POST'});
      const result = await response.json();
      if (!response.ok) throw new Error(result.error || 'The PCA experiment could not complete.');
      render(result);
      status('Experiment complete. Results show the dataset and model run.');
    } catch (error) {
      status(error.message, true);
    } finally {
      busy = false;
      $('pca-run').disabled = false;
    }
  });
  function metric(value) { return Number(value).toFixed(3); }
  function render(data) {
    const d = data.dataset, p = data.pca, c = data.classifier;
    $('pca-summary').replaceChildren();
    const fields = [['Faces', d.samples.toLocaleString()], ['Raw features / face', d.features.toLocaleString()], ['Identities', d.classes], ['PCA features', p.components], ['Train / test', `${d.train_samples} / ${d.test_samples}`], ['Test accuracy', `${(c.accuracy * 100).toFixed(1)}%`], ['Misclassified', `${c.misclassified}`]];
    for (const [label, value] of fields) {
      const card = element('div', '', 'property');
      card.append(element('small', label), element('strong', String(value)));
      $('pca-summary').append(card);
    }
    $('pca-details').textContent = `LFW dataset: ${d.samples} faces, ${d.features} features per face (${d.height} x ${d.width}), ${d.classes} identities. The split uses 80% training and 20% test data with random_state=42. PCA uses ${p.components} randomized components, whiten=True, fitted only on training faces. SVM uses an RBF kernel and balanced class weights.`;
    $('pca-samples').src = data.plots.sample_faces;
    $('pca-variance').src = data.plots.variance;
    $('pca-eigenfaces').src = data.plots.eigenfaces;
    $('pca-variance-note').textContent = p.variance_90_components === null
      ? `The fitted ${p.components} components retain ${(p.variance_at_150 * 100).toFixed(1)}% of training variance; 90% is not reached within this fitted component set.`
      : `The first ${p.variance_90_components} components retain at least 90% of training variance. The fitted set retains ${(p.variance_at_150 * 100).toFixed(1)}%.`;
    $('pca-accuracy').textContent = `Overall accuracy: ${(c.accuracy * 100).toFixed(1)}% (${c.misclassified} errors of ${d.test_samples})`;
    const tbody = $('pca-report');
    tbody.replaceChildren();
    for (const name of d.class_names) {
      const row = document.createElement('tr');
      const th = element('th', name); th.scope = 'row'; row.append(th);
      const stats = c.report[name];
      row.append(element('td', metric(stats.precision)), element('td', metric(stats.recall)), element('td', metric(stats['f1-score'])), element('td', String(stats.support)));
      tbody.append(row);
    }
    for (const name of ['macro avg', 'weighted avg']) {
      const row = document.createElement('tr');
      const th = element('th', name); th.scope = 'row'; row.append(th);
      for (const key of ['precision', 'recall', 'f1-score']) row.append(element('td', metric(c.report[name][key])));
      row.append(element('td', String(c.report[name].support)));
      tbody.append(row);
    }
    $('pca-averages').textContent = `SVM accuracy: ${(c.accuracy * 100).toFixed(1)}%. Macro average: precision ${metric(c.report['macro avg'].precision)}, recall ${metric(c.report['macro avg'].recall)}, F1 ${metric(c.report['macro avg']['f1-score'])}. Weighted average: precision ${metric(c.report['weighted avg'].precision)}, recall ${metric(c.report['weighted avg'].recall)}, F1 ${metric(c.report['weighted avg']['f1-score'])}.`;
    $('pca-failures').hidden = !data.plots.failures;
    if (data.plots.failures) $('pca-failure-gallery').src = data.plots.failures;
    $('pca-empty').hidden = true;
    $('pca-results').hidden = false;
  }
})();
