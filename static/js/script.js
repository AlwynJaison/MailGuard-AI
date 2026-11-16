(() => {
  const area = document.querySelector('#message');
  const counter = document.querySelector('#counter');
  if (area) {
    const analyze = document.querySelector('#analyze');
    const clear = document.querySelector('#clear');
    const toast = document.querySelector('#toast');
    const update = () => { counter.textContent = `${area.value.length.toLocaleString()} / 20,000`; };
    area.addEventListener('input', update);
    clear.addEventListener('click', () => { area.value = ''; update(); toast.textContent = ''; area.focus(); });
    analyze.addEventListener('click', async () => {
      toast.textContent = '';
      if (!area.value.trim()) { toast.textContent = 'Enter a message before analyzing it.'; area.focus(); return; }
      analyze.disabled = true; analyze.classList.add('loading');
      try {
        const response = await fetch('/predict', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({message:area.value})});
        const data = await response.json();
        if (!response.ok) throw new Error(data.error || 'Analysis failed.');
        const content = document.querySelector('.result-content');
        document.querySelector('.result-empty').hidden = true; content.hidden = false;
        const spam = data.result === 'SPAM';
        const badge = document.querySelector('.result-badge');
        badge.textContent = spam ? 'SPAM WARNING' : 'LIKELY LEGITIMATE'; badge.className = `result-badge${spam ? ' spam' : ''}`;
        document.querySelector('.result-title').textContent = data.result;
        document.querySelector('.confidence strong').textContent = `${data.confidence}%`;
        const meter = document.querySelector('.meter i'); meter.style.width = `${data.confidence}%`; meter.className = spam ? 'spam' : '';
        document.querySelector('.result-status').textContent = data.status;
        document.querySelector('.explain p').textContent = data.explanation;
        document.querySelector('.preview p').textContent = data.preview;
      } catch (error) { toast.textContent = error.message; }
      finally { analyze.disabled = false; analyze.classList.remove('loading'); }
    });
  }
  const list = document.querySelector('#history-list');
  if (list) {
    const toast = document.querySelector('#toast');
    const load = async () => {
      const response = await fetch('/history'); const data = await response.json();
      if (!data.items.length) { list.innerHTML = '<div class="empty-row">No predictions saved yet. <a href="/">Analyze a message</a> to get started.</div>'; return; }
      list.innerHTML = data.items.map(item => {
        const date = new Date(item.created_at).toLocaleString();
        const klass = item.label === 'SPAM' ? 'spam' : '';
        return `<div class="history-row"><p title="${escapeHtml(item.preview)}">${escapeHtml(item.preview)}</p><span class="pill ${klass}">${item.label}</span><span>${Number(item.confidence*100).toFixed(1)}%</span><time>${date}</time></div>`;
      }).join('');
    };
    document.querySelector('#clear-history').addEventListener('click', async () => {
      if (!confirm('Clear all saved prediction history?')) return;
      const response = await fetch('/history', {method:'DELETE'});
      if (response.ok) { toast.textContent = 'History cleared.'; load(); } else toast.textContent = 'Could not clear history.';
    });
    load().catch(() => { list.innerHTML = '<div class="empty-row">Could not load history. Try refreshing the page.</div>'; });
  }
  function escapeHtml(value) { return String(value).replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c])); }
})();
