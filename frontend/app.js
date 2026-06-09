/* ═══════════════════════════════════════════════
   AI Data Analyst Pro — Frontend Application
   ═══════════════════════════════════════════════ */

// ── Global State ──
const STATE = {
    sessionId: null,
    selectedModel: null,
    models: [],
    filename: null,
    ws: null,
};

const API = '';  // Same origin

// ═══════════════════════════════════════════════
// Router — Page Navigation
// ═══════════════════════════════════════════════
function navigateTo(page) {
    document.querySelectorAll('.page-section').forEach(s => s.classList.add('hidden'));
    document.querySelectorAll('.nav-link').forEach(l => l.classList.remove('active'));
    const target = document.getElementById(`page-${page}`);
    if (target) target.classList.remove('hidden');
    const link = document.querySelector(`[data-page="${page}"]`);
    if (link) link.classList.add('active');

    // Lazy-load data when navigating
    if (page === 'memory') {
        loadMemoryPage();
    } else if (page === 'observability') {
        loadObservabilityMetrics();
    } else if (STATE.sessionId) {
        if (page === 'inspection') loadInspection();
        if (page === 'profiling') loadProfiling();
        if (page === 'workspace') loadWorkspaceTrace();
        if (page === 'autotrace') loadPersistedExecutionState();
    }
}

document.getElementById('nav-links').addEventListener('click', (e) => {
    const link = e.target.closest('.nav-link');
    if (!link) return;
    e.preventDefault();
    const page = link.dataset.page;
    if (!STATE.sessionId && page !== 'upload' && page !== 'memory' && page !== 'observability') {
        showToast('Upload a file first', 'error');
        return;
    }
    navigateTo(page);
});


// ═══════════════════════════════════════════════
// Toast Notification
// ═══════════════════════════════════════════════
function showToast(msg, type = 'info') {
    const toast = document.createElement('div');
    const colors = { info: 'bg-primary/20 border-primary/30 text-primary',
                     error: 'bg-error/20 border-error/30 text-error',
                     success: 'bg-tertiary/20 border-tertiary/30 text-tertiary' };
    toast.className = `fixed top-20 right-8 z-[200] px-5 py-3 rounded-xl border ${colors[type] || colors.info} text-sm font-medium backdrop-blur-xl shadow-2xl transition-all duration-500 translate-x-0`;
    toast.textContent = msg;
    document.body.appendChild(toast);
    setTimeout(() => { toast.style.opacity = '0'; setTimeout(() => toast.remove(), 500); }, 3000);
}


// ═══════════════════════════════════════════════
// Models — Load & Select
// ═══════════════════════════════════════════════
async function loadModels() {
    try {
        const resp = await fetch(`${API}/api/models`);
        const data = await resp.json();
        STATE.models = data.models || [];
        STATE.selectedModel = data.default;
        renderModelList();
        updateTopbarModel();
    } catch (e) {
        document.getElementById('model-list').innerHTML = `<div class="text-center text-error py-6">Failed to connect to Ollama. Make sure <code>ollama serve</code> is running.</div>`;
    }
}

function renderModelList() {
    const container = document.getElementById('model-list');
    const modalList = document.getElementById('modal-model-list');
    if (!STATE.models.length) {
        container.innerHTML = `<div class="text-center text-on-surface-variant py-6">No models available</div>`;
        return;
    }
    document.getElementById('model-count-badge').textContent = `${STATE.models.length} Models Available`;
    document.getElementById('model-count-display').textContent = STATE.models.length;

    const html = STATE.models.map((m, i) => {
        const isSelected = m.id === STATE.selectedModel;
        const icon = m.type === 'cloud' ? 'cloud' : 'laptop_mac';
        const emoji = m.type === 'cloud' ? '☁️' : '💻';
        const tierBadge = m.tier === 'frontier' ? `<span class="px-2 py-0.5 bg-primary/20 text-primary text-[10px] uppercase tracking-wider rounded font-label font-bold">Frontier</span>` :
                          m.tier === 'specialist' ? `<span class="px-2 py-0.5 bg-secondary/20 text-secondary text-[10px] uppercase tracking-wider rounded font-label font-bold">Specialist</span>` : '';
        const bgClass = isSelected ? 'bg-surface-container-highest border-primary/20' : 'bg-surface-container-low hover:bg-surface-container border-transparent hover:border-outline-variant/30';

        return `<div class="model-item ${bgClass} rounded-lg p-4 cursor-pointer flex items-center justify-between border transition-colors duration-200" data-model-id="${m.id}" onclick="selectModel('${m.id}')">
            <div class="flex items-center gap-4">
                <div class="w-10 h-10 bg-surface rounded flex items-center justify-center border border-outline-variant/20 text-slate-400">
                    <span class="material-symbols-outlined text-sm">${icon}</span>
                </div>
                <div>
                    <div class="flex items-center gap-2"><p class="text-sm font-headline font-semibold text-on-surface">${m.name}</p>${tierBadge}</div>
                    <p class="text-xs text-on-surface-variant font-body">${emoji} ${m.type === 'cloud' ? 'Cloud' : 'Local'} • ${m.id}</p>
                </div>
            </div>
            <div class="w-5 h-5 rounded-full border-2 ${isSelected ? 'border-primary' : 'border-outline-variant'} flex items-center justify-center">
                ${isSelected ? '<div class="w-2.5 h-2.5 bg-primary rounded-full"></div>' : ''}
            </div>
        </div>`;
    }).join('');

    container.innerHTML = html;
    if (modalList) modalList.innerHTML = html;
}

function selectModel(modelId) {
    STATE.selectedModel = modelId;
    renderModelList();
    updateTopbarModel();
    document.getElementById('model-modal').classList.add('hidden');
    showToast(`Model: ${modelId}`, 'success');
}

function updateTopbarModel() {
    const el = document.getElementById('topbar-model');
    const model = STATE.models.find(m => m.id === STATE.selectedModel);
    if (model) {
        el.textContent = `${model.name} (${model.type === 'cloud' ? 'Cloud' : 'Local'})`;
    }
}

document.getElementById('model-selector-top').addEventListener('click', () => {
    document.getElementById('model-modal').classList.remove('hidden');
});


// ═══════════════════════════════════════════════
// File Upload
// ═══════════════════════════════════════════════
const fileInput = document.getElementById('file-input');
const uploadZone = document.getElementById('upload-zone');

uploadZone.addEventListener('dragover', (e) => { e.preventDefault(); uploadZone.classList.add('border-primary'); });
uploadZone.addEventListener('dragleave', () => uploadZone.classList.remove('border-primary'));
uploadZone.addEventListener('drop', (e) => {
    e.preventDefault();
    uploadZone.classList.remove('border-primary');
    if (e.dataTransfer.files.length) uploadFile(e.dataTransfer.files[0]);
});
fileInput.addEventListener('change', (e) => { if (e.target.files.length) uploadFile(e.target.files[0]); });

async function uploadFile(file) {
    document.getElementById('upload-progress').classList.remove('hidden');
    const form = new FormData();
    form.append('file', file);

    try {
        const resp = await fetch(`${API}/api/upload`, { method: 'POST', body: form });
        const data = await resp.json();
        if (data.error) throw new Error(data.error);

        STATE.sessionId = data.session_id;
        STATE.filename = data.filename;
        document.getElementById('topbar-filename').textContent = data.filename;

        // Enable report button
        const reportBtn = document.getElementById('btn-download-report');
        reportBtn.disabled = false;
        reportBtn.classList.remove('opacity-50', 'cursor-not-allowed');

        // Populate ML target dropdown
        const allCols = [...(data.numeric_cols || []), ...(data.categorical_cols || [])];
        const mlTarget = document.getElementById('ml-target');
        mlTarget.innerHTML = allCols.map(c => `<option value="${c}">${c}</option>`).join('');

        showToast(`Loaded ${data.filename}: ${data.rows.toLocaleString()} rows × ${data.columns} columns`, 'success');
        loadRecentSessions();
        navigateTo('inspection');
    } catch (e) {
        showToast(`Upload failed: ${e.message}`, 'error');
    } finally {
        document.getElementById('upload-progress').classList.add('hidden');
    }
}


// ═══════════════════════════════════════════════
// Data Inspection
// ═══════════════════════════════════════════════
async function loadInspection() {
    if (!STATE.sessionId) return;
    try {
        const resp = await fetch(`${API}/api/inspect/${STATE.sessionId}`);
        const d = await resp.json();

        // Metrics
        const metrics = [
            { label: 'Rows', value: d.rows.toLocaleString(), icon: 'table_rows', color: 'primary', sub: `${d.numeric_count} Num, ${d.categorical_count} Cat` },
            { label: 'Columns', value: d.columns, icon: 'view_column', color: 'primary', sub: `${d.numeric_count} Num, ${d.categorical_count} Cat` },
            { label: 'Duplicates', value: `${d.duplicates_pct}%`, icon: 'file_copy', color: d.duplicates_pct > 5 ? 'error' : 'tertiary',
              sub: d.duplicates_pct < 1 ? '✅ Acceptable' : '⚠️ Review needed' },
            { label: 'Missing Cells', value: `${d.missing_pct}%`, icon: 'texture_minus', color: d.missing_pct > 5 ? 'error' : 'tertiary',
              sub: d.missing_pct < 2 ? '✅ Acceptable' : '⚠️ Requires attention' },
            { label: 'Memory Usage', value: `${d.memory_mb}`, icon: 'memory', color: 'primary', sub: 'MB in-memory', unit: 'MB' },
        ];

        document.getElementById('inspect-metrics').innerHTML = metrics.map(m => `
            <div class="metric-card bg-surface-container border border-outline-variant/15 rounded-xl p-5 flex flex-col justify-between">
                <div class="flex justify-between items-start mb-4">
                    <span class="text-on-surface-variant font-label text-sm font-medium">${m.label}</span>
                    <span class="material-symbols-outlined text-${m.color}/50 text-[20px]">${m.icon}</span>
                </div>
                <div>
                    <div class="font-headline text-[1.75rem] font-bold text-on-surface tracking-tight leading-none">${m.value}${m.unit ? `<span class="text-lg text-on-surface-variant ml-1">${m.unit}</span>` : ''}</div>
                    <div class="text-xs text-on-surface-variant mt-2">${m.sub}</div>
                </div>
            </div>
        `).join('');

        // Profile data (for numerical summary)
        const profResp = await fetch(`${API}/api/profile/${STATE.sessionId}`);
        const prof = await profResp.json();

        document.getElementById('num-summary-body').innerHTML = (prof.numerical || []).map(r => {
            const absSkew = Math.abs(r.skewness);
            const skewColor = absSkew > 2 ? 'error' : absSkew > 1 ? 'secondary' : 'tertiary';
            const skewWidth = Math.min(absSkew / 3 * 100, 100);
            return `<tr class="group">
                <td class="px-5 py-4 font-medium text-primary group-hover:text-primary-dim">${r.feature}</td>
                <td class="px-5 py-4 text-on-surface">${r.mean}</td>
                <td class="px-5 py-4 text-on-surface-variant">${r.std}</td>
                <td class="px-5 py-4 text-on-surface-variant">${r.min}</td>
                <td class="px-5 py-4 text-on-surface-variant">${r.max}</td>
                <td class="px-5 py-4 text-right"><span class="inline-flex items-center gap-2">${r.skewness}${absSkew > 2 ? ' <span class="text-error">!</span>' : ''}
                    <div class="w-8 h-1 bg-surface-container-highest rounded-full overflow-hidden"><div class="h-full bg-${skewColor} rounded-full" style="width:${skewWidth}%"></div></div></span></td>
            </tr>`;
        }).join('');

        // Health check bars
        document.getElementById('health-missing-pct').textContent = `${d.missing_pct}%`;
        document.getElementById('health-bars').innerHTML = (d.missing_columns || []).slice(0, 8).map(c => `
            <div>
                <div class="flex justify-between text-xs mb-1"><span class="text-on-surface font-medium">${c.column}</span><span class="text-error">${c.pct}%</span></div>
                <div class="w-full bg-surface-container-highest rounded-full h-1.5 overflow-hidden">
                    <div class="bg-error h-1.5 rounded-full" style="width:${Math.max(c.pct, 2)}%"></div>
                </div>
            </div>
        `).join('') || '<div class="text-tertiary text-sm">No missing values found! 🎉</div>';

        // Load Dataset Intelligence Layer data
        loadDatasetIntelligence();

    } catch (e) { showToast(`Inspection error: ${e.message}`, 'error'); }
}


// ═══════════════════════════════════════════════
// Statistical Profiling
// ═══════════════════════════════════════════════
async function loadProfiling() {
    if (!STATE.sessionId) return;
    try {
        const resp = await fetch(`${API}/api/profile/${STATE.sessionId}`);
        const d = await resp.json();

        document.getElementById('profile-num-body').innerHTML = (d.numerical || []).map(r =>
            `<tr><td class="px-5 py-4 font-medium text-primary">${r.feature}</td>
            <td class="px-5 py-4">${r.mean}</td><td class="px-5 py-4 text-on-surface-variant">${r.std}</td>
            <td class="px-5 py-4 text-on-surface-variant">${r.skewness}</td>
            <td class="px-5 py-4 text-on-surface-variant">${r.kurtosis}</td></tr>`).join('');

        document.getElementById('profile-cat-body').innerHTML = (d.categorical || []).map(r =>
            `<tr><td class="px-5 py-4 font-medium text-secondary">${r.column}</td>
            <td class="px-5 py-4">${r.unique}</td><td class="px-5 py-4 text-on-surface-variant">${r.top_value}</td>
            <td class="px-5 py-4">${r.top_freq.toLocaleString()}</td>
            <td class="px-5 py-4 ${r.missing > 0 ? 'text-error' : 'text-tertiary'}">${r.missing}</td></tr>`).join('');
    } catch (e) { showToast(`Profiling error: ${e.message}`, 'error'); }
}


// ═══════════════════════════════════════════════
// Feature Engineering & Advanced Stats
// ═══════════════════════════════════════════════
document.getElementById('btn-run-features').addEventListener('click', async () => {
    if (!STATE.sessionId) return showToast('Upload data first', 'error');
    const btn = document.getElementById('btn-run-features');
    btn.disabled = true;
    btn.innerHTML = '<span class="material-symbols-outlined animate-spin text-sm">progress_activity</span> Running...';

    try {
        // Features
        const fResp = await fetch(`${API}/api/features/${STATE.sessionId}`, { method: 'POST' });
        const fData = await fResp.json();

        // Stats
        const sResp = await fetch(`${API}/api/stats/${STATE.sessionId}`);
        const sData = await sResp.json();

        document.getElementById('analysis-status').textContent = 'AI INFERENCE COMPLETE';

        // Stats summary cards
        const cards = [];
        if (sData.pca) cards.push({ label: 'PCA Variance', value: `${(sData.pca.total_variance * 100).toFixed(1)}%`, sub: 'PC1 + PC2', color: 'secondary' });
        if (sData.normality && sData.normality.length) {
            const normalCount = sData.normality.filter(n => n.is_normal || n.Normal === '✅').length;
            cards.push({ label: 'Normality', value: `${normalCount}/${sData.normality.length}`, sub: 'normal features', color: normalCount > 0 ? 'tertiary' : 'error' });
        }
        if (fData.engineered_features) cards.push({ label: 'New Features', value: fData.engineered_features.length, sub: 'engineered', color: 'primary' });
        if (fData.skewed_features) cards.push({ label: 'Skewed', value: fData.skewed_features.length, sub: 'features >|1|', color: 'error' });

        document.getElementById('stats-cards').innerHTML = cards.map(c => `
            <div class="bg-surface-container-low rounded-xl p-4 ghost-border-top"><div class="text-xs text-on-surface-variant uppercase tracking-wider mb-2">${c.label}</div>
            <div class="text-2xl font-headline font-semibold text-${c.color}">${c.value}</div>
            <div class="text-xs text-on-surface-variant mt-1">${c.sub}</div></div>`).join('');

        // Normality table
        let detailsHtml = '';
        if (sData.normality && sData.normality.length) {
            detailsHtml += `<div class="bg-surface-container border border-outline-variant/15 rounded-xl overflow-hidden">
                <div class="p-5 border-b border-white/5 bg-surface-container-low"><h3 class="text-lg font-headline font-semibold text-on-surface">Normality Tests</h3></div>
                <div class="overflow-x-auto"><table class="w-full text-left text-sm"><thead class="text-xs text-on-surface-variant uppercase bg-surface-container-lowest/50">
                    <tr><th class="px-5 py-3">Feature</th><th class="px-5 py-3">Shapiro W</th><th class="px-5 py-3">p-value</th><th class="px-5 py-3">Normal?</th></tr></thead>
                    <tbody class="divide-y divide-white/5">${sData.normality.map(n =>
                        `<tr><td class="px-5 py-3 text-primary">${n.Column || n.column || ''}</td>
                        <td class="px-5 py-3">${(n['Shapiro W'] || n.shapiro_w || 0).toFixed ? (n['Shapiro W'] || n.shapiro_w || 0).toFixed(4) : n['Shapiro W'] || ''}</td>
                        <td class="px-5 py-3">${(n['p-value'] || n.p_value || 0).toFixed ? (n['p-value'] || n.p_value || 0).toFixed(6) : n['p-value'] || ''}</td>
                        <td class="px-5 py-3">${n.Normal || (n.is_normal ? '✅' : '❌') || ''}</td></tr>`
                    ).join('')}</tbody></table></div></div>`;
        }
        document.getElementById('stats-details').innerHTML = detailsHtml;
        showToast('Feature engineering & stats complete!', 'success');
    } catch (e) { showToast(`Analysis error: ${e.message}`, 'error'); }
    finally { btn.disabled = false; btn.innerHTML = '<span class="material-symbols-outlined text-sm">auto_fix_high</span>Run Feature Engineering & Stats'; }
});


// ═══════════════════════════════════════════════
// Anomaly Detection
// ═══════════════════════════════════════════════
async function runAnomaly(method) {
    if (!STATE.sessionId) return showToast('Upload data first', 'error');
    const contam = document.getElementById('anomaly-contamination').value / 100;
    const thresh = document.getElementById('anomaly-zscore').value / 10;

    try {
        const resp = await fetch(`${API}/api/anomaly/${STATE.sessionId}?method=${method}&contamination=${contam}&threshold=${thresh}`, { method: 'POST' });
        const d = await resp.json();
        const resultDiv = document.getElementById(`anomaly-${method === 'isolation_forest' ? 'if' : method === 'zscore' ? 'zs' : 'iqr'}-result`);

        if (d.anomaly_count !== undefined) {
            resultDiv.innerHTML = `<div class="text-2xl font-headline font-bold text-error">${d.anomaly_count}</div>
                <div class="text-xs text-on-surface-variant">anomalies detected (${d.anomaly_pct}%)</div>`;
        } else if (d.anomalous_count !== undefined) {
            resultDiv.innerHTML = `<div class="text-2xl font-headline font-bold text-error">${d.anomalous_count}</div>
                <div class="text-xs text-on-surface-variant">anomalous rows found</div>`;
        }
        showToast(`${d.method}: Detection complete`, 'success');
    } catch (e) { showToast(`Anomaly error: ${e.message}`, 'error'); }
}


// ═══════════════════════════════════════════════
// ML Modeling
// ═══════════════════════════════════════════════
document.getElementById('btn-train-ml').addEventListener('click', async () => {
    if (!STATE.sessionId) return showToast('Upload data first', 'error');
    const target = document.getElementById('ml-target').value;
    const model = document.getElementById('ml-model').value;
    const btn = document.getElementById('btn-train-ml');
    btn.disabled = true;
    btn.innerHTML = '<span class="material-symbols-outlined animate-spin text-sm">progress_activity</span> Training...';

    try {
        const resp = await fetch(`${API}/api/ml/train/${STATE.sessionId}?target=${target}&model_name=${encodeURIComponent(model)}`, { method: 'POST' });
        const d = await resp.json();
        if (d.error) throw new Error(d.error);

        let html = `<div class="space-y-6">`;
        html += `<div class="inline-flex items-center gap-2 px-3 py-1.5 bg-surface-container rounded border border-outline-variant/30 text-xs">
            <span class="w-2 h-2 rounded-full bg-tertiary"></span>${d.model_name} — ${d.task_type}</div>`;

        if (d.task_type === 'classification') {
            html += `<div class="grid grid-cols-2 gap-4">
                <div class="bg-surface-container-low p-4 rounded-lg border border-outline-variant/10">
                    <div class="text-[10px] text-on-surface-variant uppercase tracking-wider mb-1">Accuracy</div>
                    <div class="text-3xl font-headline font-bold text-primary">${(d.accuracy || 0).toFixed(2)}</div>
                </div>
                <div class="bg-surface-container-low p-4 rounded-lg border border-outline-variant/10">
                    <div class="text-[10px] text-on-surface-variant uppercase tracking-wider mb-1">F1-Score</div>
                    <div class="text-3xl font-headline font-bold text-secondary">${(d.f1_score || 0).toFixed(2)}</div>
                </div>
            </div>`;
        } else {
            html += `<div class="grid grid-cols-2 gap-4">
                <div class="bg-surface-container-low p-4 rounded-lg border border-outline-variant/10">
                    <div class="text-[10px] text-on-surface-variant uppercase tracking-wider mb-1">R²</div>
                    <div class="text-3xl font-headline font-bold text-primary">${(d.r2 || 0).toFixed(3)}</div>
                </div>
                <div class="bg-surface-container-low p-4 rounded-lg border border-outline-variant/10">
                    <div class="text-[10px] text-on-surface-variant uppercase tracking-wider mb-1">RMSE</div>
                    <div class="text-3xl font-headline font-bold text-secondary">${(d.rmse || 0).toFixed(2)}</div>
                </div>
            </div>`;
        }

        // CV Score
        if (d.cv_mean !== undefined) {
            html += `<div class="text-xs text-on-surface-variant">Cross-Validation: ${d.cv_mean.toFixed(3)} ± ${(d.cv_std || 0).toFixed(3)} (${d.cv_scores ? d.cv_scores.length : 5}-fold)</div>`;
        }

        // Feature importance
        if (d.feature_importance && d.feature_importance.length) {
            html += `<div><h4 class="text-sm font-medium text-on-surface mb-3">Feature Importance</h4><div class="space-y-3">`;
            const maxImp = Math.max(...d.feature_importance.map(f => f.Importance || f.importance || 0));
            d.feature_importance.slice(0, 8).forEach(f => {
                const imp = f.Importance || f.importance || 0;
                const width = maxImp > 0 ? (imp / maxImp * 100) : 0;
                html += `<div><div class="flex justify-between text-xs mb-1"><span class="text-on-surface">${f.Feature || f.feature}</span>
                    <span class="text-primary font-medium">${imp.toFixed(3)}</span></div>
                    <div class="feature-bar"><div class="feature-bar-fill" style="width:${width}%"></div></div></div>`;
            });
            html += `</div></div>`;
        }
        html += `</div>`;
        document.getElementById('ml-results').innerHTML = html;
        showToast('Model trained successfully!', 'success');
    } catch (e) { showToast(`ML error: ${e.message}`, 'error'); }
    finally { btn.disabled = false; btn.innerHTML = '<span class="material-symbols-outlined text-sm">model_training</span>Train Model'; }
});


// ═══════════════════════════════════════════════
// Code Lab
// ═══════════════════════════════════════════════
document.getElementById('btn-run-code').addEventListener('click', async () => {
    if (!STATE.sessionId) return showToast('Upload data first', 'error');
    const request = document.getElementById('codelab-input').value.trim();
    if (!request) return showToast('Enter a request', 'error');
    const btn = document.getElementById('btn-run-code');
    btn.disabled = true;
    btn.innerHTML = '<span class="material-symbols-outlined animate-spin text-sm">progress_activity</span> Generating...';

    try {
        const resp = await fetch(`${API}/api/code-lab/${STATE.sessionId}?request_text=${encodeURIComponent(request)}`, { method: 'POST' });
        const d = await resp.json();
        if (d.error) throw new Error(d.error);

        // Show code
        const codeDiv = document.getElementById('codelab-code');
        codeDiv.classList.remove('hidden');
        document.getElementById('codelab-code-content').textContent = d.code || '';

        // Show output
        let outputHtml = '';
        if (d.stdout) outputHtml += `<div class="bg-surface-container-lowest rounded-lg p-4 font-mono text-xs text-tertiary mb-4 border border-outline-variant/10"><pre>${d.stdout}</pre></div>`;
        if (d.figures && d.figures.length) {
            d.figures.forEach(b64 => { outputHtml += `<img src="data:image/png;base64,${b64}" class="rounded-lg border border-outline-variant/10 mb-4 w-full"/>` });
        }
        if (d.error) outputHtml += `<div class="bg-error/10 border border-error/20 rounded-lg p-4 text-xs text-error">${d.error}</div>`;
        if (!outputHtml) outputHtml = '<div class="text-on-surface-variant text-sm">Code executed successfully (no output).</div>';

        document.getElementById('codelab-output').innerHTML = outputHtml;
        showToast('Code executed!', 'success');
    } catch (e) { showToast(`Code Lab error: ${e.message}`, 'error'); }
    finally { btn.disabled = false; btn.innerHTML = '<span class="material-symbols-outlined text-sm">play_arrow</span>Generate & Execute'; }
});


// ═══════════════════════════════════════════════
// AI Chat (WebSocket Streaming)
// ═══════════════════════════════════════════════
function connectChat() {
    if (!STATE.sessionId) return;
    const proto = location.protocol === 'https:' ? 'wss' : 'ws';
    STATE.ws = new WebSocket(`${proto}://${location.host}/api/chat/${STATE.sessionId}`);
    STATE.ws.onclose = () => { STATE.ws = null; };
    STATE.ws.onmessage = (event) => {
        try {
            const data = JSON.parse(event.data);
            if (data.type === 'planner_event') {
                handlePlannerWebSocketEvent(data.event);
            }
        } catch (e) {
            console.error("Chat WebSocket error:", e);
        }
    };
}

function addChatMessage(role, content) {
    const container = document.getElementById('chat-messages');
    const div = document.createElement('div');
    div.className = `${role === 'user' ? 'chat-msg-user ml-12' : 'chat-msg-ai mr-12'} px-4 py-3 text-sm text-on-surface whitespace-pre-wrap`;
    div.textContent = content;
    container.appendChild(div);
    container.scrollTop = container.scrollHeight;
    return div;
}

document.getElementById('btn-send-chat').addEventListener('click', () => sendChat());
document.getElementById('chat-input').addEventListener('keydown', (e) => { if (e.key === 'Enter') sendChat(); });

function sendChat() {
    if (!STATE.sessionId) return showToast('Upload data first', 'error');
    const input = document.getElementById('chat-input');
    const question = input.value.trim();
    if (!question) return;
    input.value = '';

    addChatMessage('user', question);

    if (!STATE.ws || STATE.ws.readyState !== WebSocket.OPEN) connectChat();

    // Wait for connection then send
    const trySend = () => {
        if (STATE.ws.readyState === WebSocket.OPEN) {
            const mode = document.getElementById('toggle-agentic').checked ? 'agentic' : 'single';
            STATE.ws.send(JSON.stringify({ question, model: STATE.selectedModel, mode }));

            const aiDiv = addChatMessage('ai', '');
            let fullText = '';

            STATE.ws.onmessage = (event) => {
                const data = JSON.parse(event.data);
                if (data.type === 'planner_event') {
                    handlePlannerWebSocketEvent(data.event);
                } else if (data.type === 'token') {
                    fullText += data.content;
                    aiDiv.textContent = fullText;
                    aiDiv.parentElement.scrollTop = aiDiv.parentElement.scrollHeight;
                } else if (data.type === 'step') {
                    const stepLabel = data.step.charAt(0).toUpperCase() + data.step.slice(1);
                    if (data.status === 'running') {
                        aiDiv.innerHTML += `<div class="chat-msg-step p-3 my-2"><div class="flex items-center gap-2 text-xs text-primary"><span class="material-symbols-outlined animate-spin text-[14px]">progress_activity</span>${stepLabel}...</div></div>`;
                    } else if (data.status === 'done' && typeof data.data === 'string') {
                        fullText = data.data;
                    }
                    aiDiv.parentElement.scrollTop = aiDiv.parentElement.scrollHeight;
                } else if (data.type === 'done') {
                    aiDiv.textContent = data.content;
                    aiDiv.parentElement.scrollTop = aiDiv.parentElement.scrollHeight;
                } else if (data.type === 'error') {
                    aiDiv.innerHTML = `<span class="text-error">${data.content}</span>`;
                }
            };
        } else {
            setTimeout(trySend, 200);
        }
    };
    setTimeout(trySend, 300);
}


// ═══════════════════════════════════════════════
// AI Summary
// ═══════════════════════════════════════════════
document.getElementById('btn-gen-summary').addEventListener('click', async () => {
    if (!STATE.sessionId) return showToast('Upload data first', 'error');
    const btn = document.getElementById('btn-gen-summary');
    btn.textContent = 'Generating...';
    btn.disabled = true;
    document.getElementById('summary-content').innerHTML = '<div class="flex items-center gap-2 text-primary"><span class="material-symbols-outlined animate-spin text-sm">progress_activity</span>Running full agent pipeline...</div>';

    try {
        const resp = await fetch(`${API}/api/analyze/${STATE.sessionId}`, { method: 'POST' });
        const data = await resp.json();
        
        const insight = data.insight || {};
        if (!insight.success) {
            throw new Error(insight.error || "Analysis failed");
        }
        
        const report = insight.result || {};
        
        let html = `
            <div class="space-y-4">
                <div class="text-sm font-semibold text-on-surface mb-2">Executive Summary</div>
                <div class="text-xs text-on-surface-variant leading-relaxed mb-4">${report.executive_summary || 'No summary available.'}</div>
        `;
        
        if (report.top_findings && report.top_findings.length > 0) {
            html += `
                <div class="text-sm font-semibold text-on-surface mt-4 mb-2">Top Business Findings</div>
                <div class="space-y-2">
                    ${report.top_findings.map(f => `
                        <div class="bg-surface-container-high rounded-lg p-3 border border-outline-variant/10 text-xs">
                            <div class="flex justify-between items-center mb-1">
                                <span class="font-bold text-primary">Finding #${f.rank}</span>
                                <span class="text-[9px] px-2 py-0.5 rounded border border-secondary/20 bg-secondary/10 text-secondary uppercase font-semibold">${f.confidence} confidence</span>
                            </div>
                            <div class="text-on-surface mb-1">${f.finding}</div>
                            <div class="text-on-surface-variant font-medium mt-1">Impact: <span class="text-on-surface">${f.business_impact}</span></div>
                        </div>
                    `).join('')}
                </div>
            `;
        }
        
        if (report.recommendations && report.recommendations.length > 0) {
            html += `
                <div class="text-sm font-semibold text-on-surface mt-4 mb-2">Actionable Recommendations</div>
                <div class="space-y-2">
                    ${report.recommendations.map(r => `
                        <div class="bg-surface-container-high rounded-lg p-3 border border-outline-variant/10 text-xs">
                            <div class="font-bold text-tertiary mb-1">Priority #${r.priority}: ${r.action}</div>
                            <div class="text-on-surface-variant mb-1">${r.rationale}</div>
                            <div class="text-[10px] text-on-surface-variant mt-1">Implementation: <span class="text-on-surface">${r.implementation}</span></div>
                        </div>
                    `).join('')}
                </div>
            `;
        }
        
        html += `</div>`;
        document.getElementById('summary-content').innerHTML = html;
        showToast('Full pipeline analysis complete!', 'success');
        
        // Refresh Workspace Trace UI
        loadWorkspaceTrace();
    } catch (e) { 
        document.getElementById('summary-content').innerHTML = `<div class="text-error text-sm">Failed to generate summary: ${e.message}</div>`; 
    } finally { 
        btn.textContent = 'Generate'; 
        btn.disabled = false; 
    }
});


// ═══════════════════════════════════════════════
// Report Download
// ═══════════════════════════════════════════════
document.getElementById('btn-download-report').addEventListener('click', () => {
    if (!STATE.sessionId) return;
    window.open(`${API}/api/report/${STATE.sessionId}`, '_blank');
});


// ═══════════════════════════════════════════════
// Report Download
// ═══════════════════════════════════════════════
document.getElementById('btn-download-report').addEventListener('click', () => {
    if (!STATE.sessionId) return;
    window.open(`${API}/api/report/${STATE.sessionId}`, '_blank');
});


// ═══════════════════════════════════════════════
// Phase 1: Recent Sessions
// ═══════════════════════════════════════════════
async function loadRecentSessions() {
    try {
        const resp = await fetch(`${API}/api/sessions`);
        const data = await resp.json();
        const sessions = data.sessions || [];
        const container = document.getElementById('recent-datasets-section');
        const listContainer = document.getElementById('recent-sessions-list');
        
        if (sessions.length > 0) {
            container.classList.remove('hidden');
            listContainer.innerHTML = sessions.map(s => {
                const date = new Date(s.created_at).toLocaleString();
                const scoreClass = s.quality_score >= 90 ? 'text-tertiary' : s.quality_score >= 70 ? 'text-secondary' : 'text-error';
                return `<div class="bg-surface-container-low hover:bg-surface-container border border-outline-variant/15 hover:border-primary/30 rounded-xl p-5 transition-all duration-200 cursor-pointer flex flex-col justify-between" onclick="loadSession('${s.sid}', '${s.filename}')">
                    <div class="flex justify-between items-start mb-3">
                        <div class="flex items-center gap-3">
                            <span class="material-symbols-outlined text-primary text-xl">table_chart</span>
                            <div class="font-headline font-semibold text-on-surface truncate max-w-[150px]">${s.filename}</div>
                        </div>
                        <button onclick="event.stopPropagation(); deleteSession('${s.sid}')" class="text-on-surface-variant hover:text-error transition-colors">
                            <span class="material-symbols-outlined text-[16px]">delete</span>
                        </button>
                    </div>
                    <div class="text-xs text-on-surface-variant space-y-1">
                        <div>Size: <span class="text-on-surface font-medium">${s.row_count.toLocaleString()} rows × ${s.col_count} cols</span></div>
                        <div>Quality: <span class="${scoreClass} font-bold">${s.quality_grade || '-'} (${Math.round(s.quality_score) || 0})</span></div>
                        <div class="text-[10px] text-on-surface-variant/70 mt-2">Uploaded: ${date}</div>
                    </div>
                </div>`;
            }).join('');
        } else {
            container.classList.add('hidden');
        }
    } catch (e) {
        console.error("Failed to load recent sessions", e);
    }
}

async function loadSession(sid, filename) {
    STATE.sessionId = sid;
    STATE.filename = filename;
    document.getElementById('topbar-filename').textContent = filename;
    
    // Enable report button
    const reportBtn = document.getElementById('btn-download-report');
    reportBtn.disabled = false;
    reportBtn.classList.remove('opacity-50', 'cursor-not-allowed');
    
    // Populate ML target dropdown
    try {
        const resp = await fetch(`${API}/api/inspect/${sid}`);
        const d = await resp.json();
        const allCols = d.dtypes ? d.dtypes.map(dt => dt.column) : [];
        const mlTarget = document.getElementById('ml-target');
        if (mlTarget) mlTarget.innerHTML = allCols.map(c => `<option value="${c}">${c}</option>`).join('');
    } catch (e) {
        console.error("Failed to fetch column dropdowns", e);
    }
    
    showToast(`Loaded dataset: ${filename}`, 'success');
    navigateTo('inspection');
}

async function deleteSession(sid) {
    if (!confirm("Are you sure you want to delete this session?")) return;
    try {
        const resp = await fetch(`${API}/api/sessions/${sid}`, { method: 'DELETE' });
        const data = await resp.json();
        if (data.error) throw new Error(data.error);
        showToast("Session deleted", "success");
        if (STATE.sessionId === sid) {
            STATE.sessionId = null;
            STATE.filename = null;
            document.getElementById('topbar-filename').textContent = "No file loaded";
            const reportBtn = document.getElementById('btn-download-report');
            reportBtn.disabled = true;
            reportBtn.classList.add('opacity-50', 'cursor-not-allowed');
        }
        loadRecentSessions();
    } catch (e) {
        showToast(`Failed to delete session: ${e.message}`, 'error');
    }
}


// ═══════════════════════════════════════════════
// Phase 1: Dataset Intelligence & Quality
// ═══════════════════════════════════════════════
async function loadDatasetIntelligence() {
    if (!STATE.sessionId) return;
    try {
        const resp = await fetch(`${API}/api/intelligence/${STATE.sessionId}`);
        const intel = await resp.json();
        
        document.getElementById('intel-dataset-type').textContent = intel.dataset_type || 'Unknown';
        document.getElementById('intel-domain').textContent = intel.domain || 'Unknown';
        
        const colsContainer = document.getElementById('intel-columns-list');
        const cols = intel.columns || {};
        colsContainer.innerHTML = Object.keys(cols).map(cName => {
            const cInfo = cols[cName];
            const badgeColors = {
                identifier: 'bg-primary/10 text-primary border-primary/20',
                binary_target: 'bg-secondary/10 text-secondary border-secondary/20',
                target: 'bg-secondary/10 text-secondary border-secondary/20',
                numeric: 'bg-primary/10 text-primary border-primary/20',
                categorical: 'bg-secondary/10 text-secondary border-secondary/20',
                date: 'bg-tertiary/10 text-tertiary border-tertiary/20',
                text: 'bg-surface-variant text-on-surface-variant border-outline-variant/30'
            };
            const badgeClass = badgeColors[cInfo.semantic_type] || 'bg-surface-variant text-on-surface border-outline-variant/20';
            return `<div class="bg-surface-container-low border border-white/5 rounded-lg p-3 flex items-center justify-between">
                <div>
                    <div class="flex items-center gap-2">
                        <span class="text-xs font-semibold text-on-surface">${cName}</span>
                        <span class="text-[10px] px-2 py-0.5 rounded-full border ${badgeClass}">${cInfo.semantic_type}</span>
                    </div>
                    <div class="text-[10px] text-on-surface-variant mt-1">${cInfo.description || 'No description available'}</div>
                </div>
                <div class="text-right text-[10px] text-on-surface-variant">
                    <div>Unique: <span class="text-on-surface font-medium">${cInfo.nunique}</span></div>
                    <div>Missing: <span class="${cInfo.missing_pct > 0 ? 'text-error' : 'text-tertiary'}">${cInfo.missing_pct}%</span></div>
                </div>
            </div>`;
        }).join('');

        const hypContainer = document.getElementById('intel-hypotheses-list');
        const hypotheses = intel.hypotheses || [];
        if (hypotheses.length > 0) {
            hypContainer.innerHTML = hypotheses.map(h => `<li>${h}</li>`).join('');
        } else {
            hypContainer.innerHTML = `<li class="text-on-surface-variant/50 italic">No hypotheses generated</li>`;
        }

        try {
            await fetch(`${API}/api/schema/${STATE.sessionId}`);
        } catch (err) {
            console.error("Schema Agent execution failed", err);
        }

        loadQualityDiagnostics();

    } catch (e) {
        console.error("Failed to load dataset intelligence", e);
    }
}

async function loadQualityDiagnostics() {
    if (!STATE.sessionId) return;
    
    const box = document.getElementById('quality-report-box');
    box.innerHTML = `<div class="text-center text-on-surface-variant py-8"><span class="material-symbols-outlined animate-spin text-sm">progress_activity</span> Running quality diagnostics...</div>`;
    
    try {
        const resp = await fetch(`${API}/api/quality/${STATE.sessionId}`);
        const data = await resp.json();
        
        if (!data.success) {
            box.innerHTML = `<div class="text-error text-xs p-4 bg-error/10 border border-error/20 rounded-lg">Error: ${data.error}</div>`;
            return;
        }

        const q = data.result || {};
        const gradeBadge = document.getElementById('quality-grade-badge');
        
        const assessment = q.overall_assessment || 'good';
        const badgeColors = {
            poor: 'bg-error/20 text-error border-error/30',
            needs_attention: 'bg-secondary/20 text-secondary border-secondary/30',
            good: 'bg-primary/20 text-primary border-primary/30',
            excellent: 'bg-tertiary/20 text-tertiary border-tertiary/30'
        };
        gradeBadge.className = `px-2.5 py-1 rounded text-xs font-bold font-headline border ${badgeColors[assessment] || badgeColors.good}`;
        gradeBadge.textContent = assessment.toUpperCase().replace('_', ' ');

        let html = '';
        if (q.narrative) {
            html += `<div class="bg-surface-container-low border border-white/5 rounded-lg p-4 mb-4 text-xs text-on-surface-variant leading-relaxed whitespace-pre-wrap">${q.narrative}</div>`;
        }
        
        const issues = q.issues || [];
        if (issues.length > 0) {
            html += `<div class="space-y-3">`;
            issues.forEach((issue, index) => {
                const sevColors = {
                    critical: 'bg-error/10 text-error border-error/20',
                    warning: 'bg-secondary/10 text-secondary border-secondary/20',
                    info: 'bg-primary/10 text-primary border-primary/20'
                };
                const sevClass = sevColors[issue.severity] || 'bg-surface-variant text-on-surface border-outline-variant/20';
                
                html += `<div class="bg-surface-container-low border border-white/5 rounded-lg p-4">
                    <div class="flex justify-between items-start gap-3">
                        <div>
                            <span class="text-xs font-semibold text-on-surface">${issue.column || 'Dataset'}</span>
                            <span class="text-[10px] uppercase font-bold tracking-wider px-2 py-0.5 rounded border ml-2 ${sevClass}">${issue.severity}</span>
                        </div>
                        <div class="text-[10px] text-on-surface-variant font-medium">${issue.issue}</div>
                    </div>
                    <div class="text-xs text-on-surface-variant mt-2">${issue.description}</div>`;
                
                if (issue.fix_code) {
                    const codeId = `quality-fix-code-${index}`;
                    html += `<div class="mt-3 bg-surface-container-lowest border border-outline-variant/10 rounded p-2.5 font-mono text-[10px] text-slate-300 relative group">
                        <div class="text-[8px] text-on-surface-variant uppercase tracking-widest mb-1.5 font-sans">RECOMMENDED PYTHON FIX:</div>
                        <pre id="${codeId}" class="overflow-x-auto whitespace-pre-wrap">${issue.fix_code}</pre>
                        <button onclick="navigator.clipboard.writeText(document.getElementById('${codeId}').innerText); showToast('Code copied!', 'success')" class="absolute top-2 right-2 text-primary hover:underline font-sans text-[9px] opacity-0 group-hover:opacity-100 transition-opacity">Copy</button>
                    </div>`;
                }
                
                html += `</div>`;
            });
            html += `</div>`;
        } else {
            html += `<div class="text-center text-tertiary text-xs py-6">✅ No quality issues detected in dataset!</div>`;
        }
        box.innerHTML = html;
        loadInspectionTrace();

    } catch (e) {
        box.innerHTML = `<div class="text-error text-xs p-4 bg-error/10 border border-error/20 rounded-lg flex items-center justify-between">
            <span>Failed to run quality diagnostics: ${e.message}</span>
            <button onclick="loadQualityDiagnostics()" class="text-primary hover:underline font-semibold font-body text-[10px]">Retry</button>
        </div>`;
    }
}


// ═══════════════════════════════════════════════
// Phase 1: NL to SQL Analytics
// ═══════════════════════════════════════════════
async function runSqlQuery() {
    if (!STATE.sessionId) return showToast("Upload data first", "error");
    const input = document.getElementById('sql-question-input');
    const question = input.value.trim();
    if (!question) return showToast("Enter a question", "error");
    
    const btn = document.getElementById('btn-run-sql');
    btn.disabled = true;
    btn.innerHTML = '<span class="material-symbols-outlined animate-spin text-sm">progress_activity</span> Running...';
    
    const queryDisplay = document.getElementById('sql-query-display');
    const resultsContainer = document.getElementById('sql-results-container');
    
    try {
        const resp = await fetch(`${API}/api/sql/${STATE.sessionId}`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ question })
        });
        const d = await resp.json();
        
        if (d.error) throw new Error(d.error);
        if (!d.success) throw new Error(d.error || "Execution failed");
        
        queryDisplay.classList.remove('hidden');
        document.getElementById('sql-query-code').textContent = d.sql || '';
        document.getElementById('sql-query-explanation').innerHTML = `<strong>Explanation:</strong> ${d.explanation || ''}`;
        
        resultsContainer.classList.remove('hidden');
        const head = document.getElementById('sql-results-head');
        const body = document.getElementById('sql-results-body');
        
        const columns = d.columns || [];
        const rows = d.data || [];
        
        if (columns.length > 0 && rows.length > 0) {
            head.innerHTML = `<tr>${columns.map(c => `<th class="px-4 py-3 font-semibold">${c}</th>`).join('')}</tr>`;
            body.innerHTML = rows.map(r => `<tr>${columns.map(c => {
                const val = r[c];
                return `<td class="px-4 py-2.5 text-on-surface-variant max-w-[200px] truncate" title="${val}">${val !== null ? val : '<span class="text-outline-variant">NULL</span>'}</td>`;
            }).join('')}</tr>`).join('');
        } else {
            head.innerHTML = '';
            body.innerHTML = `<tr><td class="px-4 py-6 text-center text-on-surface-variant" colspan="100%">Query executed successfully. Empty result set returned.</td></tr>`;
        }
        
        showToast("SQL Query executed!", "success");
        
    } catch (e) {
        showToast(`SQL error: ${e.message}`, 'error');
        queryDisplay.classList.add('hidden');
        resultsContainer.classList.add('hidden');
    } finally {
        btn.disabled = false;
        btn.innerHTML = '<span class="material-symbols-outlined text-sm">play_arrow</span>Run';
    }
}

document.getElementById('btn-run-sql').addEventListener('click', runSqlQuery);
document.getElementById('sql-question-input').addEventListener('keydown', (e) => {
    if (e.key === 'Enter') runSqlQuery();
});


// ═══════════════════════════════════════════════
// Phase 1.5: Demo Datasets & Leaderboard & Trace
// ═══════════════════════════════════════════════

async function loadLeaderboard() {
    try {
        const resp = await fetch(`${API}/api/leaderboard`);
        const data = await resp.json();
        
        document.getElementById('metric-intel').textContent = `${data.dataset_intelligence_accuracy}%`;
        document.getElementById('metric-target').textContent = `${data.target_detection_accuracy}%`;
        document.getElementById('metric-semantic').textContent = `${data.semantic_typing_accuracy}%`;
        document.getElementById('metric-nlsql').textContent = `${data.nl_sql_accuracy}%`;
        document.getElementById('metric-sqlexec').textContent = `${data.sql_execution_success}%`;
    } catch (e) {
        console.error("Failed to load leaderboard", e);
    }
}

async function loadDemoDatasets() {
    const grid = document.getElementById('demo-datasets-grid');
    if (!grid) return;
    try {
        const resp = await fetch(`${API}/api/demo/list`);
        const data = await resp.json();
        
        grid.innerHTML = (data.demos || []).map(demo => {
            let icon = 'table_chart';
            if (demo.name === 'titanic') icon = 'directions_boat';
            if (demo.name === 'customer_churn') icon = 'group_remove';
            if (demo.name === 'sales_transactions') icon = 'receipt_long';
            if (demo.name === 'credit_risk') icon = 'credit_score';
            if (demo.name === 'employee_attrition') icon = 'groups';
            if (demo.name === 'patient_treatment') icon = 'medical_services';
            
            return `
                <div class="bg-surface-container-low hover:bg-surface-container border border-outline-variant/10 hover:border-primary/30 rounded-xl p-5 cursor-pointer transition-all duration-300 flex flex-col justify-between group h-44" onclick="loadDemoDataset('${demo.name}')">
                    <div class="flex justify-between items-start mb-3">
                        <div class="w-10 h-10 rounded-lg bg-surface-container-high border border-outline-variant/20 flex items-center justify-center text-slate-400 group-hover:text-primary transition-colors">
                            <span class="material-symbols-outlined text-[22px]">${icon}</span>
                        </div>
                        <span class="text-[9px] uppercase font-bold tracking-wider px-2 py-0.5 rounded border border-primary/20 bg-primary/10 text-primary">Demo</span>
                    </div>
                    <div>
                        <h4 class="text-sm font-headline font-bold text-on-surface group-hover:text-primary transition-colors">${demo.label}</h4>
                        <p class="text-xs text-on-surface-variant font-body mt-1 leading-normal">${demo.description}</p>
                    </div>
                    <div class="text-[10px] text-on-surface-variant mt-2 font-medium">${demo.task}</div>
                </div>
            `;
        }).join('');
    } catch (e) {
        console.error("Failed to load demo datasets list", e);
        grid.innerHTML = `<div class="col-span-full text-center text-on-surface-variant text-xs py-4">Failed to load demo datasets.</div>`;
    }
}

async function loadDemoDataset(name) {
    document.getElementById('upload-progress').classList.remove('hidden');
    try {
        const resp = await fetch(`${API}/api/demo/${name}`, { method: 'POST' });
        const data = await resp.json();
        if (data.error) throw new Error(data.error);
        
        STATE.sessionId = data.session_id;
        STATE.filename = data.filename;
        document.getElementById('topbar-filename').textContent = data.filename;
        
        // Enable report button
        const reportBtn = document.getElementById('btn-download-report');
        reportBtn.disabled = false;
        reportBtn.classList.remove('opacity-50', 'cursor-not-allowed');
        
        // Populate ML target dropdown
        const allCols = [...(data.numeric_cols || []), ...(data.categorical_cols || [])];
        const mlTarget = document.getElementById('ml-target');
        if (mlTarget) mlTarget.innerHTML = allCols.map(c => `<option value="${c}">${c}</option>`).join('');
        
        showToast(`Loaded ${data.filename}: ${data.rows.toLocaleString()} rows × ${data.columns} cols`, 'success');
        loadRecentSessions();
        navigateTo('inspection');
    } catch (e) {
        showToast(`Demo load failed: ${e.message}`, 'error');
    } finally {
        document.getElementById('upload-progress').classList.add('hidden');
    }
}

async function loadInspectionTrace() {
    if (!STATE.sessionId) return;
    const container = document.getElementById('inspection-trace-container');
    const stepsDiv = document.getElementById('inspection-trace-steps');
    if (!container || !stepsDiv) return;

    try {
        const resp = await fetch(`${API}/api/agent-results/${STATE.sessionId}`);
        const results = await resp.json();
        
        const agentsToTrace = ['schema', 'quality'];
        const traces = agentsToTrace.map(name => results[name]).filter(Boolean);
        
        if (traces.length > 0) {
            container.classList.remove('hidden');
            
            const totalDuration = traces.reduce((acc, t) => acc + (t.duration_sec || 0), 0);
            document.getElementById('inspection-total-time').textContent = `${totalDuration.toFixed(1)} seconds`;
            
            stepsDiv.innerHTML = traces.map(t => {
                const agentLabel = t.agent.charAt(0).toUpperCase() + t.agent.slice(1) + ' Agent';
                
                let cleanModel = t.model || 'Unknown Model';
                if (cleanModel.includes('deepseek-v4-pro:cloud') || cleanModel.toLowerCase().includes('deepseek')) cleanModel = 'DeepSeek V4';
                else if (cleanModel.includes('gemma4:31b-cloud') || cleanModel.toLowerCase().includes('gemma')) cleanModel = 'Gemma4 31B';
                else if (cleanModel.includes('qwen3.5:cloud') || cleanModel.toLowerCase().includes('qwen')) cleanModel = 'Qwen3.5';
                else if (cleanModel.includes('mistral-nemo') || cleanModel.toLowerCase().includes('mistral')) cleanModel = 'Mistral Nemo';

                const badgeClass = t.cached ? 'bg-tertiary/10 text-tertiary border-tertiary/20' : 'bg-primary/10 text-primary border-primary/20';
                const badgeLabel = t.cached ? 'CACHED' : 'LIVE';

                const pct = totalDuration > 0 ? ((t.duration_sec || 0) / totalDuration * 100) : 0;

                return `
                    <div class="flex-1 bg-surface-container-low border border-outline-variant/10 rounded-xl p-5 flex flex-col justify-between min-w-[240px]">
                        <div>
                            <div class="flex justify-between items-start mb-2">
                                <span class="text-sm font-bold text-on-surface font-headline">${agentLabel}</span>
                                <span class="text-[9px] uppercase font-bold px-2 py-0.5 rounded border ${badgeClass}">${badgeLabel}</span>
                            </div>
                            <div class="text-xs text-on-surface-variant font-body mb-2">${cleanModel}</div>
                        </div>
                        <div class="mt-4">
                            <div class="flex justify-between text-xs mb-1">
                                <span class="text-on-surface-variant font-medium">Duration</span>
                                <span class="font-bold text-tertiary">${(t.duration_sec || 0).toFixed(1)}s</span>
                            </div>
                            <div class="w-full bg-surface-container-highest rounded-full h-1 overflow-hidden">
                                <div class="bg-tertiary h-1 rounded-full" style="width: ${pct}%"></div>
                            </div>
                        </div>
                    </div>
                `;
            }).join('');
        } else {
            container.classList.add('hidden');
        }
    } catch (e) {
        console.error("Failed to load inspection trace", e);
    }
}

async function loadWorkspaceTrace() {
    if (!STATE.sessionId) return;
    const container = document.getElementById('workspace-trace-container');
    const stepsDiv = document.getElementById('workspace-trace-steps');
    if (!container || !stepsDiv) return;

    try {
        const resp = await fetch(`${API}/api/agent-results/${STATE.sessionId}`);
        const results = await resp.json();
        
        const agentsToTrace = ['schema', 'quality', 'hypothesis', 'eda', 'insight'];
        const traces = agentsToTrace.map(name => results[name]).filter(Boolean);
        
        if (traces.length > 0) {
            container.classList.remove('hidden');
            
            const totalDuration = traces.reduce((acc, t) => acc + (t.duration_sec || 0), 0);
            document.getElementById('workspace-total-time').textContent = `${totalDuration.toFixed(1)} seconds`;
            
            stepsDiv.innerHTML = traces.map(t => {
                const agentLabel = t.agent.charAt(0).toUpperCase() + t.agent.slice(1) + ' Agent';
                
                let cleanModel = t.model || 'Unknown Model';
                if (cleanModel.includes('deepseek-v4-pro:cloud') || cleanModel.toLowerCase().includes('deepseek')) cleanModel = 'DeepSeek V4';
                else if (cleanModel.includes('gemma4:31b-cloud') || cleanModel.toLowerCase().includes('gemma')) cleanModel = 'Gemma4 31B';
                else if (cleanModel.includes('qwen3.5:cloud') || cleanModel.toLowerCase().includes('qwen')) cleanModel = 'Qwen3.5';
                else if (cleanModel.includes('mistral-nemo') || cleanModel.toLowerCase().includes('mistral')) cleanModel = 'Mistral Nemo';
                else if (cleanModel.toLowerCase().includes('kimi')) cleanModel = 'Kimi K2.6';

                const badgeClass = t.cached ? 'bg-tertiary/10 text-tertiary border-tertiary/20' : 'bg-primary/10 text-primary border-primary/20';
                const badgeLabel = t.cached ? 'CACHED' : 'LIVE';

                const pct = totalDuration > 0 ? ((t.duration_sec || 0) / totalDuration * 100) : 0;

                return `
                    <div class="bg-surface-container-low border border-outline-variant/10 rounded-xl p-4 flex flex-col justify-between">
                        <div class="flex justify-between items-start mb-2">
                            <div>
                                <div class="text-xs font-bold text-on-surface font-headline">${agentLabel}</div>
                                <div class="text-[10px] text-on-surface-variant font-body mt-0.5">${cleanModel}</div>
                            </div>
                            <span class="text-[9px] uppercase font-bold px-2 py-0.5 rounded border ${badgeClass}">${badgeLabel}</span>
                        </div>
                        <div class="mt-2">
                            <div class="flex justify-between text-[10px] mb-1">
                                <span class="text-on-surface-variant">Duration</span>
                                <span class="font-bold text-tertiary">${(t.duration_sec || 0).toFixed(1)}s</span>
                            </div>
                            <div class="w-full bg-surface-container-highest rounded-full h-1 overflow-hidden">
                                <div class="bg-tertiary h-1 rounded-full" style="width: ${pct}%"></div>
                            </div>
                        </div>
                    </div>
                `;
            }).join('');
        } else {
            container.classList.add('hidden');
        }
    } catch (e) {
        console.error("Failed to load workspace trace", e);
    }
}


// ═══════════════════════════════════════════════
// Phase 2: Planner & Autonomous execution
// ═══════════════════════════════════════════════
STATE.currentPlan = null;

async function loadPersistedExecutionState() {
    if (!STATE.sessionId) return;
    try {
        const resp = await fetch(`${API}/api/sessions/${STATE.sessionId}/execution-state`);
        const data = await resp.json();
        
        if (data.plan && data.state) {
            STATE.currentPlan = data.plan;
            showPlanUI(data.plan);
            
            const statusBadge = document.getElementById('autotrace-status-badge');
            const state = data.state;
            
            statusBadge.textContent = state.status.charAt(0).toUpperCase() + state.status.slice(1);
            statusBadge.className = `px-2.5 py-1 rounded-full text-[10px] font-bold uppercase tracking-wider ${
                state.status === 'completed' ? 'bg-primary/10 text-primary border-primary/20' :
                state.status === 'failed' ? 'bg-red-500/10 text-red-400 border-red-500/20' :
                state.status === 'executing' ? 'bg-secondary/10 text-secondary border-secondary/20 animate-pulse' :
                'bg-surface-container-high text-on-surface-variant'
            } border`;

            renderTraceNodes(state.trace, state.outputs);
            
            const insightCard = document.getElementById('autotrace-insight-card');
            const reportContent = document.getElementById('autotrace-report-content');
            
            let finalInsight = null;
            if (state.outputs && state.outputs.recovery_insight) {
                finalInsight = state.outputs.recovery_insight;
            } else {
                const insightStep = state.trace.find(t => t.agent === 'insight' && t.status === 'completed');
                if (insightStep && state.outputs[insightStep.step_id]) {
                    finalInsight = state.outputs[insightStep.step_id];
                }
            }
            
            if (finalInsight) {
                insightCard.classList.remove('hidden');
                reportContent.innerHTML = formatMarkdownReport(finalInsight);
            } else {
                insightCard.classList.add('hidden');
            }
        } else {
            resetAutotraceUI();
        }
    } catch (e) {
        console.error("Failed loading persisted execution state", e);
    }
}

function resetAutotraceUI() {
    STATE.currentPlan = null;
    document.getElementById('planner-query').value = '';
    document.getElementById('planner-overall-estimates').classList.add('hidden');
    document.getElementById('planner-steps-card').classList.add('hidden');
    document.getElementById('autotrace-insight-card').classList.add('hidden');
    document.getElementById('autotrace-status-badge').textContent = 'Idle';
    document.getElementById('autotrace-status-badge').className = 'px-2.5 py-1 rounded-full text-[10px] font-bold uppercase tracking-wider bg-surface-container-high text-on-surface-variant border border-outline-variant/10';
    document.getElementById('autotrace-nodes').innerHTML = `
        <div class="flex flex-col justify-center items-center text-center p-8 space-y-4">
            <div class="w-16 h-16 rounded-full bg-surface-container-high flex items-center justify-center border border-outline-variant/20 text-on-surface-variant animate-pulse">
                <span class="material-symbols-outlined text-3xl">route</span>
            </div>
            <div>
                <h4 class="text-sm font-headline font-bold text-on-surface">No plan executing</h4>
                <p class="text-xs text-on-surface-variant font-body max-w-xs mt-1">Formulate a query and generate a plan to launch the autonomous agent team.</p>
            </div>
        </div>
    `;
}

function showPlanUI(plan) {
    document.getElementById('planner-rationale').textContent = plan.rationale || '';
    
    const est = plan.estimates || {};
    document.getElementById('planner-est-steps').textContent = est.estimated_steps || plan.steps.length;
    document.getElementById('planner-est-time').textContent = `${(est.estimated_time_sec || plan.steps.length * 2.5).toFixed(1)}s`;
    document.getElementById('planner-est-calls').textContent = est.estimated_llm_calls || plan.steps.length;
    document.getElementById('planner-overall-estimates').classList.remove('hidden');

    const list = document.getElementById('planner-steps-list');
    list.innerHTML = plan.steps.map(s => {
        const deps = s.depends_on && s.depends_on.length > 0 ? ` (depends: ${s.depends_on.join(', ')})` : '';
        return `
            <div class="flex items-start gap-2 bg-surface-container-low border border-outline-variant/10 rounded-lg p-2.5">
                <span class="material-symbols-outlined text-[16px] text-primary mt-0.5">account_tree</span>
                <div>
                    <div class="text-xs font-bold text-on-surface uppercase">${s.agent}</div>
                    <div class="text-[10px] text-on-surface-variant font-body">${s.description || ''}${deps}</div>
                </div>
            </div>
        `;
    }).join('');

    document.getElementById('planner-steps-card').classList.remove('hidden');
}

async function generatePlan() {
    if (!STATE.sessionId) return showToast('Upload dataset first', 'error');
    const query = document.getElementById('planner-query').value.trim();
    const btn = document.getElementById('btn-generate-plan');
    
    btn.disabled = true;
    btn.innerHTML = `<span class="material-symbols-outlined text-sm animate-spin">sync</span>Generating Plan...`;
    
    try {
        const resp = await fetch(`${API}/api/sessions/${STATE.sessionId}/plan`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ question: query })
        });
        
        const data = await resp.json();
        if (!resp.ok) throw new Error(data.error || 'Failed generating plan');
        
        STATE.currentPlan = data.plan;
        showPlanUI(data.plan);
        showToast('Execution plan created & validated!', 'success');
        
        const statusBadge = document.getElementById('autotrace-status-badge');
        statusBadge.textContent = 'Planned';
        statusBadge.className = 'px-2.5 py-1 rounded-full text-[10px] font-bold uppercase tracking-wider bg-primary/10 text-primary border border-primary/20';

        const traceSteps = data.plan.steps.map(s => ({
            step_id: s.id,
            agent: s.agent,
            description: s.description,
            status: 'pending',
            duration_sec: 0,
            model: ''
        }));
        renderTraceNodes(traceSteps, {});
        document.getElementById('autotrace-insight-card').classList.add('hidden');
        
    } catch (e) {
        showToast(e.message, 'error');
    } finally {
        btn.disabled = false;
        btn.innerHTML = `<span class="material-symbols-outlined text-sm">route</span>Generate Execution Plan`;
    }
}

async function executePlan() {
    if (!STATE.currentPlan) return showToast('Generate a plan first', 'error');
    const query = document.getElementById('planner-query').value.trim();
    const btn = document.getElementById('btn-execute-plan');
    
    btn.disabled = true;
    btn.innerHTML = `<span class="material-symbols-outlined text-sm animate-spin">autorenew</span>Executing...`;
    
    if (!STATE.ws || STATE.ws.readyState !== WebSocket.OPEN) {
        connectChat();
    }
    
    const statusBadge = document.getElementById('autotrace-status-badge');
    statusBadge.textContent = 'Executing';
    statusBadge.className = 'px-2.5 py-1 rounded-full text-[10px] font-bold uppercase tracking-wider bg-secondary/10 text-secondary border border-secondary/20 animate-pulse';

    try {
        const resp = await fetch(`${API}/api/sessions/${STATE.sessionId}/execute`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ plan: STATE.currentPlan, question: query })
        });
        
        const data = await resp.json();
        if (!resp.ok) throw new Error(data.error || 'Execution failed');
        
        statusBadge.textContent = data.status.charAt(0).toUpperCase() + data.status.slice(1);
        statusBadge.className = `px-2.5 py-1 rounded-full text-[10px] font-bold uppercase tracking-wider ${
            data.status === 'completed' ? 'bg-primary/10 text-primary border-primary/20' :
            'bg-red-500/10 text-red-400 border-red-500/20'
        } border`;

        renderTraceNodes(data.trace, data.outputs);
        showToast(data.status === 'completed' ? 'Execution completed!' : 'Execution failed', data.status === 'completed' ? 'success' : 'error');

        const insightCard = document.getElementById('autotrace-insight-card');
        const reportContent = document.getElementById('autotrace-report-content');
        
        let finalInsight = null;
        if (data.outputs && data.outputs.recovery_insight) {
            finalInsight = data.outputs.recovery_insight;
        } else {
            const insightStep = data.trace.find(t => t.agent === 'insight' && t.status === 'completed');
            if (insightStep && data.outputs[insightStep.step_id]) {
                finalInsight = data.outputs[insightStep.step_id];
            }
        }
        
        if (finalInsight) {
            insightCard.classList.remove('hidden');
            reportContent.innerHTML = formatMarkdownReport(finalInsight);
        } else {
            insightCard.classList.add('hidden');
        }

    } catch (e) {
        showToast(e.message, 'error');
        statusBadge.textContent = 'Failed';
        statusBadge.className = 'px-2.5 py-1 rounded-full text-[10px] font-bold uppercase tracking-wider bg-red-500/10 text-red-400 border border-red-500/20';
    } finally {
        btn.disabled = false;
        btn.innerHTML = `<span class="material-symbols-outlined text-sm">play_arrow</span>Execute Agent Network`;
    }
}

function handlePlannerWebSocketEvent(event) {
    const stepId = event.step_id;
    const node = document.getElementById(`trace-node-${stepId}`);
    if (!node) return;
    
    const dot = node.querySelector('.trace-status-dot');
    const label = node.querySelector('.trace-status-label');
    const dur = node.querySelector('.trace-step-duration');

    if (event.event === 'step_started') {
        node.classList.add('border-secondary/30', 'bg-surface-container-high/50');
        dot.className = 'trace-status-dot w-8 h-8 rounded-full bg-secondary/15 flex items-center justify-center text-secondary border border-secondary/35 animate-spin';
        dot.innerHTML = `<span class="material-symbols-outlined text-base">sync</span>`;
        label.textContent = 'RUNNING';
        label.className = 'trace-status-label text-[9px] font-bold tracking-wider px-2 py-0.5 rounded bg-secondary/10 text-secondary border border-secondary/20';
    } else if (event.event === 'step_completed') {
        node.classList.remove('border-secondary/30', 'bg-surface-container-high/50');
        node.classList.add('border-primary/20');
        dot.className = 'trace-status-dot w-8 h-8 rounded-full bg-primary/10 flex items-center justify-center text-primary border border-primary/30';
        dot.innerHTML = `<span class="material-symbols-outlined text-base">check</span>`;
        label.textContent = 'COMPLETED';
        label.className = 'trace-status-label text-[9px] font-bold tracking-wider px-2 py-0.5 rounded bg-primary/10 text-primary border border-primary/20';
        if (event.duration) dur.textContent = `${event.duration.toFixed(1)}s`;
    } else if (event.event === 'step_failed') {
        node.classList.remove('border-secondary/30', 'bg-surface-container-high/50');
        node.classList.add('border-red-500/20');
        dot.className = 'trace-status-dot w-8 h-8 rounded-full bg-red-500/10 flex items-center justify-center text-red-400 border border-red-500/30';
        dot.innerHTML = `<span class="material-symbols-outlined text-base">close</span>`;
        label.textContent = 'FAILED';
        label.className = 'trace-status-label text-[9px] font-bold tracking-wider px-2 py-0.5 rounded bg-red-500/10 text-red-400 border border-red-500/20';
        if (event.duration) dur.textContent = `${event.duration.toFixed(1)}s`;
    }
}

function renderTraceNodes(trace, outputs) {
    const container = document.getElementById('autotrace-nodes');
    if (!trace || trace.length === 0) return;
    
    container.innerHTML = trace.map(t => {
        const stepId = t.step_id;
        const agentName = t.agent;
        const desc = t.description || '';
        const status = t.status || 'pending';
        const duration = t.duration_sec ? `${t.duration_sec.toFixed(1)}s` : '—';
        
        let icon = 'table_chart';
        if (agentName === 'schema') icon = 'schema';
        else if (agentName === 'quality') icon = 'fact_check';
        else if (agentName === 'hypothesis') icon = 'science';
        else if (agentName === 'eda') icon = 'monitoring';
        else if (agentName === 'insight') icon = 'psychology';
        
        let statusBadge = '';
        let dotIcon = 'hourglass_empty';
        let dotClass = 'bg-surface-container text-slate-500 border-outline-variant/30';
        let cardClass = 'bg-surface-container-low border-outline-variant/10';
        let spinnerClass = '';
        
        if (status === 'completed') {
            statusBadge = `<span class="trace-status-label text-[9px] font-bold tracking-wider px-2 py-0.5 rounded bg-primary/10 text-primary border border-primary/20">COMPLETED</span>`;
            dotIcon = 'check';
            dotClass = 'bg-primary/10 text-primary border-primary/30';
            cardClass = 'border-primary/20 bg-surface-container-low';
        } else if (status === 'failed') {
            statusBadge = `<span class="trace-status-label text-[9px] font-bold tracking-wider px-2 py-0.5 rounded bg-red-500/10 text-red-400 border border-red-500/20">FAILED</span>`;
            dotIcon = 'close';
            dotClass = 'bg-red-500/10 text-red-400 border-red-500/30';
            cardClass = 'border-red-500/20 bg-surface-container-low';
        } else if (status === 'running') {
            statusBadge = `<span class="trace-status-label text-[9px] font-bold tracking-wider px-2 py-0.5 rounded bg-secondary/10 text-secondary border border-secondary/20">RUNNING</span>`;
            dotIcon = 'sync';
            dotClass = 'bg-secondary/15 text-secondary border-secondary/35';
            spinnerClass = 'animate-spin';
            cardClass = 'border-secondary/30 bg-surface-container-high/50';
        } else {
            statusBadge = `<span class="trace-status-label text-[9px] font-bold tracking-wider px-2 py-0.5 rounded bg-surface-container text-on-surface-variant border border-outline-variant/20">PENDING</span>`;
        }
        
        const out = outputs && outputs[stepId];
        const detailButton = out ? `
            <button onclick="toggleTraceNodeInspector('${stepId}')" class="text-xs text-primary hover:underline flex items-center gap-1 font-body mt-2">
                <span class="material-symbols-outlined text-xs">search</span>Inspect Results
            </button>
        ` : '';

        const inspectorSection = out ? `
            <div id="inspector-${stepId}" class="hidden mt-4 pt-4 border-t border-outline-variant/10 text-xs text-on-surface-variant font-body space-y-2 max-h-[300px] overflow-y-auto custom-scrollbar">
                <div class="font-bold text-on-surface uppercase text-[10px] tracking-wider mb-1">Evidence & Context</div>
                <ul class="list-disc pl-4 space-y-1">
                    ${(out.evidence || []).map(e => `<li>${e}</li>`).join('') || '<li>No explicit evidence markers returned</li>'}
                </ul>
                <div class="mt-3 font-bold text-on-surface uppercase text-[10px] tracking-wider mb-1">Model Reasoning</div>
                <div class="bg-surface-container-high border border-outline-variant/10 p-2.5 rounded-lg font-mono whitespace-pre-wrap leading-relaxed text-[11px]">${
                    out.error ? `Error: ${out.error}` : JSON.stringify(out.result, null, 2)
                }</div>
            </div>
        ` : '';

        return `
            <div id="trace-node-${stepId}" class="bg-surface-container-low border ${cardClass} rounded-xl p-4 flex flex-col transition-all duration-300">
                <div class="flex items-center justify-between w-full">
                    <div class="flex items-center gap-3">
                        <div class="trace-status-dot w-8 h-8 rounded-full ${dotClass} flex items-center justify-center ${spinnerClass}">
                            <span class="material-symbols-outlined text-base">${dotIcon}</span>
                        </div>
                        <div>
                            <div class="flex items-center gap-2">
                                <h4 class="text-xs font-bold text-on-surface uppercase tracking-wider flex items-center gap-1">
                                    <span class="material-symbols-outlined text-sm text-slate-400">${icon}</span>
                                    ${agentName} step
                                </h4>
                                ${statusBadge}
                            </div>
                            <p class="text-xs text-on-surface-variant font-body mt-0.5">${desc}</p>
                        </div>
                    </div>
                    <div class="text-right flex flex-col items-end">
                        <div class="trace-step-duration text-[10px] font-bold text-on-surface-variant font-mono">${duration}</div>
                        ${t.model ? `<div class="text-[8px] text-on-surface-variant font-mono bg-surface-container px-1 py-0.5 rounded mt-1">${t.model.split(':')[0]}</div>` : ''}
                    </div>
                </div>
                ${detailButton}
                ${inspectorSection}
            </div>
        `;
    }).join('');
}

function toggleTraceNodeInspector(stepId) {
    const inspector = document.getElementById(`inspector-${stepId}`);
    if (inspector) {
        inspector.classList.toggle('hidden');
    }
}

function formatMarkdownReport(insightOutput) {
    if (!insightOutput || !insightOutput.result) return 'No report returned.';
    
    if (insightOutput.markdown_report) {
        return parseSimpleMarkdown(insightOutput.markdown_report);
    }
    
    const result = insightOutput.result;
    let md = `<h4>${result.summary || 'Summary'}</h4>`;
    
    if (result.insights && result.insights.length > 0) {
        md += `<div class="font-bold text-on-surface mt-4 mb-2 uppercase text-xs tracking-wider">Key Insights</div>`;
        md += result.insights.map(i => `
            <div class="bg-surface-container-high border border-outline-variant/15 p-3 rounded-lg mb-2">
                <strong class="text-primary text-xs font-headline block">${i.title}</strong>
                <p class="text-xs text-on-surface-variant font-body mt-1">${i.text}</p>
            </div>
        `).join('');
    }
    
    if (result.actionable_recommendations && result.actionable_recommendations.length > 0) {
        md += `<div class="font-bold text-on-surface mt-4 mb-2 uppercase text-xs tracking-wider">Actionable Recommendations</div>`;
        md += `<ul class="list-disc pl-5 space-y-1.5 text-xs text-on-surface-variant font-body">`;
        md += result.actionable_recommendations.map(r => `<li>${r}</li>`).join('');
        md += `</ul>`;
    }
    
    return md;
}

function parseSimpleMarkdown(md) {
    return md
        .replace(/### (.*)/g, '<h4 class="text-sm font-bold text-on-surface mt-4 mb-2 uppercase">$1</h4>')
        .replace(/#### (.*)/g, '<h5 class="text-xs font-bold text-primary mt-3 mb-1 uppercase">$1</h5>')
        .replace(/\*\*([^*]+)\*\*/g, '<strong class="text-on-surface font-semibold">$1</strong>')
        .replace(/^- (.*)/gm, '<li class="ml-4 list-disc text-xs text-on-surface-variant font-body my-1">$1</li>')
        .replace(/\n\n/g, '<br/>');
}


// ═══════════════════════════════════════════════
// Phase 3 — Memory & Knowledge Layer handlers
// ═══════════════════════════════════════════════

async function loadMemoryPage() {
    // 1. Fetch leaderboard
    try {
        const resp = await fetch(`${API}/api/memory/leaderboard`);
        const metrics = await resp.json();
        document.getElementById('memory-metric-p1').textContent = `${Math.round(metrics.precision_at_1 * 100)}%`;
        document.getElementById('memory-metric-p3').textContent = `${Math.round(metrics.precision_at_3 * 100)}%`;
        document.getElementById('memory-metric-mrr').textContent = `${Math.round(metrics.mrr * 100)}%`;
        document.getElementById('memory-metric-latency').textContent = `${metrics.retrieval_latency_ms}ms`;
        document.getElementById('memory-metric-contamination').textContent = metrics.contamination_failures || 0;
    } catch (e) {
        console.error("Failed to load memory leaderboard", e);
    }

    // 2. Fetch datasets and render lineage
    try {
        const resp = await fetch(`${API}/api/memory/datasets`);
        const data = await resp.json();
        const tbody = document.getElementById('memory-lineage-tbody');
        if (!data.datasets || data.datasets.length === 0) {
            tbody.innerHTML = `<tr><td colspan="6" class="py-8 text-center text-on-surface-variant/50">No registered datasets found. Try uploading a dataset first.</td></tr>`;
            return;
        }

        tbody.innerHTML = data.datasets.map(d => {
            const formattedDate = d.created_at ? new Date(d.created_at).toLocaleString() : 'N/A';
            const linksText = d.links && d.links.length > 0
                ? d.links.map(l => `<span class="px-2 py-0.5 rounded bg-primary/10 text-primary text-[10px] uppercase font-mono mr-1">${l.relationship_type} (${Math.round(l.similarity_score * 100)}%)</span>`).join('')
                : '<span class="text-on-surface-variant/40">—</span>';
                
            return `<tr class="border-b border-outline-variant/10 hover:bg-surface-container-high/50 font-body text-xs">
                <td class="py-3 px-4 font-semibold text-on-surface">${d.filename}</td>
                <td class="py-3 px-4 text-center"><span class="px-2 py-0.5 rounded bg-secondary/15 text-secondary font-bold">v${d.version}</span></td>
                <td class="py-3 px-4 font-mono text-[10px] text-on-surface-variant">${d.fingerprint ? d.fingerprint.slice(0, 12) + '...' : 'N/A'}</td>
                <td class="py-3 px-4 text-on-surface-variant">${d.row_count} × ${d.col_count}</td>
                <td class="py-3 px-4">${linksText}</td>
                <td class="py-3 px-4 text-on-surface-variant">${formattedDate}</td>
            </tr>`;
        }).join('');
    } catch (e) {
        console.error("Failed to load datasets lineage", e);
    }
}

async function searchMemory() {
    const query = document.getElementById('memory-search-input').value.trim();
    if (!query) {
        showToast('Please enter a search query', 'error');
        return;
    }

    const container = document.getElementById('memory-search-results');
    container.innerHTML = `<div class="text-center text-on-surface-variant py-8 font-body text-xs flex items-center justify-center gap-2"><span class="material-symbols-outlined animate-spin text-sm">progress_activity</span> Running semantic query...</div>`;

    try {
        const resp = await fetch(`${API}/api/memory/search?q=${encodeURIComponent(query)}`);
        const data = await resp.json();
        
        if (!data.results || data.results.length === 0) {
            container.innerHTML = `<div class="text-center text-on-surface-variant/50 py-8 font-body text-xs">No similar datasets or findings matching your query.</div>`;
            return;
        }

        container.innerHTML = data.results.map(r => {
            const reasonsText = r.reasons.map(reason => `<span class="px-2 py-0.5 rounded bg-primary/10 text-primary text-[10px] font-medium mr-1 border border-primary/20">${reason}</span>`).join('');
            
            // Build sub-memories HTML
            let memoriesHtml = '';
            
            const insights = r.memories?.insights || [];
            if (insights.length > 0) {
                memoriesHtml += `<div class="mt-4"><h5 class="text-[10px] uppercase font-bold text-on-surface mb-2 tracking-wide flex items-center gap-1"><span class="material-symbols-outlined text-xs text-primary">analytics</span> Retrieved Insights</h5>`;
                memoriesHtml += insights.map(ins => `
                    <div class="bg-surface-container-low border border-outline-variant/10 rounded-xl p-3 mb-2">
                        <strong class="text-primary text-[11px] font-headline block">${ins.title}</strong>
                        <p class="text-xs text-on-surface-variant font-body mt-1">${ins.content}</p>
                    </div>
                `).join('');
                memoriesHtml += `</div>`;
            }

            const hypotheses = r.memories?.hypotheses || [];
            if (hypotheses.length > 0) {
                memoriesHtml += `<div class="mt-4"><h5 class="text-[10px] uppercase font-bold text-on-surface mb-2 tracking-wide flex items-center gap-1"><span class="material-symbols-outlined text-xs text-secondary">psychology</span> Formulated Hypotheses</h5>`;
                memoriesHtml += hypotheses.map(hyp => {
                    const statusColor = hyp.status === 'validated' ? 'text-tertiary bg-tertiary/10 border-tertiary/20' : 'text-on-surface-variant bg-surface-container-low border-outline-variant/10';
                    return `
                        <div class="bg-surface-container-low border border-outline-variant/10 rounded-xl p-3 mb-2 flex items-start justify-between gap-3">
                            <div>
                                <p class="text-xs text-on-surface font-body">${hyp.hypothesis}</p>
                                <span class="text-[10px] text-on-surface-variant mt-1 block">Evidence Score: ${Math.round(hyp.evidence_score * 100)}%</span>
                            </div>
                            <span class="px-2 py-0.5 rounded text-[9px] uppercase font-bold border ${statusColor}">${hyp.status}</span>
                        </div>
                    `;
                }).join('');
                memoriesHtml += `</div>`;
            }

            const cleanups = r.memories?.cleaning_actions || [];
            if (cleanups.length > 0) {
                memoriesHtml += `<div class="mt-4"><h5 class="text-[10px] uppercase font-bold text-on-surface mb-2 tracking-wide flex items-center gap-1"><span class="material-symbols-outlined text-xs text-tertiary">mop</span> Applied Cleanups & Transforms</h5>`;
                memoriesHtml += cleanups.map(cl => `
                    <div class="bg-surface-container-low border border-outline-variant/10 rounded-xl p-3 mb-2">
                        <div class="flex justify-between items-center mb-1">
                            <strong class="text-tertiary text-[11px] font-headline font-semibold">Column: ${cl.column}</strong>
                            <span class="px-1.5 py-0.5 rounded bg-tertiary/10 text-tertiary text-[9px] uppercase font-bold font-mono">${cl.action_type}</span>
                        </div>
                        <p class="text-xs text-on-surface-variant font-body mt-1">${cl.description}</p>
                    </div>
                `).join('');
                memoriesHtml += `</div>`;
            }

            return `<div class="bg-surface-container-high border border-outline-variant/15 rounded-2xl p-5 shadow-lg mb-4 hover:border-primary/30 transition-all">
                <div class="flex justify-between items-start border-b border-outline-variant/10 pb-3 mb-4">
                    <div>
                        <div class="flex items-center gap-2">
                            <h4 class="font-headline font-bold text-on-surface text-sm">${r.dataset}</h4>
                            <span class="px-1.5 py-0.5 rounded bg-secondary/15 text-secondary text-[10px] font-bold">v${r.version}</span>
                        </div>
                        <div class="flex items-center mt-2 flex-wrap gap-1">
                            ${reasonsText}
                        </div>
                    </div>
                    <div class="text-right">
                        <div class="text-[10px] text-on-surface-variant uppercase tracking-wider font-semibold">Semantic Match</div>
                        <div class="text-lg font-bold text-primary">${Math.round(r.similarity * 100)}%</div>
                    </div>
                </div>
                <div class="text-xs text-on-surface-variant/90 leading-relaxed italic bg-surface-container-low/50 p-3 rounded-xl border border-outline-variant/5">
                    ${r.summary}
                </div>
                ${memoriesHtml}
            </div>`;
        }).join('');

    } catch (e) {
        console.error("Failed to run memory search query", e);
        container.innerHTML = `<div class="text-center text-error py-8 font-body text-xs">Failed to fetch memory search results.</div>`;
    }
}


// ═══════════════════════════════════════════════
// Init
// ═══════════════════════════════════════════════
document.addEventListener('DOMContentLoaded', () => {
    loadModels();
    loadRecentSessions();
    loadDemoDatasets();
    loadLeaderboard();
    
    // Bind autotrace actions
    document.getElementById('btn-generate-plan').addEventListener('click', generatePlan);
    document.getElementById('btn-execute-plan').addEventListener('click', executePlan);

    // Bind memory actions
    const btnMemSearch = document.getElementById('btn-memory-search');
    const inputMemSearch = document.getElementById('memory-search-input');
    if (btnMemSearch && inputMemSearch) {
        btnMemSearch.addEventListener('click', searchMemory);
        inputMemSearch.addEventListener('keypress', (e) => {
            if (e.key === 'Enter') searchMemory();
        });
    }

    // Bind observability refresh
    const btnRefreshObs = document.getElementById('btn-refresh-metrics');
    if (btnRefreshObs) {
        btnRefreshObs.addEventListener('click', loadObservabilityMetrics);
    }

    navigateTo('upload');
});


// ═══════════════════════════════════════════════
// Observability & Analytics Dashboard
// ═══════════════════════════════════════════════
async function loadObservabilityMetrics() {
    try {
        const resp = await fetch(`${API}/api/observability/metrics`);
        const data = await resp.json();
        if (data.error) throw new Error(data.error);
        renderObservabilityMetrics(data);
    } catch (e) {
        showToast(`Failed loading observability metrics: ${e.message}`, 'error');
    }
}

function renderObservabilityMetrics(data) {
    // Populate KPIs
    document.getElementById('obs-kpi-success').textContent = data.total_runs > 0 ? `${data.overall_success_rate}%` : '0.0%';
    document.getElementById('obs-kpi-speedup').textContent = `${data.parallel_speedup}%`;
    document.getElementById('obs-kpi-memhit').textContent = `${data.memory_hit_rate}%`;
    document.getElementById('obs-kpi-restores').textContent = data.checkpoint_restores;

    // Agent Diagnostics table
    const agentTable = document.getElementById('obs-agent-metrics-table');
    if (data.agent_metrics && data.agent_metrics.length > 0) {
        agentTable.innerHTML = data.agent_metrics.map(m => `
            <tr class="hover:bg-white/5 transition-colors">
                <td class="py-3 px-4 font-semibold text-on-surface">${m.agent}</td>
                <td class="py-3 px-4 font-mono">${m.runs}</td>
                <td class="py-3 px-4">
                    <span class="px-2 py-0.5 rounded text-[10px] font-bold ${m.success_rate >= 95 ? 'bg-emerald-500/10 text-emerald-400' : 'bg-amber-500/10 text-amber-400'}">
                        ${m.success_rate}%
                    </span>
                </td>
                <td class="py-3 px-4 font-mono text-slate-300">${m.avg_latency}s</td>
                <td class="py-3 px-4 font-mono text-slate-400">${m.retries}</td>
            </tr>
        `).join('');
    } else {
        agentTable.innerHTML = `<tr><td colspan="5" class="py-8 text-center text-on-surface-variant">No agent execution history found. Run analysis sessions to populate metrics.</td></tr>`;
    }

    // Recent executions log table
    const recentTable = document.getElementById('obs-recent-executions-table');
    if (data.recent_executions && data.recent_executions.length > 0) {
        recentTable.innerHTML = data.recent_executions.map(ex => {
            let statusPill = '';
            if (ex.status === 'completed') {
                statusPill = `<span class="px-2 py-0.5 rounded text-[10px] bg-emerald-500/10 text-emerald-400 font-bold font-mono">COMPLETED</span>`;
            } else if (ex.status === 'failed') {
                statusPill = `<span class="px-2 py-0.5 rounded text-[10px] bg-error/10 text-error font-bold font-mono">FAILED</span>`;
            } else {
                statusPill = `<span class="px-2 py-0.5 rounded text-[10px] bg-amber-500/10 text-amber-400 font-bold font-mono">${ex.status.toUpperCase()}</span>`;
            }
            const dateStr = ex.timestamp ? new Date(ex.timestamp).toLocaleString() : '—';
            return `
                <tr class="hover:bg-white/5 transition-colors">
                    <td class="py-3 px-4 font-mono text-primary font-medium">${ex.sid}</td>
                    <td class="py-3 px-4 truncate max-w-[120px] text-slate-300" title="${ex.filename}">${ex.filename}</td>
                    <td class="py-3 px-4">${statusPill}</td>
                    <td class="py-3 px-4 font-mono text-slate-300">${ex.steps_count}</td>
                    <td class="py-3 px-4 font-mono text-slate-300">${ex.checkpoint_count}</td>
                    <td class="py-3 px-4 text-slate-400 font-mono text-[10px]">${ex.graph_version}</td>
                    <td class="py-3 px-4 text-slate-400 font-mono">${dateStr}</td>
                </tr>
            `;
        }).join('');
    } else {
        recentTable.innerHTML = `<tr><td colspan="7" class="py-8 text-center text-on-surface-variant">No recent execution logs.</td></tr>`;
    }

    // Memory Quality Bar and stats
    const memQuality = data.memory_metrics || {};
    const avgConfidence = memQuality.avg_confidence || 0.0;
    const confidencePct = Math.round(avgConfidence * 100);
    document.getElementById('obs-mem-avg-confidence').textContent = avgConfidence.toFixed(2);
    document.getElementById('obs-mem-confidence-bar').style.width = `${confidencePct}%`;
    document.getElementById('obs-mem-accept-rate').textContent = `${memQuality.acceptance_rate}%`;
    document.getElementById('obs-mem-reject-rate').textContent = `${memQuality.rejection_rate}%`;
    document.getElementById('obs-mem-total-checks').textContent = memQuality.total_checks || 0;

    // Graph version distribution
    const versionContainer = document.getElementById('obs-version-dist-container');
    if (data.version_distribution && data.version_distribution.length > 0) {
        versionContainer.innerHTML = data.version_distribution.map(v => {
            const pct = data.total_runs > 0 ? Math.round((v.runs / data.total_runs) * 100) : 0;
            return `
                <div>
                    <div class="flex justify-between items-center text-xs mb-1">
                        <span class="font-mono text-slate-300 text-[10px]">${v.version}</span>
                        <span class="text-on-surface-variant font-mono font-medium text-[10px]">${v.runs} run${v.runs !== 1 ? 's' : ''} (${pct}%)</span>
                    </div>
                    <div class="h-1.5 bg-surface-container-high rounded-full overflow-hidden">
                        <div class="h-full bg-gradient-to-r from-secondary to-pink-500" style="width: ${pct}%"></div>
                    </div>
                </div>
            `;
        }).join('');
    } else {
        versionContainer.innerHTML = `<p class="text-xs text-on-surface-variant text-center py-4">No version distribution history.</p>`;
    }
}
