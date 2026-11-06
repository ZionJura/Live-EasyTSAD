'use strict';
const $ = (selector) => document.querySelector(selector);
const icons = {
  chart: '<path d="M4 19.5h16M6.5 15V9m5.5 6V4m5.5 11v-7"/><circle cx="17.5" cy="5" r="1"/>',
  book: '<path d="M12 5.5c-3-2-6-2-9-1v14c3-1 6-1 9 1 3-2 6-2 9-1v-14c-3-1-6-1-9 1Zm0 0v14M6 8h3m-3 4h3m6-4h3m-3 4h3"/>',
  grid: '<rect x="3" y="3" width="7" height="7" rx="1.5"/><rect x="14" y="3" width="7" height="7" rx="1.5"/><rect x="3" y="14" width="7" height="7" rx="1.5"/><rect x="14" y="14" width="7" height="7" rx="1.5"/>',
  window: '<rect x="3" y="4" width="18" height="16" rx="2"/><path d="M3 9h18M6 6.5h.1m3 0h.1m3 0h.1m-4 7-2 2 2 2m8-4 2 2-2 2"/>'
};
const dateFormat = new Intl.DateTimeFormat('zh-CN', {timeZone: 'Asia/Shanghai', year: 'numeric', month: '2-digit', day: '2-digit', hour: '2-digit', minute: '2-digit', hour12: false});
let catalog = null;
let filter = 'published';
let loading = false;
const formatTime = value => value ? dateFormat.format(new Date(value)).replaceAll('/', '.') : '修改时间未知';
function safeUrl(value) {
  if (!value) return null;
  try { const url = new URL(value); return ['http:', 'https:'].includes(url.protocol) && !url.username && !url.password ? url.href : null; } catch { return null; }
}
function render() {
  if (!catalog) return;
  const query = $('#search').value.trim().toLowerCase();
  const projects = catalog.projects.filter(p => (filter === 'all' || p.visibility === filter) && `${p.title} ${p.description} ${p.category} ${p.path || ''} ${p.ports.join(' ')}`.toLowerCase().includes(query));
  projects.sort($('#sort').value === 'name' ? (a,b) => a.title.localeCompare(b.title, 'zh-CN') : (a,b) => (b.modifiedAt || '').localeCompare(a.modifiedAt || '') || a.title.localeCompare(b.title));
  $('#projects').replaceChildren();
  $('#result-count').textContent = `${projects.length} 个`;
  $('#empty').hidden = projects.length > 0;
  projects.forEach((project, index) => {
    const card = $('#project-template').content.firstElementChild.cloneNode(true);
    if (index === 0 && project.modifiedAt && $('#sort').value === 'modified') card.classList.add('latest');
    card.querySelector('.project-icon').innerHTML = `<svg viewBox="0 0 24 24">${icons[project.icon] || icons.window}</svg>`;
    card.querySelector('.category').textContent = project.category;
    const badge = card.querySelector('.access-badge');
    badge.textContent = project.visibility === 'internal' ? '仅内部' : project.protected ? '需登录' : '公开';
    badge.classList.toggle('private', project.protected);
    card.querySelector('.project-title').textContent = project.title;
    card.querySelector('.description').textContent = project.description;
    card.querySelector('.route-value').textContent = project.accessMode === 'port' ? project.url : project.path || '尚未挂载公开访问路径';
    if (project.accessMode === 'port') card.querySelector('.project-route').title = '直接端口入口；外网连通性取决于服务器防火墙和安全组。';
    const status = card.querySelector('.status');
    status.classList.toggle('offline', project.status !== 'online');
    status.querySelector('span').textContent = project.status === 'online' ? '服务在线' : '暂时不可用';
    card.querySelector('.port-value').textContent = `端口 ${project.ports.join(' · ')}`;
    const source = {files: '网页文件修改', http: 'HTTP 修改时间', observed: '检测到内容变化'}[project.modifiedSource] || '最近修改';
    card.querySelector('.modified-label').textContent = source;
    const time = card.querySelector('time');
    time.textContent = formatTime(project.modifiedAt);
    if (project.modifiedAt) time.dateTime = project.modifiedAt;
    time.title = project.modifiedAt ? '北京时间 UTC+8' : '未获取到可靠的修改时间';
    const link = card.querySelector('.open-project');
    const url = safeUrl(project.url);
    if (url) { link.href = url; link.setAttribute('aria-label', `打开 ${project.title}（新窗口）`); }
    else { link.hidden = true; card.querySelector('.unpublished').hidden = false; }
    $('#projects').append(card);
  });
}
function syncLabel() {
  if (!catalog) return;
  const age = Math.max(0, Date.now() - new Date(catalog.generatedAt).getTime());
  $('#sync-status').textContent = `上次检测 ${formatTime(catalog.generatedAt)}${age > 180000 ? ' · 清单可能已过期' : ' · 每分钟更新'}`;
}
async function refresh() {
  if (loading) return;
  loading = true;
  $('#refresh').disabled = true;
  try {
    const response = await fetch('./projects.json', {cache: 'no-store', signal: AbortSignal.timeout(12000)});
    if (!response.ok) throw new Error('Unable to load catalog');
    const next = await response.json();
    if (!Array.isArray(next.projects) || !next.summary || !Number.isFinite(Date.parse(next.generatedAt))) throw new Error('Invalid catalog');
    catalog = next;
    $('#stat-projects').textContent = next.summary.projects;
    $('#stat-published').textContent = next.summary.published;
    $('#stat-ports').textContent = next.summary.htmlPortCount;
    $('#stat-internal').textContent = `另有 ${next.summary.internal} 个内部服务未挂载`;
    $('#ports-list').textContent = next.summary.htmlPorts.join(' · ') || '暂未检测到成功的 HTML 响应';
    $('#notice').hidden = true;
    syncLabel(); render();
  } catch {
    $('#notice').textContent = catalog ? '暂时无法更新清单，正在显示上次成功获取的结果。' : '项目清单暂时无法读取，请稍后点击“刷新清单”重试。';
    $('#notice').hidden = false;
    if (!catalog) { $('#projects').replaceChildren(); $('#sync-status').textContent = '等待项目清单'; }
  } finally { loading = false; $('#refresh').disabled = false; }
}
$('.filters').addEventListener('click', event => {
  const button = event.target.closest('[data-filter]');
  if (!button) return;
  filter = button.dataset.filter;
  document.querySelectorAll('[data-filter]').forEach(b => { b.classList.toggle('active', b === button); b.setAttribute('aria-pressed', String(b === button)); });
  render();
});
$('#search').addEventListener('input', render);
$('#sort').addEventListener('change', render);
$('#refresh').addEventListener('click', refresh);
$('#reset').addEventListener('click', () => { $('#search').value = ''; $('[data-filter="published"]').click(); });
refresh();
setInterval(() => { if (!document.hidden) refresh(); }, 60000);
document.addEventListener('visibilitychange', () => { if (!document.hidden) refresh(); });
