'use strict';

const TOKEN = document.querySelector('meta[name=token]').content;
const TILE = 256;
const CHUNK = 16;
const MIN_ZOOM = -4;
const MAX_ZOOM = 0;
const FAR = 512;
const CAND_REACH = 2048;
const POLL_MS = 1000;
const COUNTS_MS = 250;
const TILE_RETRY_MS = 1500;
const NET_TRIES = 3;
const NET_RETRY_MS = 500;
const HEAT_RETRY_MS = 2000;
const HOVER_MS = 120;
const HEAT_PICK_PX = 6;
const LEGEND_ROOM_PX = 800;
const SCALE_PX = 90;
const SCALE_STEPS = [16, 32, 64, 128, 256, 512, 1000, 2000, 4000];
const MB = 1024;
const RUNNING = 'running';
const STOPPED_STATES = ['interrupted', 'failed', 'cancelled'];
const RINGS = { edge: 'right at the edge', '1-32': '1 to 32 blocks out', '33-128': '33 to 128 blocks out', '129-256': '129 to 256 blocks out', '257-512': '257 to 512 blocks out' };
const KIND = { search: 'Best-match search', preview: 'Preview', apply: 'Apply' };
const STATE = { interrupted: 'stopped before it finished', failed: 'stopped with an error', cancelled: 'stopped by you', running: 'running now', done: 'finished', superseded: 'replaced by a newer preview', queued: 'waiting to start' };
const RUN = { search: 'Searching', preview: 'Building the preview', apply: 'Applying', rollback: 'Rolling back', verify: 'Checking your world', resume: 'Continuing', discard: 'Discarding', setup: 'Setting up the generator', clean: 'Removing the generator' };
const LIVE = { download: 'Downloading the GTNH server pack', unpack: 'Unpacking the GTNH server pack', java: 'Looking for Java', generate: 'Generating the new terrain' };
const ACTIONS = [
  [/run (this|it) again with --accept-eula.*|run: rwg2-migrator generator setup.*/i, "tick 'I accept the Minecraft EULA' on the World screen, then try again"],
  [/(resume it|finish it first|run it again)[^:]*: rwg2-migrator jobs resume \S+/i, 'click Continue for that step on the Home screen'],
  [/rwg2-migrator jobs resume \S+/i, 'click Continue for that step on the Home screen'],
  [/run this again with --confirm-stopped/i, "tick 'My server or game is stopped', then try again"],
  [/--set-populated/i, "tick 'Mark the … chunks you protect that the game never decorated as already decorated' on the Preview screen"],
  [/without --blend-biomes/i, "without 'Blend the biomes at the border'"],
  [/Make the area again \(rwg2-migrator area[^)]*\), or add these chunks with --add, then run preview again/i, "open the Area screen, where your teams' new claims are now included, click Continue, then search again"],
  [/(Search this area first|Run the search again): rwg2-migrator search .*/i, 'search again on the Best matches screen'],
  [/(Roll back|run)([^:]*): rwg2-migrator rollback .*/i, 'click Roll back on the Verify screen$2'],
  [/,? or run apply with --config-dir pointing at it/i, ', or pick its config folder on the World screen'],
  [/run again with --allow-shared-instance/i, "tick 'Go ahead anyway' on the Apply screen"],
  [/then run apply again with --auto-restart-off/i, "then tick 'I turned off anything that restarts my server' and click Apply now again"],
  [/pass (another folder|a folder on another drive) to --backup-copy/i, "choose another folder for 'Also copy the backup to another drive'"],
  [/ \(rwg2-migrator generator setup\)/i, ''],
  [/List them with: rwg2-migrator jobs list/i, 'go to the Home screen and open your world from there'],
  [/Build a new preview of the same match: rwg2-migrator preview .*/i, "click 'Build a new preview of this match' on the Preview screen"],
  [/run rwg2-migrator verify on the world/i, 'check your world on the Verify screen'],
  [/Apply the newest preview \(rwg2-migrator jobs list\)/i, 'open the newest preview from the Home screen and apply that one'],
  [/then run rollback again with --auto-restart-off/i, "then tick 'I turned off anything that restarts my server by itself' and click Roll back again"],
  [/Make it with: rwg2-migrator area .*/i, 'choose the area again on the Area screen and click Continue'],
  [/ \(rwg2-migrator ui shows it again\)/i, ''],
  [/Run preview again with a larger --ring \(at least 3\); if it repeats, report it with the job folder at/i, "build the preview again; if it repeats, click 'Copy diagnostic info' and report it at"],
  [/, or cancel it: rwg2-migrator jobs cancel \S+/i, ', or stop it first'],
  [/Pick one of them with --pick N .*/i, 'pick the match again on the Best matches screen'],
  [/, or pass --cache a new, empty folder on a drive with more room/i, ' on that drive'],
  [/, or pass --cache a folder whose path has only plain letters, for example \S+/i, ''],
  [/Pass --cache a shorter folder, for example (\S+)/i, 'close the app, then start it again from a console with a shorter cache folder: rwg2-migrator ui --cache $1'],
  [/ and pass --java\b/i, ''],
  [/\brun preview and apply again\b/i, 'build the preview and apply it again'],
  [/\brun (the )?preview again\b/i, 'build the preview again'],
  [/\brun apply again\b/i, 'click Apply now again'],
  [/\brun (this |it )?again\b/i, 'try again'],
];
const REASONS = [
  [/^search job \S+ was made for another area than .*/i, 'these matches were found before the area to keep was changed'],
  [/^(preview|search job|apply job|job) \d{8}-\d{6}-[0-9a-f]+/i, (m, what) => `this ${what.replace(/ job$/i, '')}`],
];

const $ = (s, el = document) => el.querySelector(s);
const $$ = (s, el = document) => [...el.querySelectorAll(s)];
const n = (v) => Number(v).toLocaleString('en-US');
const pct = (v, d = 1) => `${(100 * v).toFixed(d)}%`;
const key = (x, z) => `${x},${z}`;
const unkey = (k) => k.split(',').map(Number);
const LL = (x, z) => L.latLng(-z, x);
const XZ = (ll) => ({ x: Math.floor(ll.lng), z: Math.floor(-ll.lat) });
const cap = (s) => (s ? s[0].toUpperCase() + s.slice(1) : s);
const base = (p) => (p || '').split(/[\\/]/).filter(Boolean).pop() || p;
const clock = (s) => `${Math.floor(s / 60)}:${String(Math.floor(s % 60)).padStart(2, '0')}`;
const gb = (mb) => `${(mb / MB).toFixed(1)} GB`;
const blocks = (v) => `${n(v)} block${Number(v) === 1 ? '' : 's'}`;
const mostSteps = (median) => (Number(median) === 0 ? 'most steps flat' : `most steps ${blocks(median)} or less`);
const chunks = (v) => `${n(v)} chunk${Number(v) === 1 ? '' : 's'}`;
const files = (v) => `${n(v)} file${Number(v) === 1 ? '' : 's'}`;

function h(tag, attrs, ...kids) {
  const el = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs || {})) {
    if (k === 'on') for (const [ev, fn] of Object.entries(v)) el.addEventListener(ev, fn);
    else if (k === 'class') el.className = v;
    else if (k === 'text') el.textContent = v;
    else if (v === true) el.setAttribute(k, '');
    else if (v !== false && v != null) el.setAttribute(k, v);
  }
  for (const kid of kids.flat()) if (kid != null && kid !== false) el.append(kid.nodeType ? kid : String(kid));
  return el;
}

/* Chromium fails requests in flight, even to 127.0.0.1, when the computer's network changes; a step started twice is refused. */
async function send(path, init) {
  for (let tries = 1; ; tries++) {
    try {
      return await fetch(`/api/${path}`, init);
    } catch (e) {
      if (tries >= NET_TRIES) throw { reason: 'the app does not answer', action: 'Check that the rwg2-migrator window is still open, then reload this page' };
      await wait(NET_RETRY_MS);
    }
  }
}

/* X-World: the world this page shows. The open world is one for every tab, so a request about another one is refused. */
async function api(path, body) {
  const init = { headers: { 'X-Token': TOKEN, 'Content-Type': 'application/json', ...(S.world ? { 'X-World': encodeURIComponent(S.world) } : {}) } };
  if (body !== undefined) Object.assign(init, { method: 'POST', body: JSON.stringify(body) });
  const r = await send(path, init);
  const v = await r.json().catch(() => ({ error: { reason: `the app answered ${r.status}`, action: 'Reload this page' } }));
  if (v && v.error) throw v.error;
  return v;
}
async function step(path, body) {
  const r = await api(path, body);
  S.waiting = r.started;
  return r;
}
const qs = (o) => new URLSearchParams(o).toString();
const wait = (ms) => new Promise((ok) => setTimeout(ok, ms));

function plainAction(text) {
  let out = text || '';
  for (const [re, plain] of ACTIONS) out = out.replace(re, plain);
  return out;
}
function plainReason(text) {
  let out = text || '';
  for (const [re, plain] of REASONS) out = out.replace(re, plain);
  return out;
}

const S = { world: null, state: null, screen: null, legend: null, after: null, verify: null, preview: {}, search: {}, area: null, follow: null };

/* ---------- error panel, confirm, clipboard ---------- */

let lastError = null;
function fail(err) {
  if (!err || !err.reason) err = { reason: String((err && err.message) || err), action: "Reload this page; if it happens again, click 'Copy diagnostic info' and report it at the link below" };
  lastError = err;
  $('#error-reason').textContent = `${cap(plainReason(err.reason))}.`;
  $('#error-action').textContent = `${cap(plainAction(err.action)).replace(/\.$/, '')}.`;
  $('#error-logs').textContent = S.state ? `${S.state.data}${S.state.data.includes('\\') ? '\\' : '/'}logs` : '';
  $('#error').hidden = false;
}
$('#error-close').onclick = () => { $('#error').hidden = true; };
$('#error-copy').onclick = async () => {
  const d = await api('diag').catch(() => ({ text: '' }));
  await copy(`${d.text}\nShown: ${lastError ? `${lastError.reason}. Next: ${lastError.action}` : ''}`);
  $('#error-copy').textContent = 'Copied';
  setTimeout(() => { $('#error-copy').textContent = 'Copy diagnostic info'; }, 1500);
};

async function copy(text) {
  try {
    await navigator.clipboard.writeText(text);
  } catch (e) {
    const t = h('textarea', {}, text);
    document.body.append(t);
    t.select();
    document.execCommand('copy');
    t.remove();
  }
}

function confirmBox(title, text, yes) {
  return new Promise((ok) => {
    $('#confirm-title').textContent = title;
    $('#confirm-text').textContent = text;
    $('#confirm-yes').textContent = yes;
    $('#confirm').hidden = false;
    const done = (v) => { $('#confirm').hidden = true; ok(v); };
    $('#confirm-yes').onclick = () => done(true);
    $('#confirm-no').onclick = () => done(false);
  });
}

/* ---------- folder browser ---------- */

function browseFor(mode, start) {
  return new Promise((ok) => {
    const box = $('#browser');
    $('#browser-title').textContent = mode === 'zip' ? 'Pick the GTNH server zip' : 'Pick a folder';
    $('#browser-use').hidden = mode === 'zip';
    let here = start || '';
    const close = (v) => { box.hidden = true; ok(v); };
    async function show(path) {
      try {
        const r = await api(`browse?${qs({ path, ext: mode === 'zip' ? '.zip' : '' })}`);
        here = r.path;
        $('#browser-path').value = r.path;
        $('#browser-up').disabled = r.parent === null;
        $('#browser-up').onclick = () => show(r.parent || '');
        const list = $('#browser-list');
        list.replaceChildren(...r.entries.map((e) => h('li', { on: { click: () => (e.kind === 'file' ? close(e.path) : show(e.path)) } },
          e.name, e.kind !== 'folder' ? h('span', { class: 'k' }, { drive: 'drive', world: 'world', server: 'server', file: 'zip' }[e.kind]) : null)));
        if (!r.entries.length) list.append(h('li', { class: 'muted' }, mode === 'zip' ? 'No zip file here' : 'No folders here'));
      } catch (e) { fail(e); }
    }
    $('#browser-go').onclick = () => show($('#browser-path').value);
    $('#browser-cancel').onclick = () => close(null);
    $('#browser-use').onclick = () => close(here || null);
    box.hidden = false;
    show(here);
  });
}

/* ---------- maps ---------- */

function baseMap(el, interactive = true) {
  const m = L.map(el, {
    crs: L.CRS.Simple, minZoom: MIN_ZOOM, maxZoom: MAX_ZOOM, zoomSnap: 1, zoomDelta: 1, attributionControl: false,
    zoomControl: false, dragging: interactive, scrollWheelZoom: interactive, doubleClickZoom: interactive, boxZoom: false,
    keyboard: interactive, touchZoom: interactive, inertia: true,
  });
  if (interactive) { L.control.zoom({ position: 'bottomleft' }).addTo(m); scale(m); }
  return m;
}

const waitingTiles = new Set();

function retryTile(t) {
  clearTimeout(t.retry);
  const tries = (Number(t.dataset.tries) || 0) + 1;
  t.dataset.tries = tries;
  t.src = `${t.src.replace(/&r=\d+$/, '')}&r=${tries}`;
}

function tileLayer(url, extra) {
  const layer = L.tileLayer(url, { tileSize: TILE, minZoom: MIN_ZOOM, maxZoom: MAX_ZOOM, noWrap: true, keepBuffer: 3, ...extra });
  layer.on('tileerror', (e) => {
    const t = e.tile;
    waitingTiles.add(t);
    clearTimeout(t.retry);
    t.retry = setTimeout(() => { if (layer._map && t.isConnected) retryTile(t); }, TILE_RETRY_MS);
  });
  layer.on('tileload tileunload', (e) => { waitingTiles.delete(e.tile); clearTimeout(e.tile.retry); });
  return layer;
}

function blockDrawn(e) {
  for (const t of [...waitingTiles]) {
    const p = t.isConnected && t.src.match(/\/tiles\/[^/]+\/(-?\d+)\/(-?\d+)\/(-?\d+)\.png\?v=([^&]+)/);
    if (!p) { waitingTiles.delete(t); continue; }
    const dx = Number(p[2]) - e.x;
    const dy = Number(p[3]) - e.y;
    if (p[4] === e.v && Number(p[1]) === e.z && dx >= 0 && dx < e.n && dy >= 0 && dy < e.n) retryTile(t);
  }
}

/* A browser opens at most 6 connections to a host across all its tabs: one tab streams and passes each block on to the others. */
function tileStream() {
  const name = `rwg2-migrator-tiles-${TOKEN}`;
  if (!navigator.locks || !window.BroadcastChannel) return;
  const bus = new BroadcastChannel(name);
  bus.onmessage = (m) => blockDrawn(m.data);
  navigator.locks.request(name, () => new Promise(() => {
    const src = new EventSource(`/api/tilestream?${qs({ t: TOKEN })}`);
    src.onmessage = (m) => { const e = JSON.parse(m.data); blockDrawn(e); bus.postMessage(e); };
  }));
}

function blockBounds(b) {
  return L.latLngBounds(LL(b[0], b[3]), LL(b[2], b[1]));
}

function cellBounds(cells, grow = 0) {
  if (!cells.length) return null;
  let x0 = Infinity; let z0 = Infinity; let x1 = -Infinity; let z1 = -Infinity;
  for (const [x, z] of cells) { x0 = Math.min(x0, x); z0 = Math.min(z0, z); x1 = Math.max(x1, x); z1 = Math.max(z1, z); }
  return [x0 * CHUNK - grow, z0 * CHUNK - grow, (x1 + 1) * CHUNK + grow, (z1 + 1) * CHUNK + grow];
}

function scale(m) {
  const Ctl = L.Control.extend({
    onAdd() { this.el = h('div', { class: 'scale' }); m.on('zoomend', () => this.update()); this.update(); return this.el; },
    update() {
      const ppb = 2 ** m.getZoom();
      const blocks = SCALE_STEPS.find((b) => b * ppb >= SCALE_PX) || SCALE_STEPS[SCALE_STEPS.length - 1];
      this.el.replaceChildren(h('span', {}, `${n(blocks)} blocks`), h('div', { style: `width:${blocks * ppb}px` }));
    },
  });
  new Ctl({ position: 'bottomleft' }).addTo(m);
}

/* Folded on a map too narrow for it (side by side at 1366 px), where it would cover the base; a click on it is remembered. */
function legend(m, heat) {
  const lg = S.legend;
  const Ctl = L.Control.extend({
    onAdd() {
      const rows = lg.looks.map((l) => h('div', {}, h('i', { style: `background:${l.colour}` }), l.label));
      rows.push(h('div', {}, h('i', { style: 'background:transparent;border:2px solid #fff' }), 'Kept area (with the margin)'));
      rows.push(h('div', {}, h('i', { style: 'background:transparent;border:2px solid #000;outline:1px solid #fff' }), 'Protected area (kept as it is)'));
      if (heat) {
        rows.push(h('div', {}, h('i', { style: `background:${lg.taint}` }), 'Tainted Land'));
        const stops = lg.heat.map((s, i) => `${s.colour} ${(100 * i) / (lg.heat.length - 1)}%`).join(',');
        rows.push(h('div', {}, 'Step at the kept border (blocks)'), h('div', { class: 'ramp', style: `background:linear-gradient(90deg,${stops})` }),
          h('div', { class: 'ticks' }, ...lg.heat.map((s, i) => h('span', {}, i === lg.heat.length - 1 ? `${s.step}+` : s.step))));
      }
      const summary = h('summary', {}, 'Legend');
      const box = h('details', { class: 'legend' }, summary, h('div', {}, ...rows));
      box.open = S.legendOpen ?? m.getSize().x >= LEGEND_ROOM_PX;
      summary.addEventListener('click', () => { S.legendOpen = !box.open; });
      L.DomEvent.disableClickPropagation(box);
      L.DomEvent.disableScrollPropagation(box);
      return box;
    },
  });
  return new Ctl({ position: 'bottomright' }).addTo(m);
}

const CellsLayer = L.GridLayer.extend({
  options: { minZoom: MIN_ZOOM, maxZoom: MAX_ZOOM },
  createTile(coords) {
    const c = h('canvas', { width: TILE, height: TILE });
    const bpp = 2 ** -coords.z;
    const span = TILE * bpp;
    const t = { cx0: (coords.x * span) / CHUNK, cz0: (coords.y * span) / CHUNK, count: span / CHUNK, s: CHUNK / bpp };
    this.options.draw(c.getContext('2d'), t);
    return c;
  },
});

const Backdrop = L.GridLayer.extend({
  options: { minZoom: MIN_ZOOM, maxZoom: MAX_ZOOM, tileSize: TILE },
  createTile() { return h('div', { class: 'backdrop' }); },
});

function cellsIn(t, set, fn) {
  for (const k of set) {
    const [cx, cz] = unkey(k);
    const i = cx - t.cx0;
    const j = cz - t.cz0;
    if (i >= 0 && i < t.count && j >= 0 && j < t.count) fn(i * t.s, j * t.s, cx, cz);
  }
}

function edges(ctx, t, set, style) {
  ctx.beginPath();
  cellsIn(t, set, (px, pz, cx, cz) => {
    if (!set.has(key(cx, cz - 1))) { ctx.moveTo(px, pz); ctx.lineTo(px + t.s, pz); }
    if (!set.has(key(cx, cz + 1))) { ctx.moveTo(px, pz + t.s); ctx.lineTo(px + t.s, pz + t.s); }
    if (!set.has(key(cx - 1, cz))) { ctx.moveTo(px, pz); ctx.lineTo(px, pz + t.s); }
    if (!set.has(key(cx + 1, cz))) { ctx.moveTo(px + t.s, pz); ctx.lineTo(px + t.s, pz + t.s); }
  });
  for (const [colour, width] of style) { ctx.strokeStyle = colour; ctx.lineWidth = width; ctx.stroke(); }
}

function outlineLayer(kept, core) {
  return new CellsLayer({ pane: 'overlayPane', draw: (ctx, t) => { edges(ctx, t, kept, [['#fff', 2]]); edges(ctx, t, core, [['#fff', 4], ['#000', 2]]); } });
}

function dilate(cells, r) {
  const out = new Set();
  for (const k of cells) {
    const [x, z] = unkey(k);
    for (let dx = -r; dx <= r; dx += 1) for (let dz = -r; dz <= r; dz += 1) out.add(key(x + dx, z + dz));
  }
  return out;
}

function keptDistance(x, z, kept) {
  let best = Infinity;
  for (const k of kept) {
    const [cx, cz] = unkey(k);
    const gx = Math.max(cx * CHUNK - x, 0, x - cx * CHUNK - (CHUNK - 1));
    const gz = Math.max(cz * CHUNK - z, 0, z - cz * CHUNK - (CHUNK - 1));
    best = Math.min(best, Math.hypot(gx, gz));
  }
  return best;
}

/* layerAt(p, m): the layer under the pointer, so a CANDIDATE or SMOOTHED map reads out what it shows, not today's world. */
function readout(m, context, layerAt) {
  const Ctl = L.Control.extend({ onAdd() { this.el = h('div', { class: 'readout' }, 'Point at the map'); return this.el; } });
  const ctl = new Ctl({ position: 'bottomleft' }).addTo(m);
  let timer = null;
  let seq = 0;
  m.on('mousemove', (e) => {
    clearTimeout(timer);
    timer = setTimeout(async () => {
      const p = XZ(e.latlng);
      const mine = (seq += 1);
      const parts = [`x ${p.x}, z ${p.z}`];
      try {
        const layer = layerAt ? layerAt(p, m) : null;
        const r = await api(`probe?${qs({ ...p, ...(layer ? { layer, zoom: m.getZoom() } : {}) })}`);
        if (mine !== seq) return;
        parts.push(r.pending ? 'Not drawn here yet' : r.biome ? `${r.biome}, ground at y ${r.ground}` : 'not explored');
      } catch (err) { /* the readout is best effort */ }
      const extra = context ? context(p, m) : [];
      ctl.el.replaceChildren(...[...parts, ...extra].map((s) => h('div', {}, s)));
    }, HOVER_MS);
  });
  return ctl;
}

function placeText(p, kept, core) {
  const k = key(Math.floor(p.x / CHUNK), Math.floor(p.z / CHUNK));
  if (core.has(k)) return 'Protected: kept exactly as it is';
  if (kept.has(k)) return 'In the safety margin: old terrain, smoothed at its border';
  if (!kept.size) return null;
  return `${n(Math.round(keptDistance(p.x, p.z, kept)))} blocks from the kept area`;
}

function note(container, text) {
  let el = $('.mapnote', container);
  if (!text) { if (el) el.remove(); return; }
  if (!el) { el = h('div', { class: 'mapnote', role: 'status' }); container.append(el); }
  el.textContent = text;
}

/* The generator that draws the matches: starting, or refused (low memory, the EULA, a changed config) with its next action. */
function mapNotes(st) {
  const p = st.map_problem;
  const text = p ? `The new terrain cannot be drawn now: ${p.reason}. ${cap(plainAction(p.action)).replace(/\.$/, '')}.` : cap(st.map_status);
  for (const c of [S.compare, S.pcompare]) if (c) note(c.el, text);
  if (p && p.reason !== S.mapProblem && ['matches', 'preview'].includes(S.screen)) fail(p);
  S.mapProblem = p ? p.reason : null;
}

/* ---------- comparison view: swipe or side by side ---------- */

function Compare(el) {
  const self = { el, mode: 'swipe', layers: {}, left: null, right: null, choices: [], maps: [], heat: {}, markers: [], kept: new Set(), core: new Set(), view: null, frac: 0.5 };

  function heatDots(m, side) {
    if (m.heatLayer) m.heatLayer.remove();
    const sets = self.mode === 'swipe' ? [[self.left, 'heatLeft'], [self.right, 'heatRight']] : [[side, 'overlayPane']];
    m.heatLayer = L.layerGroup(sets.flatMap(([s, pane]) => {
      const renderer = L.canvas({ padding: 0.5, pane });
      return (self.heat[s] || []).map((d) => L.circleMarker(LL(d[0] + 0.5, d[1] + 0.5), { renderer, radius: 3, stroke: false, fillColor: d[3], fillOpacity: 1, interactive: false }));
    })).addTo(m);
  }

  function overlays(m, side) {
    outlineLayer(self.kept, self.core).addTo(m);
    heatDots(m, side);
    for (const mk of self.markers) {
      L.circleMarker(LL(mk.x, mk.z), { radius: 9, color: mk.colour, weight: 3, fill: false }).bindTooltip(mk.text, { permanent: true, direction: 'right' }).addTo(m);
    }
  }

  function sideAt(p, m) {
    if (self.mode !== 'swipe') return m === self.maps[0] ? self.left : self.right;
    return m.latLngToContainerPoint(LL(p.x, p.z)).x < m.getSize().x * self.frac ? self.left : self.right;
  }

  function heatNear(p, m) {
    const tol = HEAT_PICK_PX * 2 ** -m.getZoom();
    const all = self.heat[sideAt(p, m)] || [];
    let best = null;
    for (const d of all) {
      const dist = Math.max(Math.abs(d[0] - p.x), Math.abs(d[1] - p.z));
      if (dist <= tol && (!best || dist < best[0])) best = [dist, d[2]];
    }
    return best ? [`Step at the kept border here: ${blocks(best[1])}`] : [];
  }

  function context(p, m) {
    return [placeText(p, self.kept, self.core), ...heatNear(p, m)].filter(Boolean);
  }

  function label(m, text, right) {
    const box = m.getContainer();
    box.append(h('div', { class: 'side-label', style: right ? 'right:10px' : 'left:10px;top:50px' }, text));
  }

  function swipe(m) {
    const pane = m.createPane('right');
    pane.style.zIndex = 250;
    const heatPanes = ['heatLeft', 'heatRight'].map((name) => { const p = m.createPane(name); p.style.zIndex = 410; return p; });
    new Backdrop({ pane, bounds: self.layers[self.right].bounds }).addTo(m);
    tileLayer(self.layers[self.right].url, { pane, bounds: self.layers[self.right].bounds }).addTo(m);
    const handle = h('div', { class: 'swipe' });
    m.getContainer().append(handle);
    function clip() {
      const size = m.getSize();
      const x = size.x * self.frac;
      handle.style.left = `${x}px`;
      const nw = m.containerPointToLayerPoint([0, 0]);
      const se = m.containerPointToLayerPoint(size);
      const cut = m.containerPointToLayerPoint([x, 0]).x;
      pane.style.clip = `rect(${nw.y}px, ${se.x}px, ${se.y}px, ${cut}px)`;
      heatPanes[0].style.clip = `rect(${nw.y}px, ${cut}px, ${se.y}px, ${nw.x}px)`;
      heatPanes[1].style.clip = pane.style.clip;
    }
    L.DomEvent.disableClickPropagation(handle);
    handle.addEventListener('pointerdown', (e) => {
      handle.setPointerCapture(e.pointerId);
      const move = (ev) => { const r = m.getContainer().getBoundingClientRect(); self.frac = Math.min(Math.max((ev.clientX - r.left) / r.width, 0), 1); clip(); };
      handle.addEventListener('pointermove', move);
      handle.addEventListener('pointerup', () => handle.removeEventListener('pointermove', move), { once: true });
    });
    m.on('move zoom resize', clip);
    clip();
  }

  self.build = function build() {
    const view = self.maps[0] ? [self.maps[0].getCenter(), self.maps[0].getZoom()] : self.view;
    for (const m of self.maps) m.remove();
    self.maps = [];
    el.replaceChildren();
    if (!self.left || !self.right) { el.append(h('div', { class: 'empty' }, self.empty || 'Nothing to show yet')); return; }
    const box = h('div', { class: 'cmp-maps' });
    el.append(box, bar());
    const panes = self.mode === 'swipe' ? [h('div')] : [h('div'), h('div')];
    box.append(...panes);
    panes.forEach((p, i) => {
      const m = baseMap(p);
      const side = i === 0 ? self.left : self.right;
      tileLayer(self.layers[side].url, { bounds: self.layers[side].bounds }).addTo(m);
      if (self.mode === 'swipe') swipe(m);
      overlays(m, side);
      readout(m, context, sideAt);
      if (i === panes.length - 1) legend(m, true);
      label(m, self.layers[self.left].label, false);
      if (self.mode === 'swipe' || i === 1) label(m, self.layers[self.right].label, true);
      if (i === 1) { const l = $('.side-label', p); if (l && l.textContent === self.layers[self.left].label) l.remove(); }
      if (view) m.setView(view[0], view[1], { animate: false });
      else if (self.fit) m.fitBounds(self.fit);
      self.maps.push(m);
    });
    if (self.maps.length === 2) syncMaps(...self.maps);
  };

  function bar() {
    const b = h('div', { class: 'cmp-bar' });
    for (const [mode, text] of [['swipe', 'Swipe'], ['side', 'Side by side']]) {
      b.append(h('button', { class: self.mode === mode ? 'on' : '', on: { click: () => { self.mode = mode; self.build(); } } }, text));
    }
    if (self.choices.length > 2) {
      for (const side of ['left', 'right']) {
        const sel = h('select', { 'aria-label': side === 'left' ? 'Left map' : 'Right map', on: { change: (e) => { self[side] = e.target.value; self.build(); } } },
          ...self.choices.map((c) => h('option', { value: c, selected: self[side] === c }, self.layers[c].label)));
        b.append(sel);
      }
    }
    return b;
  }

  self.set = function set(opts) {
    Object.assign(self, opts);
    self.build();
  };
  self.setHeat = function setHeat(heat) {
    self.heat = heat;
    self.maps.forEach((m, i) => heatDots(m, i === 0 ? self.left : self.right));
  };
  return self;
}

function syncMaps(a, b) {
  let busy = false;
  const follow = (from, to) => from.on('move zoomend', () => {
    if (busy) return;
    busy = true;
    to.setView(from.getCenter(), from.getZoom(), { animate: false });
    busy = false;
  });
  follow(a, b);
  follow(b, a);
}

/* The seam heat of a map while still() says its view is on screen; null once it is not, and its refusal is then nobody's. */
async function heatFor(params, still) {
  while (still()) {
    try {
      const r = await api(`heat?${qs(params)}`);
      if (r.dots) return r.dots;
    } catch (e) {
      if (still()) throw e;
    }
    await wait(HEAT_RETRY_MS);
  }
  return null;
}

/* ---------- state polling, busy indicator, tab title ---------- */

function liveText(r) {
  const e = r.live;
  if (!e) return '';
  if (e.step === 'boot') return e.event === 'ready' ? 'Generator ready' : `Starting the generator (${e.seconds || 0} s)`;
  if (e.step === 'check') return 'Checking the generator';
  if (e.event === 'progress' && e.total) return `${LIVE[e.step] || 'Working'}: ${n(e.done)} of ${n(e.total)}`;
  if (e.step && LIVE[e.step]) return LIVE[e.step];
  if (e.text && /^(preview|generator|generate|apply):/.test(e.text)) return cap(e.text.replace(/^\w+: /, ''));
  return '';
}

function busy(st) {
  const r = st.runner;
  const on = r && r.running;
  $('#busy').hidden = !on;
  if (on) {
    const live = liveText(r);
    $('#busy-text').textContent = `${RUN[r.kind] || 'Working'} · ${clock(Date.now() / 1000 - r.started)}${live ? ` · ${live}` : ''}`;
    document.title = `${S.progressTitle || RUN[r.kind] || 'Working'} ${clock(Date.now() / 1000 - r.started)} – rwg2-migrator`;
  } else {
    document.title = r && r.code === 0 ? 'Done – rwg2-migrator' : 'rwg2-migrator';
  }
  for (const b of $$('#steps button')) b.disabled = !allowed(b.dataset.go, st);
}

function allowed(screen, st) {
  const w = st.world;
  const scanned = w && w.scan && w.scan.chunks !== undefined;
  if (screen === 'area') return !!scanned;
  if (screen === 'matches') return !!(scanned && st.area);
  if (screen === 'preview') return !!(scanned && (S.pick || S.preview.job));
  if (screen === 'apply') return !!(scanned && S.preview.job);
  if (screen === 'verify') return !!w;
  return true;
}

async function poll() {
  let st = null;
  try {
    st = await api('state');
  } catch (e) {
    $('#busy').hidden = false;
    $('#busy-text').textContent = 'The app does not answer: is its window still open?';
  }
  if (st) {
    try {
      const was = S.state && S.state.runner;
      S.state = st;
      if (!S.world && st.world) S.world = st.world.world;
      busy(st);
      mapNotes(st);
      const r = st.runner;
      const mine = r && (r.started === S.waiting || (was && was.running && was.started === r.started));
      if (mine && !r.running) { S.waiting = null; finished(r); }
      const hook = screens[S.screen] && screens[S.screen].state;
      if (hook) hook.call(screens[S.screen], st);
    } catch (e) { console.error(e); }
  }
  setTimeout(poll, POLL_MS);
}

async function stepJob() {
  for (;;) {
    const st = await api('state');
    S.state = st;
    if (st.runner && st.runner.job) return st.runner.job;
    if (st.runner && !st.runner.running) return null;
    await wait(POLL_MS / 2);
  }
}

function live(job, id) {
  const r = S.state.runner;
  return job.state === RUNNING || job.state === 'queued' || !!(r && r.running && r.job === id);
}

function jobOf(id) {
  return (id && S.state && S.state.jobs.find((x) => x.id === id)) || {};
}
const jobKind = (id) => jobOf(id).kind || null;

function finished(r) {
  const kind = r.kind === 'resume' ? jobKind(r.job) : r.kind;
  const job = r.job;
  if (r.stopped) { stoppedStep(kind, job); return; }
  if (r.error && !r.result) { fail(r.error); }
  if (kind === 'search' && job) go('matches', job);
  else if (kind === 'preview' && job) go('preview', job);
  else if (kind === 'apply' && !r.error) applied(r.result, jobOf(job).world || (S.apply && S.apply.data && S.apply.data.world));
  else if (kind === 'verify' && r.result) { S.verify = r.result; screens.verify.show(); }
  else if (kind === 'rollback' && r.result) rolledBack(r.result).catch(fail);
  else if (kind === 'setup') { if (!r.error) S.state.generator = true; screens.world.enter(); }
  else if (kind === 'clean') { if (!r.error) S.state.generator = false; if (S.screen === 'world') screens.world.enter(); }
  else if (S.screen === 'home') screens.home.enter();
}

/* Stopped with the Stop button: no error; the step's screen (or Home) offers Continue. */
function stoppedStep(kind, job) {
  if (kind === 'search' && job) go('matches', job);
  else if (kind === 'preview' && job) go('preview', job);
  else go('home');
}

async function applied(result, world) {
  S.after = result;
  try {
    if (world) await ensureWorld(world, true);
    go('after');
  } catch (e) { fail(e); }
}

async function rolledBack(r) {
  S.verify = null;
  go('verify');
  const gone = Object.entries(r.configs_by_hand || {}).map(([path, lines]) => h('div', {},
    h('p', { class: 'warn' }, `Not restored, its folder is gone: ${path}. If a server or game still uses that file, set these lines back there:`),
    h('pre', { class: 'wrap' }, lines.join('\n'))));
  $('#v-result').replaceChildren(h('p', { class: 'ok' }, `Your world from before Apply is back. The migrated world is kept in ${r.rolled_back}.`),
    ...gone, h('p', {}, 'Start your server or game again when you are ready.'));
  $('#v-result').hidden = false;
  await ensureWorld(r.world, true);
  screens.verify.enter();
}

/* A world opened before a step changed it (Apply, Roll back) is opened again with fresh, so its card is what it is now. */
async function ensureWorld(path, fresh) {
  const w = S.state && S.state.world;
  if (fresh || !w || w.world !== path) await api('world', { path });
  for (;;) {
    const st = await api('state');
    S.state = st;
    const sc = st.world && st.world.scan;
    if (st.world) S.world = st.world.world;
    if (st.world && (st.world.migrated || (sc && sc.chunks !== undefined))) return st.world;
    if (sc && sc.error) throw sc.error;
    await wait(POLL_MS);
  }
}

function follow(jobId, onEvents) {
  if (S.follow) S.follow.close();
  const f = { job: jobId, at: 0, on: true, close() { f.on = false; if (S.follow === f) S.follow = null; } };
  S.follow = f;
  (async () => {
    while (f.on) {
      const r = await api(`events?${qs({ job: jobId, at: f.at })}`).catch(() => null);
      if (!f.on) return;
      if (r) {
        f.at = r.at;
        try { onEvents(r.end ? [...r.events, { event: 'end' }] : r.events); } catch (e) { console.error(e); }
        if (r.end) { f.close(); return; }
      }
      await wait(POLL_MS);
    }
  })();
  return f;
}

/* ---------- screens ---------- */

const screens = {};

function go(name, arg) {
  for (const s of $$('.screen')) s.hidden = s.id !== `screen-${name}`;
  for (const b of $$('#steps button')) b.classList.toggle('on', b.dataset.go === name);
  if (S.follow && name !== S.screen) { S.follow.close(); S.follow = null; }
  S.screen = name;
  S.progressTitle = null;
  history.replaceState(null, '', `${location.pathname}${location.search}#${name}${arg ? `/${arg}` : ''}`);
  Promise.resolve(screens[name].enter(arg)).catch(fail);
}

for (const b of $$('#steps button')) b.onclick = () => go(b.dataset.go);

/* Home */
screens.home = {
  enter() {
    const st = S.state;
    if (!st) return;
    const open = st.jobs.filter((j) => STOPPED_STATES.includes(j.state) || j.state === RUNNING).reverse();
    const done = st.jobs.filter((j) => j.state === 'done' && KIND[j.kind]).reverse().slice(0, 6);
    $('#home-continue').hidden = !open.length;
    $('#home-jobs').replaceChildren(...open.map((j) => jobRow(j, true)));
    $('#home-done').hidden = !done.length;
    $('#home-finished').replaceChildren(...done.map((j) => jobRow(j, false)));
    $('#home-inuse').replaceChildren(...(st.world ? [h('p', { class: 'small muted' }, `Open now: ${base(st.world.world)}`), inUse(st.world)] : []));
    $('#home-recent').hidden = !st.recent.length;
    $('#home-worlds').replaceChildren(...st.recent.map((p) => h('li', {}, h('span', { class: 'what' }, h('b', {}, base(p)), h('br'), h('span', { class: 'muted small' }, p)),
      h('span', { class: 'acts' }, h('button', { on: { click: () => openWorld(p) } }, 'Open')))));
  },
  state(st) {
    const sig = JSON.stringify([st.jobs.map((j) => j.state), st.recent, st.world && st.world.in_use]);
    if (sig !== this.sig) { this.sig = sig; this.enter(); }
  },
};
$('#home-start').onclick = () => go('world');

function jobRow(j, open) {
  const acts = [];
  if (open && j.state === RUNNING) acts.push(h('button', { on: { click: () => showJob(j) } }, 'Show'));
  if (open && j.state !== RUNNING) {
    acts.push(h('button', { class: 'primary', on: { click: () => resume(j) } }, 'Continue'));
    acts.push(h('button', { class: 'ghost', on: { click: () => discard(j) } }, 'Discard'));
  }
  if (!open && !j.rolled_back) acts.push(h('button', { on: { click: () => showJob(j) } }, { search: 'See the matches', preview: 'See the preview', apply: 'Show the rules' }[j.kind]));
  if (!open && j.kind === 'preview' && !j.applied) acts.push(h('button', { class: 'ghost', on: { click: () => discard(j) } }, 'Discard'));
  const when = new Date(j.created * 1000).toLocaleString();
  const state = j.rolled_back ? 'rolled back: the world from before Apply is back' : STATE[j.state] || j.state;
  return h('li', {}, h('span', { class: 'what' }, h('b', {}, `${KIND[j.kind] || j.kind} · ${base(j.world)}`), h('br'),
    h('span', { class: 'muted small' }, `${cap(state)} · started ${when}`)), h('span', { class: 'acts' }, ...acts));
}

async function showJob(j) {
  try {
    await ensureWorld(j.world, j.kind === 'apply');
    if (j.kind === 'search') go('matches', j.id);
    else if (j.kind === 'preview') go('preview', j.id);
    else if (j.kind === 'apply') { S.after = null; go('after'); }
  } catch (e) { fail(e); }
}

async function resume(j) {
  try {
    if (j.kind === 'apply' && !j.open && j.preview) { await ensureWorld(j.world); go('apply', j.preview); return; }
    if (j.kind !== 'apply') await ensureWorld(j.world);
    await step('resume', { job: j.id });
    S.state = await api('state');
    if (j.kind === 'search') go('matches', j.id);
    else if (j.kind === 'preview') go('preview', j.id);
    else screens.home.enter();
  } catch (e) { fail(e); }
}

async function discard(j) {
  const ok = await confirmBox('Discard this step?', `${KIND[j.kind] || j.kind} of ${base(j.world)}: its progress is deleted${j.kind === 'preview' ? ', with its working copy next to your world' : ''}. Your world is not touched.`, 'Discard');
  if (ok) step('discard', { job: j.id }).catch(fail);
}

/* World */
screens.world = {
  async enter() {
    const st = S.state;
    $('#world-eula').checked = !!(st && st.eula);
    generatorLine(st);
    if (st && st.world) { $('#world-path').value = st.world.world; worldCard(st.world); }
    if (!this.saves) {
      this.saves = true;
      const r = await api('saves').catch(() => ({ saves: [] }));
      $('#world-saves').replaceChildren(...(r.saves.length ? r.saves.map((s) => h('li', {},
        h('span', { class: 'what' }, h('b', {}, s.name), h('br'), h('span', { class: 'muted small' }, s.error || `Last played ${new Date(s.last_played).toLocaleString()} · ${s.path}`)),
        h('span', { class: 'acts' }, h('button', { on: { click: () => openWorld(s.path, false) } }, 'Open')))) : [h('li', { class: 'muted' }, 'No single-player save found in the usual launcher folders.')]));
    }
  },
  state(st) {
    if (st.world && st.world.world === S.world) worldCard(st.world);
    generatorLine(st);
  },
};

function generatorLine(st) {
  const set = !!(st && st.generator);
  const line = $('#world-generator');
  if (line.dataset.set === String(set)) return;
  line.dataset.set = String(set);
  line.replaceChildren(...(set
    ? ['The generator is set up on this computer. ', h('button', { class: 'small-btn', on: { click: removeGenerator } }, 'Remove the downloaded generator')]
    : ['The generator (the GTNH server pack, 568 MB, once per GTNH version) is downloaded and set up when the search starts.']));
}

async function removeGenerator() {
  if (!(await confirmBox('Remove the downloaded generator?', 'The GTNH server pack and the Java the tool downloaded are deleted (about 0.8 GB). Your worlds, previews and backups are not touched. The next search downloads and sets it up again.', 'Remove'))) return;
  step('clean', {}).catch(fail);
}

/* remote undefined keeps the choice last made for this world. */
async function openWorld(path, remote) {
  try {
    if (S.screen !== 'world') go('world');
    $('#world-path').value = path;
    const card = await api('world', { path, remote });
    S.world = card.world;
    dropArea();
    S.state = await api('state');
    worldCard(card);
  } catch (e) { fail(e); }
}

$('#world-open').onclick = () => openWorld($('#world-path').value, $('#world-remote').checked);
$('#world-browse').onclick = async () => { const p = await browseFor('folder', $('#world-path').value); if (p) { $('#world-path').value = p; openWorld(p, $('#world-remote').checked); } };
$('#world-eula').onchange = (e) => api('eula', { accept: e.target.checked }).then(() => { S.state.eula = e.target.checked; }).catch(fail);
$('#world-zip').onclick = async () => { const p = await browseFor('zip', ''); if (p) step('setup', { zip: p }).catch(fail); };
$('#world-continue').onclick = () => go('area');

function inUse(w) {
  const u = w.in_use;
  const text = {
    stopped: `Your world is stopped (${u.detail}). Keep it closed until Apply is done.`,
    running: `Close the game or stop the server first: your world is open right now (${u.detail}). Keep it closed until Apply is done.`,
    unknown: `The tool cannot tell whether your world is open (${u.detail}). Close the game or stop the server first, and keep it closed until Apply is done.`,
  }[u.state] || u.detail;
  return h('div', { class: `inuse ${u.state}`, role: 'status' }, text);
}

const MIGRATED = {
  applied: (m) => `This tool migrated this world to RWG2 Smooth${m.date ? ` on ${m.date}` : ''}. Check it on the Verify screen after every GTNH update; Roll back is there too.`,
  staged: () => 'An Apply of this world has not finished. Click Continue for it on the Home screen.',
  'rolling back': () => 'A Roll back of this world stopped halfway, so the world folder is not there now. Finish it on the Verify screen: tick the boxes and click Roll back.',
};

function migratedCard(w) {
  const m = w.migrated;
  const sig = JSON.stringify([w.world, m, w.in_use]);
  if ($('#world-card').dataset.sig === sig) return;
  $('#world-card').dataset.sig = sig;
  const acts = [h('button', { class: 'primary', on: { click: () => go('verify') } }, 'Go to Verify')];
  if (m.state === 'applied') acts.push(h('button', { on: { click: () => { S.after = null; go('after'); } } }, 'Show the rules'));
  $('#world-card').replaceChildren(inUse(w), h('p', {}, (MIGRATED[m.state] || MIGRATED.staged)(m)), h('div', { class: 'row' }, ...acts));
  $('#world-card').hidden = false;
  $('#world-next').hidden = true;
}

const NO_CONFIG = {
  world: "No config folder of its own was found next to this world. The search and the preview use GTNH's default RWG2 settings. Apply writes no config file; it shows the 4 lines to add by hand to config/rwg2.cfg of the server or game instance that will run this world. If you have that config folder, pick it here.",
  migrated: 'No config folder of its own was found next to this world, so Verify cannot check the 4 lines. If you have the config folder of the server or game instance that runs it, pick it here.',
};

function configBlock(w) {
  $('#world-config-text').textContent = w.config
    ? `In use: ${w.config}${w.config_from ? ` (${w.config_from})` : ''}. If this is not the config folder of the server or game instance that runs this world, pick that one here.`
    : NO_CONFIG[w.migrated ? 'migrated' : 'world'];
  const input = $('#world-config-path');
  if (input.dataset.world !== w.world) { input.dataset.world = w.world; input.value = w.config_chosen || ''; }
  $('#world-config').hidden = false;
}

async function useConfig() {
  try {
    const card = await api('config', { world: S.world, path: $('#world-config-path').value.trim() });
    S.state = await api('state');
    $('#world-config-path').dataset.world = '';
    worldCard(card);
  } catch (e) { fail(e); }
}

$('#world-config-use').onclick = useConfig;
$('#world-config-browse').onclick = async () => { const p = await browseFor('folder', $('#world-config-path').value || S.world); if (p) { $('#world-config-path').value = p; useConfig(); } };

function worldCard(w) {
  $('#world-remote').checked = !!w.remote;
  configBlock(w);
  if (w.migrated) { migratedCard(w); return; }
  $('#world-card').dataset.sig = '';
  const s = w.scan || {};
  const ready = s.chunks !== undefined;
  const formats = ready ? Object.entries(s.formats).map(([f, c]) => `${f} ${n(c)}`).join(', ') : '';
  const claims = Object.entries(w.claims).map(([t, c]) => `${t} ${n(c)}`).join(', ') || 'none';
  const beyond = Object.keys(w.mod_diff.beyond_pack);
  const rows = [
    ['World', w.world], ['GTNH', `${w.gtnh} (legacy RWG, can be migrated)`], ['Chunks', ready ? `${n(s.chunks)} in the overworld (${formats}); ${n(s.unpopulated)} not decorated yet; ${n(s.unreadable)} unreadable` : '…'],
    ['Claims', claims], ['Free space', `${gb(w.free_disk_mb)} on the world's drive`], ['Free memory', gb(w.free_ram_mb)],
    ['Mods beyond the pack', beyond.length ? beyond.join(', ') : 'none'],
  ];
  const kids = [inUse(w), h('dl', { class: 'kv' }, ...rows.flatMap(([k, v]) => [h('dt', {}, k), h('dd', {}, v)]))];
  if (beyond.length) kids.push(h('p', { class: 'small muted' }, 'If one of these adds biomes or world generation, the new terrain the tool makes will not contain its features.'));
  for (const warning of w.warnings || []) kids.push(h('p', { class: 'warn small' }, cap(warning)));
  if (w.onedrive) kids.push(h('p', { class: 'warn small' }, 'This world is in a OneDrive folder: OneDrive holds files open and can upload half-written ones. Pause syncing while you use this tool.'));
  if (s.error) kids.push(h('p', { class: 'warn' }, `${cap(s.error.reason)}. ${cap(plainAction(s.error.action))}.`));
  else if (!ready) kids.push(h('p', {}, `Reading the map of your world: ${n(s.done || 0)} of ${n(s.total || 0)} region files`), h('div', { class: 'bar' }, h('i', { style: `width:${s.total ? (100 * s.done) / s.total : 2}%` })));
  $('#world-card').replaceChildren(...kids);
  $('#world-card').hidden = false;
  $('#world-next').hidden = false;
  $('#world-continue').disabled = !ready;
}

/* Area */
function dropArea() {
  if (!S.area) return;
  clearTimeout(S.area.timer);
  if (S.area.map) S.area.map.remove();
  S.area = null;
}

/* The screen takes no click while it loads a world it does not hold yet: its controls work on S.area. */
screens.area = {
  async enter() {
    const st = S.state;
    if (!st || !st.world) { go('world'); return; }
    const world = S.world;
    const load = {};
    this.load = load;
    const current = () => this.load === load && S.world === world;
    $('#screen-area').inert = !(S.area && S.area.world === world);
    const w = await api('world');
    if (!current()) return;
    if (S.area && S.area.world === world) {
      for (const [t, cells] of Object.entries(w.claims)) S.area.claims[t] = new Set(cells.map((c) => key(...c)));
      teamList(w.claims);
      recompute();
      S.area.map.invalidateSize();
      return;
    }
    S.legend = S.legend || await api('legend');
    if (!current()) return;
    const saved = st.area || {};
    const A = {
      world, claims: {}, teams: new Set(saved.teams || []), painted: new Set((saved.painted || []).map((c) => key(...c))),
      erased: new Set((saved.erased || []).map((c) => key(...c))), margin: saved.margin || 1, cells: {}, shown: new Set(), tool: 'move', brush: 1,
      suggest: w.suggest, core: new Set(), kept: new Set(),
    };
    for (const [t, cells] of Object.entries(w.claims)) A.claims[t] = new Set(cells.map((c) => key(...c)));
    dropArea();
    S.area = A;
    $('#area-extra').hidden = true;
    $('#area-margin').value = A.margin;
    $('#area-margin-v').textContent = chunks(A.margin);
    teamList(w.claims);
    const m = baseMap($('#area-map'));
    A.map = m;
    tileLayer(w.layer, { bounds: blockBounds(w.card.bounds) }).addTo(m);
    A.layer = new CellsLayer({ pane: 'overlayPane', draw: drawArea }).addTo(m);
    legend(m, false);
    readout(m, (p) => [placeText(p, A.kept, A.core)].filter(Boolean));
    painting(m);
    $('#area-continue').disabled = true;
    recompute();
    $('#screen-area').inert = false;
    const claimed = [...Object.values(w.claims)].flat();
    const fit = cellBounds([...A.core].map(unkey), FAR / 2) || cellBounds(claimed, FAR / 2);
    if (fit) m.fitBounds(blockBounds(fit));
    else m.setView(LL(w.spawn[0], w.spawn[1]), -1);
  },
};

function teamList(claims) {
  const names = Object.keys(claims);
  $('#area-teams').hidden = !names.length;
  $('#area-team-list').replaceChildren(...names.map((t) => h('label', {}, h('input', { type: 'checkbox', checked: S.area.teams.has(t), on: { change: (e) => {
    if (e.target.checked) S.area.teams.add(t); else S.area.teams.delete(t);
    recompute();
  } } }), `Team ${t}: ${chunks(claims[t].length)}`)));
}

function recompute() {
  const A = S.area;
  const core = new Set(A.painted);
  for (const t of A.teams) for (const k of A.claims[t] || []) core.add(k);
  for (const k of A.erased) core.delete(k);
  A.core = core;
  A.kept = dilate(core, A.margin);
  redrawArea();
  clearTimeout(A.timer);
  A.timer = setTimeout(counts, COUNTS_MS);
}

let redrawQueued = false;
function redrawArea() {
  if (redrawQueued) return;
  redrawQueued = true;
  requestAnimationFrame(() => { redrawQueued = false; if (S.area && S.area.layer) S.area.layer.redraw(); });
}

function drawArea(ctx, t) {
  const A = S.area;
  const red = new Set();
  for (const layer of A.shown) for (const c of A.cells[layer] || []) red.add(key(...c));
  ctx.fillStyle = 'rgba(255,60,40,.55)';
  cellsIn(t, red, (px, pz) => ctx.fillRect(px, pz, t.s, t.s));
  ctx.fillStyle = 'rgba(255,255,255,.14)';
  cellsIn(t, A.core, (px, pz) => ctx.fillRect(px, pz, t.s, t.s));
  if (t.s >= CHUNK / 2) {
    ctx.strokeStyle = 'rgba(0,0,0,.18)';
    ctx.lineWidth = 1;
    ctx.beginPath();
    for (let i = 0; i <= t.count; i += 1) { ctx.moveTo(i * t.s + 0.5, 0); ctx.lineTo(i * t.s + 0.5, TILE); ctx.moveTo(0, i * t.s + 0.5); ctx.lineTo(TILE, i * t.s + 0.5); }
    ctx.stroke();
  }
  edges(ctx, t, A.kept, [['#fff', 2]]);
  edges(ctx, t, A.core, [['#fff', 4], ['#000', 2]]);
}

function areaBody(save) {
  const A = S.area;
  return { teams: [...A.teams], painted: [...A.painted].map(unkey), erased: [...A.erased].map(unkey), margin: A.margin, save };
}

async function counts() {
  const A = S.area;
  if (!A.core.size) {
    for (const id of ['#c-core', '#c-kept', '#c-never']) $(id).textContent = '0';
    $('#area-unreadable').hidden = true;
    $('#area-parts').hidden = true;
    $('#area-removed').replaceChildren(h('li', { class: 'muted' }, 'Choose chunks to keep: import your claims, paint on the map, or let the tool suggest them.'));
    $('#area-continue').disabled = true;
    A.cells = {};
    return;
  }
  try {
    const r = await api('area', areaBody(false));
    if (A !== S.area) return;
    A.cells = r.cells;
    A.result = r;
    $('#c-core').textContent = n(r.core);
    $('#c-kept').textContent = n(r.kept.length);
    $('#c-never').textContent = n(r.never_generated);
    removedList(r);
    $('#area-unreadable').textContent = r.unreadable || '';
    $('#area-unreadable').hidden = !r.unreadable;
    $('#area-parts').textContent = r.parts_warning || '';
    $('#area-parts').hidden = !r.parts_warning;
    $('#area-continue').disabled = !!r.unreadable;
    redrawArea();
  } catch (e) { fail(e); }
}

function removedList(r) {
  const rm = r.removed;
  const kinds = (g, k) => Object.entries(g.named || g.types).slice(0, k).map(([t, c]) => `${t} ${n(c)}`).join(', ');
  const words = (g) => Object.keys(g.named || g.types).slice(0, 3).join(', ') || 'blocks';
  const beside = rm.beside && rm.beside.count
    ? `. Their chunks also hold ${n(rm.beside.count)} more, of kinds villages and dungeons have too (${kinds(rm.beside, 3)}…); "Add these chunks" keeps them all`
    : '';
  const graves = rm.graves.map((g) => `${g.name}'s at ${g.x}, ${g.y}, ${g.z} (${n(g.stacks)} stacks of items)`).join('; ');
  const others = Object.entries(rm.other_teams_claims).map(([t, c]) => `${t} ${n(c)}`).join(', ');
  const rows = [
    [`${n(r.dropped)} overworld chunks become new terrain`, null],
    [`Probably yours (a player's name in their data, or a kind of block world generation never makes), outside the kept area and deleted at Apply: ${blocks(rm.yours.count)}${rm.yours.count ? ` (${kinds(rm.yours, 5)})` : ''}${beside}`, 'yours'],
    ...(rm.graves.length ? [[`Graves outside the kept area, with a player's items: ${graves}`, 'graves']] : []),
    [`${n(rm.portal_blocks)} nether-portal blocks`, 'portals'],
    [`Other teams' claims outside: ${others || 'none'}`, 'claims'],
    [`Players standing outside: ${rm.players.join(', ') || 'none'} (Apply moves them into the kept area)`, 'players'],
    [`Beds outside: ${rm.beds.join(', ') || 'none'}`, 'beds'],
    [rm.spawn_in_kept ? 'The world spawn is inside the kept area' : 'The world spawn falls in the new terrain (/setworldspawn in game moves it)', 'spawn'],
    [`${n(rm.worldgen.count)} ${words(rm.worldgen)}… from world generation (villages, dungeons, ruins): not counted`, null, 'muted'],
  ];
  $('#area-removed').replaceChildren(...rows.map(([text, layer, quiet]) => {
    const cells = layer ? r.cells[layer] || [] : [];
    const acts = cells.length ? h('div', { class: 'acts' },
      h('button', { class: 'small-btn', on: { click: () => { toggleShown(layer); removedList(r); } } }, S.area.shown.has(layer) ? 'Hide on the map' : 'Show on the map'),
      h('button', { class: 'small-btn', on: { click: () => addCells(layer) } }, `Add these chunks (${n(cells.length)})`)) : null;
    return h('li', { class: quiet || '' }, cells.length ? h('span', { class: 'chip' }) : null, text, acts);
  }));
}

function toggleShown(layer) {
  const A = S.area;
  if (A.shown.has(layer)) A.shown.delete(layer); else A.shown.add(layer);
  if (A.shown.has(layer)) {
    const b = cellBounds(A.cells[layer], CHUNK * 4);
    if (b) A.map.fitBounds(blockBounds(b).extend(A.map.getBounds()));
  }
  redrawArea();
}

function addCells(layer) {
  const A = S.area;
  for (const c of A.cells[layer] || []) { const k = key(...c); A.painted.add(k); A.erased.delete(k); }
  A.shown.delete(layer);
  recompute();
}

function painting(m) {
  const A = S.area;
  let down = false;
  let start = null;
  let rect = null;
  const chunkAt = (ll) => { const p = XZ(ll); return [Math.floor(p.x / CHUNK), Math.floor(p.z / CHUNK)]; };
  function brush(ll, erase) {
    const [cx, cz] = chunkAt(ll);
    const r = (A.brush - 1) / 2;
    for (let dx = -r; dx <= r; dx += 1) for (let dz = -r; dz <= r; dz += 1) mark(key(cx + dx, cz + dz), erase);
    recompute();
  }
  function mark(k, erase) {
    if (erase) { A.painted.delete(k); if (Object.values(A.claims).some((s) => s.has(k))) A.erased.add(k); } else { A.painted.add(k); A.erased.delete(k); }
  }
  m.on('mousedown', (e) => {
    if (A.tool === 'move') return;
    e.originalEvent.preventDefault();
    down = true;
    start = chunkAt(e.latlng);
    if (A.tool !== 'rect') brush(e.latlng, A.tool === 'erase');
  });
  m.on('mousemove', (e) => {
    if (!down) return;
    if (A.tool === 'rect') {
      const [cx, cz] = chunkAt(e.latlng);
      const b = [Math.min(cx, start[0]) * CHUNK, Math.min(cz, start[1]) * CHUNK, (Math.max(cx, start[0]) + 1) * CHUNK, (Math.max(cz, start[1]) + 1) * CHUNK];
      if (rect) rect.setBounds(blockBounds(b)); else rect = L.rectangle(blockBounds(b), { color: '#6cbf4f', weight: 2, interactive: false }).addTo(m);
    } else brush(e.latlng, A.tool === 'erase');
  });
  const up = (e) => {
    if (!down) return;
    down = false;
    if (A.tool === 'rect' && rect) {
      const b = rect.getBounds();
      const erase = e.shiftKey || (e.originalEvent && e.originalEvent.shiftKey);
      const x0 = Math.round(b.getWest() / CHUNK); const x1 = Math.round(b.getEast() / CHUNK);
      const z0 = Math.round(-b.getNorth() / CHUNK); const z1 = Math.round(-b.getSouth() / CHUNK);
      for (let x = x0; x < x1; x += 1) for (let z = z0; z < z1; z += 1) mark(key(x, z), erase);
      rect.remove();
      rect = null;
      recompute();
    }
  };
  m.on('mouseup', up);
  document.addEventListener('mouseup', up);
  m.on('unload', () => document.removeEventListener('mouseup', up));
}

for (const b of $$('[data-tool]')) {
  b.onclick = () => {
    for (const o of $$('[data-tool]')) o.classList.toggle('on', o === b);
    S.area.tool = b.dataset.tool;
    if (b.dataset.tool === 'move') S.area.map.dragging.enable(); else S.area.map.dragging.disable();
  };
}
$('#area-brush').onchange = (e) => { S.area.brush = Number(e.target.value); };
$('#area-margin').oninput = (e) => { S.area.margin = Number(e.target.value); $('#area-margin-v').textContent = chunks(e.target.value); recompute(); };
$('#area-claims').onclick = () => {
  const A = S.area;
  if (!A.teams.size) for (const t of Object.keys(A.claims)) A.teams.add(t);
  teamList(Object.fromEntries(Object.entries(A.claims).map(([t, s]) => [t, [...s]])));
  recompute();
  const b = cellBounds([...A.teams].flatMap((t) => [...(A.claims[t] || [])].map(unkey)), FAR / 2);
  if (b) A.map.fitBounds(blockBounds(b));
};
$('#area-suggest').onclick = () => {
  const s = S.area.suggest;
  if (!s.main.length) { fail({ reason: 'no chunk of this world was played in for an hour or more', action: 'Paint the chunks to keep on the map instead' }); return; }
  for (const c of s.main) { const k = key(...c); S.area.painted.add(k); S.area.erased.delete(k); }
  extraList();
  recompute();
  const b = cellBounds(s.main, FAR / 2);
  if (b) S.area.map.fitBounds(blockBounds(b));
};

function extraList() {
  const A = S.area;
  const extra = A.suggest.extra;
  $('#area-extra').hidden = !extra.length;
  $('#area-extra').replaceChildren(
    h('p', { class: 'muted small' }, 'You also played in these places, far from your base. They are left out unless you tick them:'),
    ...extra.map((p) => h('label', { class: 'tick' },
      h('input', { type: 'checkbox', checked: p.cells.every((c) => A.painted.has(key(...c))), on: { change: (e) => keepExtra(p, e.target.checked) } }),
      ` ${chunks(p.chunks)}, about ${n(p.blocks)} blocks ${p.direction}`)));
}

function keepExtra(p, on) {
  const A = S.area;
  for (const c of p.cells) { const k = key(...c); if (on) { A.painted.add(k); A.erased.delete(k); } else A.painted.delete(k); }
  recompute();
}
$('#area-clear').onclick = async () => {
  if (!(await confirmBox('Clear the area?', 'Every chunk you chose, painted or imported is unselected.', 'Clear'))) return;
  Object.assign(S.area, { painted: new Set(), erased: new Set(), teams: new Set() });
  $('#area-extra').hidden = true;
  teamList(Object.fromEntries(Object.entries(S.area.claims).map(([t, s]) => [t, [...s]])));
  recompute();
};
$('#area-continue').onclick = async () => {
  try {
    await api('area', areaBody(true));
    S.state = await api('state');
    S.search = {};
    go('matches');
  } catch (e) { fail(e); }
};

/* Best matches */
screens.matches = {
  async enter(jobId) {
    const st = S.state;
    if (!st || !st.world) { go('world'); return; }
    S.legend = S.legend || await api('legend');
    let auto = false;
    if (!jobId && !S.search.fresh) {
      const same = st.jobs.filter((j) => j.kind === 'search' && j.world === S.world && j.state !== 'superseded');
      if (same.length) { jobId = same[same.length - 1].id; auto = true; }
    }
    S.search = { job: jobId || null, selected: 0, cards: [], alts: false, auto };
    if (S.follow) { S.follow.close(); S.follow = null; }
    if (!S.compare) S.compare = Compare($('#m-compare'));
    S.compare.set({ left: null, right: null, empty: 'The maps of the best matches appear here as the search finds them.' });
    $('#m-metrics').hidden = true;
    $('#m-badges').hidden = true;
    $('#m-cards').replaceChildren();
    $('#m-more').hidden = true;
    $('#m-excellent').hidden = true;
    $('#m-changed').hidden = true;
    $('#m-stopped').hidden = true;
    $('#m-setup').hidden = !!jobId;
    $('#m-progress').hidden = true;
    if (!jobId) { await newSearch(); return; }
    await loadSearch();
  },
  state(st) {
    const r = st.runner;
    const searching = r && r.running && r.job && jobOf(r.job).world === S.world && (r.kind === 'search' || (r.kind === 'resume' && jobKind(r.job) === 'search'));
    if (searching && (S.search.job !== r.job || !S.follow) && !S.search.attaching) {
      S.search.job = r.job;
      S.search.attaching = true;
      loadSearch().catch(fail).finally(() => { S.search.attaching = false; });
    }
  },
};

async function setupLooks(baseName) {
  const r = await api(`look?${qs(baseName ? { base: baseName } : {})}`);
  S.look = S.look && baseName ? S.look : r;
  const sel = $('#m-base');
  if (!baseName) sel.replaceChildren(...r.biomes.map((b) => h('option', { value: b, selected: b === r.reference.name }, b)));
  sel.dataset.measured = baseName ? sel.dataset.measured : r.reference.name;
  $('#m-swatch').style.background = r.reference.colour;
  const colour = Object.fromEntries(S.legend.looks.map((l) => [l.label, l.colour]));
  $('#m-looks').replaceChildren(...r.looks.map((l) => h('label', {}, h('input', { type: 'checkbox', value: l.key, checked: l.banned, disabled: l.fixed }),
    h('span', { class: 'sw', style: `background:${colour[l.label]}` }), l.label)));
  $('#m-eula').hidden = !!S.state.eula;
  $('#m-noconfig').hidden = !!(S.state.world && S.state.world.config);
  $('#m-start').disabled = !S.state.eula;
}

async function newSearch() {
  $('#m-start').disabled = true;
  S.look = null;
  await setupLooks();
}

/* Matches found for another area than the one chosen now: Pick would keep the old area, so it is not offered. */
function changedNote(r, setup) {
  const c = r.area_changed;
  $('#m-changed').hidden = !c;
  if (!c) return;
  $('#m-changed-text').textContent = setup
    ? `You changed the area to keep since the last search (now ${chunks(c.core)} protected, ${n(c.kept)} kept): search again so the matches fit it. The earlier matches stay on the Home screen.`
    : `These matches were found for another area to keep (${chunks(r.area.core.length)} protected, ${n(r.area.kept.length)} kept); yours is now ${chunks(c.core)} protected, ${n(c.kept)} kept. Search again so the matches fit it: only a match of the area you chose can be picked.`;
  $('#m-changed-again').hidden = setup;
}

function stoppedNote(r, running) {
  const stopped = !running && !r.candidates && STOPPED_STATES.includes(r.job.state);
  $('#m-stopped').hidden = !stopped;
  if (stopped) $('#m-stopped-text').textContent = `This search ${STATE[r.job.state]}. The cards show the best matches it had found so far; continue it to finish it and pick one.`;
}

$('#m-base').onchange = (e) => setupLooks(e.target.value).catch(fail);
$('#m-start').onclick = async () => {
  if (!S.look) return;
  try {
    const sel = $('#m-base');
    const ticks = $$('#m-looks input');
    const changedBans = ticks.some((t) => t.checked !== S.look.looks.find((l) => l.key === t.value).banned) || sel.value !== sel.dataset.measured;
    const r = await step('search', {
      base: sel.value !== sel.dataset.measured ? sel.value : null,
      banned: changedBans ? ticks.filter((t) => t.checked).map((t) => t.value) : null,
      tolerance: Number($('#m-tolerance').value),
    });
    $('#m-setup').hidden = true;
    $('#m-progress').hidden = false;
    $('#m-stage').textContent = 'Starting…';
    const job = await stepJob();
    if (job) { history.replaceState(null, '', `#matches/${job}`); S.search.job = job; await loadSearch(); }
  } catch (e) { fail(e); }
};
$('#m-stop').onclick = () => api('cancel', { job: S.search.job }).then(() => { $('#m-stage').textContent = 'Stopping after the current batch, then describing the best so far…'; }).catch(fail);
$('#m-alts').onclick = () => { S.search.alts = !S.search.alts; renderCards(); };
$('#m-again').onclick = () => { S.search.fresh = true; go('matches'); S.search.fresh = false; };
$('#m-changed-again').onclick = () => $('#m-again').onclick();
$('#m-stopped-again').onclick = () => $('#m-again').onclick();
$('#m-continue').onclick = () => resume({ id: S.search.job, kind: 'search', world: S.world });

async function loadSearch() {
  const id = S.search.job;
  const r = await api(`job?${qs({ id })}`);
  S.search.data = r;
  S.search.kept = new Set((r.area ? r.area.kept : []).map((c) => key(...c)));
  S.search.core = new Set((r.area ? r.area.core : []).map((c) => key(...c)));
  S.search.started = r.job.created;
  const running = live(r.job, id);
  if (S.search.auto && r.area_changed && !running) {
    S.search.job = null;
    $('#m-progress').hidden = true;
    $('#m-setup').hidden = false;
    changedNote(r, true);
    await newSearch();
    return;
  }
  changedNote(r, false);
  stoppedNote(r, running);
  $('#m-progress').hidden = !running;
  $('#m-setup').hidden = true;
  if (r.candidates) S.search.cards = r.candidates.candidates.map((c, i) => ({ c, kind: 'final', rank: i + 1 }));
  else S.search.cards = (r.best || []).map((c, i) => ({ c, kind: 'best', rank: i + 1 }));
  $('#m-more').hidden = !r.candidates;
  renderCards();
  if (running) followSearch(id);
}

/* A resumed search first replays its finished stages: the count, the stage and the bar never go back, except on a reset. */
function searchEvent(e) {
  const s = S.search;
  if (e.step === 'job' && e.event === 'start') { if (s.runFrom) s.ran = (s.ran || 0) + s.lastT - s.runFrom; s.runFrom = e.t; }
  if (e.t) s.lastT = e.t;
  if (e.event === 'reset') { s.share = 0; s.tried = 0; }
  if (e.label && (e.progress || 0) >= (s.share || 0)) {
    $('#m-stage').textContent = e.label;
    if (e.progress) { $('#m-bar').style.width = `${100 * e.progress}%`; s.share = e.progress; }
    S.progressTitle = `Searching: ${e.label}`;
  }
  if (e.event === 'batch') s.tried = Math.max(s.tried || 0, e.cache);
  if (e.event === 'batch' || e.event === 'stage') $('#m-time').textContent = `${n(s.tried || 0)} places tried${eta()}`;
  if (e.event === 'rendered') $('#m-stage').textContent = 'Drew the land map around your base';
  if (e.event === 'reset') $('#m-stage').textContent = 'Your world or settings changed since this search last ran: it starts again';
  if (e.event === 'excellent') $('#m-excellent').hidden = false;
  if (e.step === 'search' && e.event === 'done') { $('#m-stage').textContent = 'Describing the best matches (water, Tainted Land)…'; $('#m-bar').style.width = '100%'; }
}

function followSearch(id) {
  follow(id, (events) => {
    events.forEach(searchEvent);
    const best = events.filter((e) => e.event === 'best').pop();
    if (best && !(S.search.data && S.search.data.candidates)) {
      S.search.cards = best.top.map((c, i) => ({ c, kind: 'best', rank: i + 1 }));
      renderCards();
    }
    if (events.some((e) => e.event === 'end')) loadSearch().catch(fail);
  });
}

/* Timed on the events' own clock, not this computer's: a replayed search's events are hours old. */
function eta() {
  const s = S.search, p = s.share, from = s.runFrom || s.started;
  if (!p || !from || !s.lastT) return '';
  const left = (((s.ran || 0) + s.lastT - from) * (1 - p)) / p;
  return ` · about ${Math.max(1, Math.round(left / 60))} min left`;
}

function cfgName(c) { return c.config.join('_'); }

function renderCards() {
  const s = S.search;
  const list = s.cards.slice();
  if (s.alts && s.data && s.data.candidates) list.push(...s.data.candidates.alternatives.map((c, i) => ({ c, kind: 'alt', rank: i + 1 })));
  $('#m-alts').textContent = s.alts ? 'Hide the alternatives' : 'Show more alternatives';
  $('#m-cards').replaceChildren(...list.map((item, i) => candCard(item, i)));
  $('#m-badges').hidden = !list.some((item) => item.kind === 'final');
  if (!list.length) $('#m-cards').append(h('p', { class: 'muted' }, s.data && s.data.candidates ? 'No match is on dry land (no ocean nearby, little of your land below sea level). Try a larger area, or move How picky toward tolerant.' : 'No match on dry land yet.'));
  s.list = list;
  if (list.length) selectCard(Math.min(s.selected, list.length - 1), true);
}

function badges(item) {
  const c = item.c;
  if (item.kind === 'best') return [h('span', { class: 'badge so' }, 'Best so far'), c.excellent ? h('span', { class: 'badge ex' }, 'Excellent match') : null];
  if (item.kind === 'alt') return [h('span', { class: 'badge so' }, 'Looks more like your base, rougher border')];
  return c.card.badges.map((b) => h('span', { class: `badge${b.startsWith('Excellent') ? ' ex' : ''}` }, b));
}

function candCard(item, i) {
  const c = item.c;
  const lines = item.kind === 'best' ? c.lines : c.card.lines;
  const title = item.kind === 'alt' ? `Alternative ${item.rank}` : `#${item.rank}`;
  const mini = h('div', { class: 'mini' });
  const details = h('div', { class: 'details', hidden: true }, detailsTable(c));
  const el = h('div', { class: 'cand', on: { click: () => selectCard(i) } },
    h('div', { class: 'head' }, h('span', { class: 'rank' }, title), ...badges(item)),
    h('ul', {}, ...lines.map((l) => h('li', {}, l))), mini, details,
    h('div', { class: 'acts' },
      h('button', { class: 'small-btn', on: { click: (e) => { e.stopPropagation(); details.hidden = !details.hidden; } } }, 'Details'),
      item.kind !== 'best' && !(S.search.data && S.search.data.area_changed) ? h('button', { class: 'primary small-btn', on: { click: (e) => { e.stopPropagation(); pickCard(item); } } }, 'Pick this one') : null));
  setTimeout(() => miniMap(mini, c), 0);
  return el;
}

function detailRows(c) {
  const rows = [
    ['Steps across the border', `${mostSteps(c.median)}, 90% at most ${blocks(c.p90)}`],
    ['Steps over 10 / over 30 blocks', `${pct(c.over10)} / ${pct(c.over30)}`],
    ...Object.entries(RINGS).map(([k, label]) => [`Not like your base, ${label}`, pct(c.inc[k])]),
    ['Real land that would be under water', pct(c.below_sea, 2)],
  ];
  if (c.ocean !== undefined) rows.push(['Nearest ocean', c.ocean === null ? 'none within 3,500 blocks' : `${n(Math.round(c.ocean))} blocks`]);
  if (c.taint !== undefined) rows.push(['Nearest Tainted Land', c.taint ? `${n(Math.round(c.taint.blocks))} blocks ${c.taint.direction}` : 'none within 3,500 blocks']);
  if (c.edge_biomes) rows.push(['At the edge', Object.entries(c.edge_biomes).map(([b, v]) => `${b} ${v}%`).join(', ')]);
  return rows;
}

function detailsTable(c) {
  return h('table', {}, ...detailRows(c).map(([k, v]) => h('tr', {}, h('td', {}, k), h('td', {}, v))));
}

function showMetrics(title, c) {
  $('#m-metrics-title').textContent = `${title}: the numbers`;
  $('#m-metrics-body').replaceChildren(h('div', { class: 'mgrid' }, ...detailRows(c).map(([k, v]) => h('div', {}, h('span', { class: 'muted' }, `${k}: `), v))));
  $('#m-metrics').hidden = false;
  for (const m of S.compare.maps) m.invalidateSize();
}
$('#m-metrics').addEventListener('toggle', () => { for (const m of S.compare ? S.compare.maps : []) m.invalidateSize(); });

async function layerUrl(name) {
  const r = await api(`layer?${qs({ name })}`);
  return r.url;
}

function keptBox(kept = S.search.kept) {
  return cellBounds([...kept].map(unkey), FAR / 2);
}

async function miniMap(el, c) {
  try {
    const box = keptBox();
    if (!box) return;
    const m = baseMap(el, false);
    tileLayer(await layerUrl(`cand.${S.search.job}.${cfgName(c)}`), { bounds: blockBounds(cellBounds([...S.search.kept].map(unkey), CAND_REACH)) }).addTo(m);
    outlineLayer(S.search.kept, S.search.core).addTo(m);
    m.fitBounds(blockBounds(box));
  } catch (e) { /* a card without a mini-map still reads */ }
}

/* A card selected before a newer selection or another search (Search again, a reload of the list) is left alone. */
async function selectCard(i, keepView) {
  const s = S.search;
  const current = () => S.search === s && s.selected === i && S.screen === 'matches';
  try {
    await showCard(s, i, keepView, current);
  } catch (e) {
    if (!current()) return;
    S.compare.set({ left: null, right: null, empty: `This match cannot be drawn: ${e.reason || e}.` });
    fail(e);
  }
}

async function showCard(s, i, keepView, current) {
  s.selected = i;
  $$('.cand').forEach((el, j) => el.classList.toggle('on', i === j));
  const item = s.list[i];
  if (!item) return;
  const c = item.c;
  const right = `cand.${s.job}.${cfgName(c)}`;
  const bounds = blockBounds(cellBounds([...s.kept].map(unkey), CAND_REACH));
  const layers = {
    [`now.${s.job}`]: { url: await layerUrl(`now.${s.job}`), label: 'NOW', bounds },
    [right]: { url: await layerUrl(right), label: item.kind === 'alt' ? `ALTERNATIVE ${item.rank}` : `CANDIDATE #${item.rank}`, bounds },
  };
  if (!current()) return;
  const markers = c.taint ? [{ x: c.taint.point[0], z: c.taint.point[1], colour: S.legend.taint, text: `Tainted Land, ${n(Math.round(c.taint.blocks))} blocks ${c.taint.direction}` }] : [];
  const sameRight = S.compare.right === right;
  const unchanged = keepView && sameRight && S.compare.layers[right].label === layers[right].label && S.compare.markers.length === markers.length;
  showMetrics(layers[right].label, c);
  if (!unchanged) S.compare.set({ layers, left: `now.${s.job}`, right, kept: s.kept, core: s.core, markers, heat: sameRight ? S.compare.heat : {}, fit: blockBounds(keptBox(s.kept)), view: null });
  if (!sameRight) {
    const still = () => S.screen === 'matches' && S.compare.right === right;
    heatFor({ job: s.job, config: cfgName(c) }, still).then((dots) => { if (dots && still()) S.compare.setHeat({ [right]: dots }); }).catch(fail);
  }
}

function pickCard(item) {
  const rank = item.kind === 'alt' ? S.search.data.candidates.candidates.length + item.rank : item.rank;
  S.pick = { job: S.search.job, rank, c: item.c };
  go('preview', `s:${S.search.job}:${rank}`);
}

function chosen(result, rank) {
  const n = result.candidates.length;
  if (rank <= n) return { c: result.candidates[rank - 1], label: `Candidate #${rank}` };
  return { c: result.alternatives[rank - n - 1], label: `Alternative ${rank - n}` };
}

/* Preview */
screens.preview = {
  async enter(arg) {
    $('#p-setup').hidden = true;
    S.legend = S.legend || await api('legend');
    if (!S.pcompare) S.pcompare = Compare($('#p-compare'));
    S.pcompare.set({ left: null, right: null, empty: 'The working copy appears here once the preview is built.' });
    $('#p-result').hidden = true;
    $('#p-apply').hidden = true;
    $('#p-steps').replaceChildren();
    if (arg && arg.startsWith('s:')) {
      const [, job, rank] = arg.split(':');
      const r = await api(`job?${qs({ id: job })}`);
      const { c, label } = chosen(r.candidates, Number(rank));
      S.pick = { job, rank: Number(rank), c };
      S.preview = { job: null, search: job, rank: Number(rank), c, label, area: r.area };
      choiceCard();
      const raw = (r.area.status && r.area.status.unpopulated_core) || [];
      $('#p-populated-row').hidden = !raw.length;
      $('#p-noconfig').hidden = !!(S.state.world && S.state.world.config);
      $('#p-populated').checked = false;
      $('#p-populated-text').textContent = `Mark the ${chunks(raw.length)} you protect that the game never decorated as already decorated, so the new terrain adds no trees or ores to them (a 1-byte change each; without this, the preview stops)`;
      $('#p-setup').hidden = false;
      return;
    }
    const id = arg || S.preview.job || lastPreview();
    if (!id) { go(S.pick ? 'matches' : 'home'); return; }
    await loadPreview(id);
  },
};

function lastPreview() {
  const st = S.state;
  const ps = st.jobs.filter((j) => j.kind === 'preview' && j.world === S.world && j.state !== 'superseded');
  return ps.length ? ps[ps.length - 1].id : null;
}

function choiceCard() {
  const p = S.preview;
  if (!p.c) { $('#p-choice').replaceChildren(); return; }
  $('#p-choice').replaceChildren(h('b', {}, p.label), h('ul', {}, ...((p.c.card && p.c.card.lines) || []).map((l) => h('li', {}, l))));
}

$('#p-start').onclick = async () => {
  try {
    const p = S.preview;
    await step('preview', { search: p.search, pick: p.rank, trees: $('#p-trees').checked, blend: $('#p-blend').checked, set_populated: !!($('#p-populated') && $('#p-populated').checked) });
    $('#p-setup').hidden = true;
    const job = await stepJob();
    if (job) await loadPreview(job);
  } catch (e) { fail(e); }
};
$('#p-apply').onclick = () => go('apply', S.preview.job);

async function loadPreview(id) {
  history.replaceState(null, '', `#preview/${id}`);
  const r = await api(`job?${qs({ id })}`);
  const sj = await api(`job?${qs({ id: r.args.search })}`);
  const { c, label } = chosen(sj.candidates, r.args.pick);
  S.preview = { job: id, search: r.args.search, rank: r.args.pick, c, label, area: r.area, data: r };
  choiceCard();
  $('#p-setup').hidden = true;
  const running = live(r.job, id);
  stepList(r.steps, running);
  if (running) {
    follow(id, (events) => {
      const gen = events.filter((e) => e.step === 'generate' && e.event === 'progress').pop();
      if (gen) S.progressTitle = `Preview: generating ${Math.round((100 * gen.done) / gen.total)}%`;
      if (events.some((e) => e.event === 'end')) loadPreview(id).catch(fail);
    });
    clearInterval(S.preview.timer);
    S.preview.timer = setInterval(async () => {
      if (S.screen !== 'preview' || S.preview.job !== id) { clearInterval(S.preview.timer); return; }
      const now = await api(`job?${qs({ id })}`).catch(() => null);
      if (now) stepList(now.steps, now.job.state === RUNNING);
    }, POLL_MS * 2);
    return;
  }
  clearInterval(S.preview.timer);
  if (r.job.state !== 'done') {
    $('#p-result').hidden = false;
    $('#p-result').replaceChildren(h('p', { class: 'warn' }, `This preview ${STATE[r.job.state] || r.job.state}.`),
      STOPPED_STATES.includes(r.job.state) ? h('button', { class: 'primary', on: { click: () => resume({ id, kind: 'preview', world: r.job.world }) } }, 'Continue it') : null);
    return;
  }
  previewResult(r);
  if (r.applied) { appliedNote(r); return; }
  await previewMaps(r);
}

/* After Apply the working copy is the world, or after a Roll back the world set aside: this preview cannot be applied again. */
function appliedNote(r) {
  $('#p-apply').hidden = true;
  const again = h('button', { class: 'primary', on: { click: () => go('preview', `s:${r.args.search}:${r.args.pick}`) } }, 'Build a new preview of this match');
  const verify = h('button', { class: 'primary', on: { click: () => go('verify') } }, 'Go to Verify');
  $('#p-result').append(r.rolled_back
    ? h('p', { class: 'warn' }, 'This preview was applied, then rolled back: your world is the one from before Apply again, and the migrated world was set aside. To migrate it again, build a new preview of this match.')
    : h('p', { class: 'warn' }, 'This preview is applied: your world uses it now. Its maps are on the Verify screen.'), h('div', { class: 'row' }, r.rolled_back ? again : verify));
  S.pcompare.set({ left: null, right: null, empty: r.rolled_back ? 'Build a new preview of this match to see it on the map again.' : 'Your world after Apply is on the Verify screen\'s map.' });
}

function stepList(steps, running) {
  let first = true;
  $('#p-steps').replaceChildren(...steps.map((s) => {
    const now = running && !s.done && first;
    if (!s.done) first = false;
    return h('li', { class: s.done ? 'done' : now ? 'now' : '' }, s.label + (now ? '…' : ''));
  }));
}

function seamText(s) {
  if (!s || !s.pairs) return 'no border pairs';
  return `${mostSteps(s.median)}, 90% at most ${blocks(s.p90)}, ${s.over10_pct}% over 10, the largest ${blocks(s.max)} (${n(s.pairs)} pairs)`;
}

function previewResult(r) {
  const p = r.report;
  const t = r.transplant;
  const kids = [];
  if (t) {
    const out = p ? p.outside_overworld : null;
    const changed = out && out.changed_by_transplant.length ? `, except those the migration changes on purpose (${out.changed_by_transplant.join(', ')})` : '';
    kids.push(h('p', { class: 'ok' }, `${n(p ? p.core_identical : t.core_identical)} of ${n(r.area.core.length)} protected chunks identical`
      + (out ? `; other dimensions and files identical${changed} (${n(out.files)} files outside the overworld)` : '')));
  }
  if (p) {
    const nt = p.new_terrain;
    kids.push(h('dl', { class: 'kv' },
      h('dt', {}, 'Game vs preview'), h('dd', {}, nt.columns ? `the generated ground matches the preview exactly in ${nt.exact_pct}% of columns and within 1 block in ${nt.within1_pct}%; typical difference ${blocks(nt.median)} (${n(nt.columns)} columns)` : 'not measured'),
      h('dt', {}, 'Border before smoothing'), h('dd', {}, seamText(p.seams.before.kept_border)),
      h('dt', {}, 'Border after smoothing'), h('dd', {}, seamText(p.seams.after.kept_border)),
      h('dt', {}, 'Columns reshaped'), h('dd', {}, `${n(p.columns_reshaped)} (${n(p.core_columns_changed)} in protected chunks)`),
      p.smoothing_removed ? h('dt', {}, 'Removed where reshaped') : null, p.smoothing_removed ? h('dd', {}, `${n(p.smoothing_removed.entities)} animals or other entities, ${n(p.smoothing_removed.tile_entities)} crops, flower pots or similar blocks`) : null,
      p.trees ? h('dt', {}, 'Trees') : null, p.trees ? h('dd', {}, `${n(p.trees.boot.trees)} trees replanted (${n(p.trees.log)} log blocks)`) : null,
      p.biome_blend ? h('dt', {}, 'Biome blend') : null, p.biome_blend ? h('dd', {}, `${n(p.biome_blend.columns_changed)} margin columns took the new biome`) : null));
  }
  $('#p-result').replaceChildren(...kids);
  $('#p-result').hidden = false;
  $('#p-apply').hidden = false;
}

async function previewMaps(r) {
  const s = S.preview;
  const kept = new Set(r.area.kept.map((c) => key(...c)));
  const core = new Set(r.area.core.map((c) => key(...c)));
  const bounds = blockBounds(cellBounds(r.area.kept, CAND_REACH));
  const cand = `cand.${s.search}.${cfgName(s.c)}`;
  const layers = {
    [`now.${s.search}`]: { url: await layerUrl(`now.${s.search}`), label: 'NOW', bounds },
    [cand]: { url: await layerUrl(cand), label: s.label.toUpperCase(), bounds },
  };
  const smooth = `smooth.${s.job}`;
  const smoothUrl = await layerUrl(smooth).catch(() => null);
  if (smoothUrl) layers[smooth] = { url: smoothUrl, label: 'SMOOTHED', bounds };
  S.pcompare.set({ layers, choices: Object.keys(layers), left: smoothUrl ? cand : `now.${s.search}`, right: smoothUrl ? smooth : cand, kept, core, markers: [], heat: {}, fit: blockBounds(cellBounds(r.area.kept, FAR / 2)), view: null });
  const heat = {};
  const still = () => S.screen === 'preview' && S.preview.job === s.job;
  const before = heatFor({ job: s.search, config: cfgName(s.c) }, still).then((d) => { heat[cand] = d; }).catch(fail);
  const after = smoothUrl ? heatFor({ preview: s.job }, still).then((d) => { heat[smooth] = d; }).catch(fail) : null;
  await Promise.all([before, after]);
  if (still()) S.pcompare.setHeat(heat);
}

/* Apply */
screens.apply = {
  async enter(id) {
    const pid = id || S.preview.job;
    if (!pid) { go('preview'); return; }
    S.apply = { preview: pid, ticks: {} };
    await checklist();
  },
};

async function checklist() {
  const A = S.apply;
  const r = await api(`apply/checklist?${qs({ preview: A.preview })}`);
  const remote = r.remote;
  A.data = r;
  if (A.ticks.taint_off === undefined) A.ticks.taint_off = r.taint_off_default;
  if (A.ticks.move_players === undefined) A.ticks.move_players = true;
  if (A.ticks.zip === undefined) A.ticks.zip = remote;
  const tick = (name, text) => h('label', { class: 'tick' }, h('input', { type: 'checkbox', checked: !!A.ticks[name], on: { change: (e) => { A.ticks[name] = e.target.checked; gate(); } } }), text);
  const items = [];
  if (r.ready) items.push(h('li', { class: 'bad' }, `${cap(plainReason(r.ready.reason))}. ${cap(plainAction(r.ready.action))}.`));
  const use = r.in_use;
  if (use.state === 'stopped') items.push(h('li', { class: 'okay' }, `The world is stopped: ${use.detail}.`));
  else if (use.state === 'running') items.push(h('li', { class: 'bad' }, `Your server or game is still running (${use.detail}). Stop it, then click Check again.`));
  else items.push(h('li', {}, `The tool cannot tell whether your server or game is stopped (${use.detail}).`, tick('confirm_stopped', 'My server or game is stopped')));
  A.restarts = r.kind === 'server' && !remote;
  if (A.restarts) items.push(h('li', {}, tick('auto_restart_off', 'I turned off anything that restarts my server by itself (your host panel\'s auto-restart, a restart script, or a systemd service) until Apply finishes')));
  const rm = r.removed;
  const others = Object.entries(rm.other_teams_claims || {}).map(([t, c]) => `${t} ${n(c)}`).join(', ');
  const yours = rm.yours ? rm.yours.count : rm.tile_entities_other || 0;
  const graves = (rm.graves || []).length ? ` and ${n(rm.graves.length)} grave${rm.graves.length === 1 ? '' : 's'}` : '';
  items.push(h('li', {}, `Removed for good: ${n(r.dropped)} overworld chunks become new terrain, with ${blocks(yours)} probably yours${graves}. Other teams' claims outside the kept area: ${others || 'none'}.`));
  items.push(h('li', {}, tick('move_players', `Move players standing outside the kept area to its nearest protected chunk${rm.players && rm.players.length ? ` (${rm.players.join(', ')})` : ''}`)));
  if (r.other_saves.length) {
    const away = r.kind === 'singleplayer'
      ? 'Make a copy of the instance in your launcher for this world first (Prism Launcher: right-click the instance, Copy), or:'
      : 'Move this world into a folder of its own first and search and preview it there (Apply then writes no config file and shows the 4 lines to add by hand), or:';
    items.push(h('li', { class: A.ticks.allow_shared_instance ? '' : 'bad' }, `Other worlds on the config folder ${r.config} use RWG2 too: ${r.other_saves.join(', ')}. Its rwg2.cfg is shared, so their unexplored terrain would change. ${away}`,
      tick('allow_shared_instance', `Go ahead anyway: only the unexplored terrain of ${r.other_saves.join(', ')} changes`)));
  }
  if (r.config_changed) items.push(h('li', { class: 'bad' }, `This preview was made ${r.config ? `on the config folder ${r.config}` : "on GTNH's default RWG2 settings (no config folder)"}, but the World screen now uses ${r.config_shown ? `the config folder ${r.config_shown}` : 'none'}. Search and preview again, so that Apply writes the 4 offset lines where the World screen says.`));
  else if (r.config) items.push(h('li', {}, `Apply writes the 4 offset lines into rwg2.cfg in the config folder ${r.config}.`));
  else items.push(h('li', {}, 'No config folder: Apply writes no config file. Afterwards it shows the 4 lines to add by hand to config/rwg2.cfg of the server or game instance that will run this world, before it starts. To have Apply write them instead, pick that config folder on the World screen and search again.'));
  if (r.taint) items.push(h('li', {}, tick('taint_off', `${r.config ? 'Turn off Tainted Land spreading (biome_taint_spread=0 in config/Thaumcraft.cfg; reversible)' : 'Add the line that turns off Tainted Land spreading (biome_taint_spread=0) to the lines to set by hand'}. The nearest Tainted Land is ${n(Math.round(r.taint.blocks))} blocks ${r.taint.direction}.`)));
  items.push(h('li', {}, tick('copy_backup', 'Also copy the backup to another drive (recommended)'),
    h('div', { class: 'row' }, h('button', { class: 'small-btn', on: { click: async () => { const p = await browseFor('folder', A.ticks.backup_copy || ''); if (p) { A.ticks.backup_copy = p; A.ticks.copy_backup = true; checklist().catch(fail); } } } }, 'Choose folder…'), h('span', { class: 'muted small' }, A.ticks.backup_copy || 'no folder chosen'))));
  if (remote) items.push(h('li', {}, tick('zip', 'Also write an upload zip with the right folder layout')));
  if (r.kind === 'singleplayer') items.push(h('li', {}, tick('reset_journeymap', 'Rename the JourneyMap overworld map of this world, so it is drawn again (the game must be closed; waypoints are kept)')));
  items.push(h('li', {}, `Free space on the world's drive: ${gb(r.free_disk_mb)}. Apply takes about half a minute, longer for a very large world. Your current world is kept as the backup in ${r.backup_folder}.`));
  items.push(h('li', { class: 'bad' }, 'Rolling back later means losing all play since Apply, in every dimension.'));
  $('#a-list').replaceChildren(...items);
  gate();
}

function gate() {
  const A = S.apply;
  const r = A.data;
  const stopped = r.in_use.state === 'stopped' || (r.in_use.state === 'unknown' && A.ticks.confirm_stopped);
  const ok = !r.ready && !r.config_changed && stopped && (!A.restarts || A.ticks.auto_restart_off) && (!r.other_saves.length || A.ticks.allow_shared_instance) && (!A.ticks.copy_backup || A.ticks.backup_copy);
  $('#a-go').disabled = !ok || !!(S.state.runner && S.state.runner.running);
}

$('#a-check').onclick = () => checklist().catch(fail);
$('#a-go').onclick = async () => {
  const A = S.apply;
  if (!(await confirmBox('Apply now?', 'Your world is swapped for the preview. Keep your server and game stopped until this finishes.', 'Apply'))) return;
  try {
    $('#a-go').disabled = true;
    $('#a-progress').hidden = false;
    await step('apply', {
      preview: A.preview, zip: !!A.ticks.zip, leave_players: !A.ticks.move_players, allow_shared_instance: !!A.ticks.allow_shared_instance,
      reset_journeymap: !!A.ticks.reset_journeymap, confirm_stopped: !!A.ticks.confirm_stopped, auto_restart_off: !!A.ticks.auto_restart_off,
      taint_off: !!A.ticks.taint_off, backup_copy: A.ticks.copy_backup ? A.ticks.backup_copy : null,
    });
  } catch (e) { $('#a-progress').hidden = true; fail(e); }
};

/* After Apply */
screens.after = {
  async enter() {
    if (!S.world) { go('home'); return; }
    $('#after-body').hidden = true;
    $('#after-none').hidden = true;
    let r;
    try {
      r = await api(`record?${qs({ world: S.world })}`);
    } catch (e) {
      $('#after-none').textContent = `${cap(e.reason)}. ${cap(plainAction(e.action))}.`;
      $('#after-none').hidden = false;
      return;
    }
    $('#after-body').hidden = false;
    const rec = { ...r.record, ...(S.after || {}) };
    $('#after-core').textContent = rec.core_identical !== undefined
      ? `${n(rec.core_identical)} of ${n(rec.core_present)} protected chunks are identical to your world before Apply.`
        + (rec.saved_since_swap ? ` The world was played after the swap: the ${files(rec.saved_since_swap)} your game saved since then are not compared with the preview.` : '')
      : `${n(rec.core)} protected chunks and ${n(rec.kept)} kept chunks.`;
    $('#after-lines').textContent = rec.offset_lines.join('\n');
    $('#after-byhand').hidden = !rec.by_hand;
    if (rec.by_hand) {
      const taint = rec.by_hand.taint ? ', and set I:biome_taint_spread=0 in the general section of config/Thaumcraft.cfg (it stops Tainted Land spreading)' : '';
      $('#after-byhand').textContent = `No config folder of its own was found, so Apply wrote no config file. Before the server or game that will run this world starts, set the 4 lines below in its config/rwg2.cfg, in the "expert / worldgen migration" section${taint}. They are saved in ${r.by_hand}.`;
    }
    $('#after-rules').replaceChildren(...rec.rules.slice(1).map((x) => h('li', {}, `${x}.`)));
    $('#after-map').textContent = rec.journeymap ? `The JourneyMap overworld map of this world was renamed to ${r.journeymap}, so it redraws; waypoints are kept. VisualProspecting veins stay valid.` : r.map_tip;
    const up = rec.upload;
    $('#after-upload').hidden = !up;
    if (up) $('#after-steps').replaceChildren(...up[0].slice(1).map((x) => h('li', {}, x.replace(/^\d+\. /, ''))), up[1] ? h('li', {}, `The upload zip: ${up[1]}`) : null);
    $('#after-backup').textContent = r.backup + (rec.backup_copy ? ` (also copied to ${rec.backup_copy})` : '');
    S.backupPath = r.backup;
  },
};
$('#after-copy-lines').onclick = () => copy($('#after-lines').textContent);
$('#after-open').onclick = () => api('open', { path: S.backupPath }).catch(fail);
$('#after-verify').onclick = () => go('verify');

/* Verify */
screens.verify = {
  enter() {
    const w = S.state && S.state.world && S.state.world.world === S.world ? S.state.world : null;
    $('#v-world').textContent = S.world ? `World: ${S.world}${w && w.remote ? ' (copied from another machine)' : ''}` : '';
    $('#v-run').disabled = !!(w && w.migrated && w.migrated.state === 'rolling back');
    $('#v-restart').parentElement.hidden = !!w && !(w.kind === 'server' && !w.remote);
    if (S.verify) this.show();
    afterMap(w).catch(fail);
  },
  show() {
    const r = S.verify;
    const box = $('#v-result');
    const kids = [h('h2', {}, 'config/rwg2.cfg')];
    if (r.config === null) {
      kids.push(h('p', { class: 'warn' }, 'No config folder of its own was found next to this world, so the 4 offset lines were not checked. Pick its config folder on the World screen to check them, or:'),
        h('pre', { class: 'wrap' }, r.by_hand.join('\n')));
    } else if (!Object.keys(r.wrong).length) kids.push(h('p', { class: 'ok' }, 'The 4 offset lines match the migration record.'));
    for (const [k, [want, got]] of Object.entries(r.wrong)) kids.push(h('p', { class: r.restored.includes(k) ? 'ok' : 'no' }, `${k} is ${got === null ? 'missing' : got}; the migration wrote ${want}${r.restored.includes(k) ? ' (restored)' : ''}`));
    if (Object.keys(r.wrong).length && !r.restored.length) kids.push(h('button', { class: 'primary', on: { click: () => step('verify', { world: S.world, restore: true }).catch(fail) } }, 'Restore the 4 offset lines'));
    for (const f of r.leftovers) kids.push(h('p', { class: 'no' }, `region/${f.file} holds ${chunks(f.old_chunks)} saved before Apply: the old terrain came back with an upload that merged into the old folder. Delete the world folder on the host and upload the migrated one again.`));
    const since = r.since;
    kids.push(h('h2', {}, 'Chunks generated since Apply'));
    if (!since) kids.push(h('p', { class: 'muted' }, 'Not checked: the preview of this migration is no longer in the data folder, or it did not generate the new terrain.'));
    else {
      kids.push(h('p', {}, `${n(since.new_chunks)} new chunks; ${since.new_terrain.columns ? `their ground is ${since.new_terrain.within1_pct}% within 1 block of the preview, typical difference ${blocks(since.new_terrain.median)}; ${n(since.off_chunks)} of ${n(since.compared_chunks)} chunks are off by a block or more` : 'none on the preview\'s map yet'}.`));
      if (since.off_cluster >= 4) kids.push(h('p', { class: 'no' }, `${n(since.off_cluster)} touching chunks around x ${since.off_at[0]}, z ${since.off_at[1]} do not follow the preview: they were generated while the 4 lines were missing or different. They keep that terrain; look at them in game.`));
      kids.push(h('p', {}, `Border now: ${seamText(since.seams.kept_border)}`));
    }
    if (since && since.off_cluster >= 4 && S.vcompare && S.vcompare.left) {
      S.vcompare.set({ markers: [{ x: since.off_at[0], z: since.off_at[1], colour: '#ff4b3a', text: `${n(since.off_cluster)} chunks here do not follow the preview` }], view: null });
    }
    const clean = !Object.keys(r.wrong).length && !r.leftovers.length && !(since && since.off_cluster >= 4);
    kids.push(h('p', { class: clean ? 'ok' : 'warn' }, clean ? (r.config === null ? 'No other problem found.' : 'No problem found.') : 'See the lines above.'));
    box.replaceChildren(...kids);
    box.hidden = false;
  },
};
/* The world after Apply: PREVIEW (the match as the preview drew it) | NOW, with the seam heat along the kept border now;
   BEFORE APPLY in the menus. Its scans are read once in the background. */
async function afterMap(w) {
  if (!S.vcompare) S.vcompare = Compare($('#v-compare'));
  const mine = (S.vmap = {});
  const still = () => S.screen === 'verify' && S.vmap === mine;
  const empty = (text) => S.vcompare.set({ left: null, right: null, layers: {}, choices: [], markers: [], heat: {}, empty: text });
  if (!(w && w.migrated && w.migrated.state === 'applied')) { empty('The map of your world after Apply shows here for a world this tool migrated.'); return; }
  S.legend = S.legend || await api('legend');
  for (let shown = null; still();) {
    let r;
    try {
      r = await api('aftermap');
    } catch (e) {
      if (still()) empty(`The map cannot be drawn: ${plainReason(e.reason)}. ${cap(plainAction(e.action)).replace(/\.$/, '')}.`);
      return;
    }
    if (!still()) return;
    if (!r.pending) {
      const bounds = blockBounds(cellBounds(r.kept, CAND_REACH));
      for (const l of Object.values(r.layers)) l.bounds = bounds;
      S.vcompare.set({ layers: r.layers, choices: Object.keys(r.layers), left: r.left, right: r.right, kept: new Set(r.kept.map((c) => key(...c))),
        core: new Set(r.core.map((c) => key(...c))), markers: [], heat: r.heat, fit: blockBounds(cellBounds(r.kept, FAR / 2)), view: null });
      return;
    }
    if (r.pending !== shown) { shown = r.pending; empty(`${cap(r.pending)}…`); }
    await wait(HEAT_RETRY_MS);
  }
}

$('#v-run').onclick = () => step('verify', { world: S.world }).catch(fail);
$('#v-rollback').onclick = async () => {
  const restarts = !$('#v-restart').parentElement.hidden;
  if (restarts && !$('#v-restart').checked) { fail({ reason: "the box 'I turned off anything that restarts my server by itself' is not ticked", action: "Turn off anything that restarts your server by itself, tick the box, then click Roll back again" }); return; }
  if (!(await confirmBox('Roll back?', 'The world from before Apply comes back. All play since Apply is lost, in every dimension.', 'Roll back'))) return;
  step('rollback', { world: S.world, auto_restart_off: restarts, confirm_stopped: $('#v-stopped').checked }).catch(fail);
};
$('#busy-stop').onclick = async () => {
  if (await confirmBox('Stop this step?', 'It stops at once. You can continue it later: a search, a preview or Apply from the Home screen, a Roll back from the Verify screen.', 'Stop')) api('stop', {}).catch(fail);
};

/* ---------- start ---------- */

(async function start() {
  try {
    S.state = await api('state');
    S.world = S.state.world && S.state.world.world;
    busy(S.state);
  } catch (e) { fail(e); }
  const [name, arg] = location.hash.slice(1).split('/');
  const r = S.state && S.state.runner;
  const kind = r && r.running && r.job ? (r.kind === 'resume' ? jobKind(r.job) : r.kind) : null;
  if (kind === 'search' || kind === 'preview') go(kind === 'search' ? 'matches' : 'preview', r.job);
  else if (screens[name] && (name === 'home' || name === 'world' || allowed(name, S.state))) go(name, arg);
  else go('home');
  tileStream();
  setTimeout(poll, POLL_MS);
}());
