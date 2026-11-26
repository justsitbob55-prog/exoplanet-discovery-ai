import { useEffect, useMemo, useRef, useState } from 'react'
import {
  Activity, AlertCircle, ArrowDownToLine, ArrowUpRight, Atom, Check, ChevronDown,
  CircleHelp, Database, FileChartColumn, FileUp, Gauge, LayoutDashboard, Menu,
  Orbit, Play, RotateCcw, Settings2, ShieldCheck, Sparkles,
  Telescope, X,
} from 'lucide-react'
import {
  Brush, CartesianGrid, ErrorBar, Line, LineChart, ReferenceArea, ResponsiveContainer,
  Tooltip, XAxis, YAxis,
} from 'recharts'

type Point = { time: number; flux: number; uncertainty?: [number, number]; missing?: boolean }
type Dataset = {
  id?: string
  name: string
  source: string
  points: Point[]
  demo: boolean
  processed?: boolean
  missingCount?: number
}
type Page = 'Overview' | 'Data Explorer' | 'Light Curve Analyzer' | 'Detection Workspace' | 'Candidate Explorer' | 'Model Insights' | 'Settings'

const API_BASE = import.meta.env.VITE_API_BASE_URL ?? 'http://localhost:8000/api'
const pages: { name: Page; icon: typeof LayoutDashboard; section: string }[] = [
  { name: 'Overview', icon: LayoutDashboard, section: 'WORKSPACE' },
  { name: 'Data Explorer', icon: Database, section: 'WORKSPACE' },
  { name: 'Light Curve Analyzer', icon: Activity, section: 'WORKSPACE' },
  { name: 'Detection Workspace', icon: Sparkles, section: 'DISCOVERY' },
  { name: 'Candidate Explorer', icon: Orbit, section: 'DISCOVERY' },
  { name: 'Model Insights', icon: Gauge, section: 'SYSTEM' },
  { name: 'Settings', icon: Settings2, section: 'SYSTEM' },
]

function makeDemo(): Dataset {
  const points = Array.from({ length: 240 }, (_, index) => {
    const time = index * 0.05
    const phase = ((time % 2.8) + 2.8) % 2.8
    const transit = phase > 1.33 && phase < 1.46 ? 0.009 : 0
    const ripple = Math.sin(index * 2.17) * 0.0015 + Math.cos(index * 0.71) * 0.0007
    return { time: Number(time.toFixed(3)), flux: Number((1 + ripple - transit).toFixed(5)), uncertainty: [0.0005, 0.0005] as [number, number] }
  })
  return { name: 'Synthetic transit · 12.0 d', source: 'DEMO DATA', points, demo: true }
}

function App() {
  const [page, setPage] = useState<Page>('Overview')
  const [dataset, setDataset] = useState<Dataset>(makeDemo)
  const [apiOnline, setApiOnline] = useState(false)
  const [apiMessage, setApiMessage] = useState('Checking local analysis service…')
  const [busy, setBusy] = useState(false)
  const [notice, setNotice] = useState('')
  const [menuOpen, setMenuOpen] = useState(false)
  const [smooth, setSmooth] = useState(false)
  const [showErrorBars, setShowErrorBars] = useState(true)
  const inputRef = useRef<HTMLInputElement>(null)

  useEffect(() => {
    const check = () => fetch(`${API_BASE}/health`).then((response) => {
      if (!response.ok) throw new Error('Backend returned an error')
      setApiOnline(true)
      setApiMessage('Analysis API connected')
    }).catch(() => {
      setApiOnline(false)
      setApiMessage('API offline · demo mode available')
    })
    void check()
    const interval = window.setInterval(() => void check(), 30000)
    return () => window.clearInterval(interval)
  }, [])

  const points = dataset.points
  const range = points.length ? [points[0].time, points[points.length - 1].time] : [0, 1]
  const average = useMemo(() => points.length ? points.reduce((sum, point) => sum + point.flux, 0) / points.length : 0, [points])
  async function handleUpload(file?: File) {
    if (!file) return
    setNotice('')
    const extension = file.name.split('.').pop()?.toLowerCase()
    if (!['csv', 'fits', 'fit', 'fts'].includes(extension ?? '')) {
      setNotice('Unsupported file. Choose a .csv or .fits light curve.')
      return
    }
    if (file.size > 10 * 1024 * 1024) {
      setNotice('This file exceeds the 10 MB upload limit.')
      return
    }
    setBusy(true)
    if (extension === 'csv') {
      try {
        const text = await file.text()
        const rows = text.split(/\r?\n/).filter(Boolean)
        const headers = rows[0]?.split(',').map((header) => header.trim().toLowerCase()) ?? []
        const timeColumn = headers.findIndex((header) => ['time', 'bjd', 'btjd', 'jd'].includes(header))
        const fluxColumn = headers.findIndex((header) => ['flux', 'pdcsap_flux', 'sap_flux', 'normalized_flux'].includes(header))
        if (timeColumn < 0 || fluxColumn < 0) throw new Error('CSV needs recognizable time and flux columns.')
        const parsed = rows.slice(1).map((row) => row.split(',')).map((cells) => ({
          time: Number(cells[timeColumn]), flux: Number(cells[fluxColumn]),
        })).filter((point) => Number.isFinite(point.time) && Number.isFinite(point.flux)).sort((a, b) => a.time - b.time)
        const missingCount = rows.length - 1 - parsed.length
        if (parsed.length < 10) throw new Error('At least 10 valid time/flux rows are required.')
        const form = new FormData()
        form.append('file', file)
        let uploaded: { id: string; name: string; source: string; time: number[]; flux: number[]; flux_err: number[]; metadata?: Record<string, unknown> } | undefined
        try {
          const response = await fetch(`${API_BASE}/light-curves/upload`, { method: 'POST', body: form })
          const body = await response.json()
          if (!response.ok) throw new Error(body.detail ?? 'The API rejected this file.')
          uploaded = body
        } catch (error) {
          if (apiOnline) throw error
        }
        const loadedPoints = uploaded ? uploaded.time.map((time, index) => ({ time, flux: uploaded!.flux[index], uncertainty: [uploaded!.flux_err[index], uploaded!.flux_err[index]] as [number, number] })) : parsed
        setDataset({ id: uploaded?.id, name: file.name, source: uploaded ? 'CSV upload · API validated' : 'CSV upload · local preview', points: loadedPoints, demo: false, missingCount: uploaded ? Number(uploaded.metadata?.dropped_invalid_rows ?? 0) : missingCount })
        setNotice(uploaded ? `Loaded ${loadedPoints.length.toLocaleString()} validated cadences.` : `Loaded ${loadedPoints.length.toLocaleString()} points locally. Start the API to enable preprocessing.`)
        setPage('Light Curve Analyzer')
      } catch (error) {
        setNotice(error instanceof Error ? error.message : 'Could not read this CSV file.')
      }
    } else {
      if (!apiOnline) {
        setNotice('FITS files need the local API for safe parsing. Start the backend and try again.')
      } else {
        try {
          const form = new FormData()
          form.append('file', file)
          const response = await fetch(`${API_BASE}/light-curves/upload`, { method: 'POST', body: form })
          const body = await response.json()
          if (!response.ok) throw new Error(body.detail ?? 'The API rejected this file.')
          setDataset({ id: body.id, name: file.name, source: 'FITS upload · API validated', points: body.time.map((time: number, index: number) => ({ time, flux: body.flux[index], uncertainty: [body.flux_err[index], body.flux_err[index]] as [number, number] })), demo: false, missingCount: Number(body.metadata?.dropped_invalid_rows ?? 0) })
          setNotice(`Loaded ${body.cadence_count.toLocaleString()} FITS cadences.`)
          setPage('Light Curve Analyzer')
        } catch (error) {
          setNotice(error instanceof Error ? error.message : 'Could not parse this FITS file.')
        }
      }
    }
    setBusy(false)
  }

  async function runPreprocessing() {
    if (!dataset.id || !apiOnline) {
      setNotice('Preprocessing requires a dataset uploaded through the connected API.')
      return
    }
    setBusy(true)
    setNotice('')
    try {
      const response = await fetch(`${API_BASE}/light-curves/preprocess`, {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ light_curve_id: dataset.id }),
      })
      const body = await response.json()
      if (!response.ok) throw new Error(body.detail ?? 'Preprocessing failed.')
      setDataset({ ...dataset, points: body.time.map((time: number, index: number) => ({ time, flux: body.flux[index], uncertainty: [body.flux_err[index], body.flux_err[index]] as [number, number] })), processed: true })
      setNotice(`Preprocessing complete · ${body.cadence_count.toLocaleString()} cadences retained.`)
    } catch (error) {
      setNotice(error instanceof Error ? error.message : 'Preprocessing failed.')
    } finally {
      setBusy(false)
    }
  }

  function resetAnalysis() {
    setDataset(makeDemo())
    setNotice('Analysis reset to the synthetic demonstration light curve.')
  }

  function loadDemo() {
    setDataset(makeDemo())
    setNotice('Synthetic demonstration dataset loaded. No scientific result is implied.')
  }

  function uploadButton(label = 'Upload light curve', compact = false) {
    return <button className={compact ? 'button button-outline button-small' : 'button button-primary'} onClick={() => inputRef.current?.click()} disabled={busy}>
      {busy ? <span className="spinner" /> : <FileUp size={16} />} {busy ? 'Loading…' : label}
    </button>
  }

  function chart(height = 290) {
    return <>
      {dataset.missingCount ? <div className="missing-indicator"><AlertCircle size={14} /> {dataset.missingCount} invalid or missing rows were excluded during ingestion.</div> : null}
      <div className="chart-wrap" style={{ height }}>
      <ResponsiveContainer width="100%" height="100%">
        <LineChart data={points} margin={{ top: 10, right: 12, bottom: 0, left: -18 }}>
          <CartesianGrid stroke="#203247" strokeDasharray="3 7" vertical={false} />
          <XAxis dataKey="time" type="number" domain={range} tickLine={false} axisLine={false} tick={{ fill: '#8294a8', fontSize: 11 }} tickFormatter={(value: number) => value.toFixed(1)} minTickGap={42} />
          <YAxis domain={['dataMin - 0.003', 'dataMax + 0.003']} tickLine={false} axisLine={false} tick={{ fill: '#8294a8', fontSize: 11 }} tickFormatter={(value: number) => value.toFixed(3)} width={56} />
          <Tooltip contentStyle={{ background: '#101d2b', border: '1px solid #2a4057', borderRadius: 8, color: '#edf4fb' }} labelStyle={{ color: '#91a5b8' }} labelFormatter={(value) => `Time ${Number(value).toFixed(3)} d`} formatter={(value) => [Number(value).toFixed(5), 'Normalized flux']} />
          {dataset.demo && points.length > 10 && <ReferenceArea x1={1.33} x2={1.46} fill="#ec6b8c" fillOpacity={0.10} />}
          <Line type={smooth ? 'monotone' : 'linear'} dataKey="flux" stroke="#70d5d0" strokeWidth={1.7} dot={false} activeDot={{ r: 4, fill: '#f1b96d', stroke: '#08131f', strokeWidth: 2 }} isAnimationActive={false}>
            {showErrorBars && <ErrorBar dataKey="uncertainty" direction="y" width={2} stroke="#70d5d0" strokeOpacity={0.35} strokeWidth={0.7} />}
          </Line>
          {points.length > 40 && <Brush dataKey="time" height={17} stroke="#5d9c9b" fill="#0c1a28" travellerWidth={7} tickFormatter={(value: number) => value.toFixed(1)} />}
        </LineChart>
      </ResponsiveContainer>
      </div>
    </>
  }

  function pageContent() {
    switch (page) {
      case 'Overview': return <>
        <div className="welcome-row"><div><div className="eyebrow">MISSION CONTROL / OVERVIEW</div><h1>Good evening, explorer.</h1><p className="subhead">Your light-curve lab is ready for its next observation.</p></div><div className="welcome-actions">{uploadButton('Import data')}</div></div>
        <div className="demo-ribbon"><span className="demo-dot" /> DEMO DATA <span>Dashboard values below are illustrative and are not scientific results.</span></div>
        <section className="stat-grid">
          <Stat label="Light curves processed" value="—" note="No real datasets yet" icon={Activity} tone="teal" />
          <Stat label="Candidates identified" value="—" note="ML detector not available" icon={Sparkles} tone="violet" />
          <Stat label="Confirmed exoplanets" value="—" note="No validated findings" icon={Orbit} tone="amber" />
          <Stat label="Data sources" value="2" note="CSV · FITS supported" icon={Database} tone="blue" />
        </section>
        <section className="overview-grid">
          <div className="panel chart-panel"><div className="panel-heading"><div><div className="eyebrow">CURRENT PREVIEW</div><h2>Light curve</h2></div><span className={`tag ${dataset.demo ? 'tag-demo' : 'tag-live'}`}>{dataset.demo ? 'SYNTHETIC' : 'UPLOADED'}</span></div><div className="chart-title-line"><span>{dataset.name}</span><span className="mono">{points.length} points</span></div>{chart(270)}<div className="chart-legend"><span><i className="legend-line" /> Normalized flux</span>{dataset.demo && <span><i className="legend-window" /> Simulated transit</span>}<button className="text-button" onClick={() => setPage('Light Curve Analyzer')}>Open analyzer <ArrowUpRight size={13} /></button></div></div>
          <div className="panel activity-panel"><div className="panel-heading"><div><div className="eyebrow">RECENT ACTIVITY</div><h2>Analysis log</h2></div><button className="icon-button" title="Refresh status" onClick={() => window.location.reload()}><RotateCcw size={15} /></button></div><div className="activity-list"><ActivityItem icon={Check} color="green" title={dataset.demo ? 'Demo workspace initialized' : 'Dataset loaded into workspace'} time={dataset.demo ? 'Just now' : 'Ready'} detail={dataset.demo ? 'Synthetic light curve ready to explore' : `${points.length.toLocaleString()} uploaded cadences available`} /><ActivityItem icon={Database} color="blue" title="Data services configured" time="Ready" detail="CSV and FITS upload supported" /><ActivityItem icon={AlertCircle} color="amber" title="Detection model unavailable" time="Pending" detail="No predictions will be generated" /></div><button className="button button-quiet activity-link" onClick={() => setPage('Detection Workspace')}>View detection workspace <ArrowUpRight size={14} /></button></div>
        </section>
        <section className="lower-grid"><div className="panel source-panel"><div className="panel-heading"><div><div className="eyebrow">INGESTION</div><h2>Data sources</h2></div><button className="text-button" onClick={() => setPage('Data Explorer')}>Explore <ArrowUpRight size={13} /></button></div><SourceRow icon={FileChartColumn} title="CSV light curves" detail="Time / flux columns" status="READY" /><SourceRow icon={Telescope} title="FITS observations" detail="Astronomy table extensions" status="READY" /></div><div className="panel status-panel"><div className="panel-heading"><div><div className="eyebrow">SYSTEM STATUS</div><h2>Services</h2></div><span className={`status-pill ${apiOnline ? 'online' : 'offline'}`}><i />{apiOnline ? 'CONNECTED' : 'API OFFLINE'}</span></div><div className="service-row"><span>Analysis API</span><span className="service-result">{apiMessage}</span></div><div className="service-row"><span>Detection model</span><span className="service-result muted">Not implemented</span></div><div className="service-row"><span>Dataset storage</span><span className="service-result muted">Session memory only</span></div></div></section>
      </>
      case 'Data Explorer': return <><PageHeader eyebrow="WORKSPACE / DATA EXPLORER" title="Explore your observations" subtitle="Upload a light curve and inspect its contents before analysis." actions={<>{uploadButton('Upload CSV / FITS') }<button className="button button-outline" onClick={loadDemo}><Sparkles size={15} /> Load demo</button></>} /><div className="explorer-grid"><div className="panel upload-panel"><div className="upload-symbol"><ArrowDownToLine size={22} /></div><h2>Bring in an observation</h2><p>CSV and FITS files up to 10 MB. CSV requires recognizable time and flux columns.</p>{uploadButton('Choose file')}<div className="format-list"><span>.CSV</span><span>.FITS</span><span>.FIT</span><span>.FTS</span></div>{!apiOnline && <div className="inline-warning"><AlertCircle size={15} /> CSV preview works offline; FITS parsing needs the API.</div>}</div><div className="panel dataset-info"><div className="panel-heading"><div><div className="eyebrow">SELECTED DATASET</div><h2>{dataset.name}</h2></div><span className={`tag ${dataset.demo ? 'tag-demo' : 'tag-live'}`}>{dataset.demo ? 'DEMO DATA' : 'UPLOADED'}</span></div><div className="info-grid"><Info label="Source" value={dataset.source} /><Info label="Valid cadences" value={points.length.toLocaleString()} /><Info label="Time span" value={`${(range[1] - range[0]).toFixed(2)} d`} /><Info label="Time column" value="time" /><Info label="Flux column" value="flux" /><Info label="Missing values" value={dataset.missingCount ? String(dataset.missingCount) : 'None detected'} /></div><div className="table-title">DATA PREVIEW <span>First 8 rows</span></div><DataTable points={points.slice(0, 8)} /></div></div></>
      case 'Light Curve Analyzer': return <><PageHeader eyebrow="WORKSPACE / ANALYSIS" title="Light curve analyzer" subtitle="Inspect normalized flux over time. The demo signal is synthetic, not an observation." actions={<>{uploadButton('Upload light curve')}<button className="button button-outline" onClick={loadDemo}><Sparkles size={15} /> Load demo</button></>} /><div className="panel analyzer-panel"><div className="panel-heading analyzer-heading"><div><div className="eyebrow">{dataset.demo ? 'SYNTHETIC DEMONSTRATION' : 'UPLOADED OBSERVATION'}</div><h2>{dataset.name}</h2></div><div className="chart-tools"><span className="tag tag-demo">{dataset.demo ? 'DEMO DATA' : dataset.processed ? 'PREPROCESSED' : 'RAW INPUT'}</span><button className="icon-button" title="Reset analysis" onClick={resetAnalysis}><RotateCcw size={15} /></button></div></div><div className="analyzer-metrics"><Metric label="Cadences" value={points.length.toLocaleString()} /><Metric label="Time baseline" value={`${(range[1] - range[0]).toFixed(2)} d`} /><Metric label="Mean flux" value={average.toFixed(5)} /><Metric label="Quality" value={dataset.demo ? 'Synthetic' : dataset.processed ? 'Processed' : 'Unprocessed'} /></div>{chart(410)}<div className="chart-legend"><span><i className="legend-line" /> Normalized flux</span>{dataset.demo && <span><i className="legend-window" /> Simulated dip</span>}<span className="chart-help"><CircleHelp size={13} /> Hover for values</span></div><div className="analysis-controls"><div><button className="button button-outline" onClick={runPreprocessing} disabled={busy}>{busy ? <span className="spinner" /> : <Activity size={15} />}{busy ? 'Processing…' : 'Run preprocessing'}</button><button className="button button-quiet" onClick={resetAnalysis}><RotateCcw size={14} /> Reset</button></div><span className="control-note">{apiOnline ? 'Preprocessing runs on the local API.' : 'Connect the API to enable preprocessing.'}</span></div></div></>
      case 'Detection Workspace': return <><PageHeader eyebrow="DISCOVERY / DETECTION WORKSPACE" title="Detection workspace" subtitle="Prepare an observation for a future model-backed search." actions={uploadButton('Select dataset')} /><div className="workspace-grid"><div className="panel workflow-panel"><div className="panel-heading"><div><div className="eyebrow">ANALYSIS PIPELINE</div><h2>Workflow progress</h2></div><span className="status-pill offline"><i /> MODEL PENDING</span></div><div className="workflow-step done"><span className="step-marker"><Check size={14} /></span><div><b>Dataset selected</b><small>{dataset.name}</small></div><span>COMPLETE</span></div><div className={`workflow-step ${dataset.processed ? 'done' : 'current'}`}><span className="step-marker">{dataset.processed ? <Check size={14} /> : '2'}</span><div><b>Preprocessing</b><small>{dataset.processed ? 'Quality checks and normalization complete' : 'Run preprocessing from the analyzer'}</small></div><span>{dataset.processed ? 'COMPLETE' : 'READY'}</span></div><div className="workflow-step disabled"><span className="step-marker">3</span><div><b>ML detection</b><small>Detection model is not implemented</small></div><span>UNAVAILABLE</span></div><div className="workflow-step disabled"><span className="step-marker">4</span><div><b>Candidate review</b><small>Waiting for a validated model output</small></div><span>LOCKED</span></div><button className="button button-primary workflow-action" onClick={() => setPage('Light Curve Analyzer')}><Activity size={15} /> Open analyzer</button></div><div className="panel results-placeholder"><div className="empty-orbit"><Orbit size={31} /></div><div className="eyebrow">RESULTS PANEL</div><h2>ML detection model is not yet available.</h2><p>No candidate predictions are generated in this workspace. The current light curve can be inspected and preprocessed only.</p><div className="candidate-empty-line"><span>Candidate results</span><span>Not available</span></div></div></div></>
      case 'Candidate Explorer': return <><PageHeader eyebrow="DISCOVERY / CANDIDATES" title="Candidate explorer" subtitle="Scientifically validated findings will appear here when produced by an implemented model." /><div className="panel candidate-state"><div className="candidate-orbit"><Orbit size={38} /></div><span className="tag tag-outline">NO VALIDATED CANDIDATES</span><h2>No candidate results yet</h2><p>The demo light curve contains a simulated transit-like dip for visualization only. It is not a detection and has no orbital or probability values.</p><button className="button button-outline" onClick={() => setPage('Detection Workspace')}>Open detection workspace <ArrowUpRight size={14} /></button></div><div className="panel candidate-table"><div className="panel-heading"><div><div className="eyebrow">CANDIDATE REGISTER</div><h2>Validated findings</h2></div><span className="mono muted">0 records</span></div><div className="table-scroll"><table><thead><tr><th>IDENTIFIER</th><th>STATUS</th><th>ORBITAL PERIOD</th><th>TRANSIT DEPTH</th><th>SNR</th><th>PROBABILITY</th></tr></thead><tbody><tr><td colSpan={6} className="empty-table">No model-generated candidates to display.</td></tr></tbody></table></div></div></>
      case 'Model Insights': return <><PageHeader eyebrow="SYSTEM / MODEL INSIGHTS" title="Model insights" subtitle="Transparent status for the detection system. No training or evaluation is claimed." /><div className="model-status-banner"><div className="model-status-icon"><Atom size={22} /></div><div><div className="eyebrow">IMPLEMENTATION STATUS</div><h2>Detection model not implemented</h2><p>Training has not started. Evaluation metrics will appear only after a real model is trained and evaluated.</p></div><span className="status-pill offline"><i /> UNAVAILABLE</span></div><div className="insight-grid"><Insight title="Training status" value="Not started" detail="No training dataset or run recorded" icon={Play} /><Insight title="Evaluation metrics" value="Not available" detail="No model evaluation has been performed" icon={FileChartColumn} /><Insight title="Model registry" value="Empty" detail="No checkpoints are registered" icon={Database} /></div><div className="panel future-panel"><div className="panel-heading"><div><div className="eyebrow">EVALUATION SURFACE</div><h2>Performance visualizations</h2></div><span className="tag tag-outline">AWAITING MODEL</span></div><div className="placeholder-chart"><div className="placeholder-grid" /><div className="placeholder-message"><Activity size={22} /><span>ROC, precision-recall, and confusion matrix views will populate from measured evaluation results.</span></div></div></div></>
      case 'Settings': return <><PageHeader eyebrow="SYSTEM / PREFERENCES" title="Workspace settings" subtitle="Local display and data preferences." /><div className="settings-layout"><div className="panel settings-panel"><div className="panel-heading"><div><div className="eyebrow">PREPROCESSING</div><h2>Analysis preferences</h2></div><Activity size={17} className="muted-icon" /></div><Setting title="Outlier filtering" detail="Robust MAD threshold from backend configuration" value="5 sigma" /><Setting title="Detrending" detail="Polynomial baseline removal" value="Enabled" /><Setting title="Normalization" detail="Scale flux around the configured percentile" value="95th percentile" /></div><div className="panel settings-panel"><div className="panel-heading"><div><div className="eyebrow">VISUALIZATION</div><h2>Chart preferences</h2></div><Gauge size={17} className="muted-icon" /></div><ToggleSetting title="Smooth chart line" detail="Use a curved interpolation between measured points" value={smooth} onChange={setSmooth} /><ToggleSetting title="Show uncertainty indicators" detail="Display uncertainty overlays when available" value={showErrorBars} onChange={setShowErrorBars} /><div className="setting-row"><div><b>Dataset preference</b><small>Current selection remains available for this session</small></div><span className="setting-value">Session only</span></div></div><div className="panel settings-panel app-info"><div className="panel-heading"><div><div className="eyebrow">APPLICATION</div><h2>About this workspace</h2></div><ShieldCheck size={17} className="muted-icon" /></div><div className="service-row"><span>Project</span><span className="service-result">Exoplanet Discovery AI</span></div><div className="service-row"><span>Version</span><span className="service-result mono">0.1.0 · preview</span></div><div className="service-row"><span>API</span><span className="service-result">{apiMessage}</span></div></div></div></>
    }
  }

  return <div className="app-shell">
    <input ref={inputRef} className="visually-hidden" type="file" accept=".csv,.fits,.fit,.fts,text/csv,application/fits" onChange={(event) => { void handleUpload(event.target.files?.[0]); event.target.value = '' }} />
    <aside className={`sidebar ${menuOpen ? 'sidebar-open' : ''}`}>
      <div className="brand"><div className="brand-mark"><Atom size={21} strokeWidth={1.8} /></div><div><b>EXOPLANET</b><span>DISCOVERY AI</span></div><button className="sidebar-close icon-button" onClick={() => setMenuOpen(false)} title="Close navigation"><X size={17} /></button></div>
      <div className="workspace-select"><div className="workspace-avatar">OD</div><div><b>Orbital Dynamics</b><small>Research workspace</small></div><ChevronDown size={14} /></div>
      <nav className="navigation">{(['WORKSPACE', 'DISCOVERY', 'SYSTEM'] as const).map((section) => <div key={section}><div className="nav-section">{section}</div>{pages.filter((item) => item.section === section).map((item) => { const Icon = item.icon; return <button key={item.name} className={`nav-item ${page === item.name ? 'active' : ''}`} onClick={() => { setPage(item.name); setMenuOpen(false) }}><Icon size={17} strokeWidth={1.8} /><span>{item.name}</span>{item.name === 'Detection Workspace' && <span className="nav-soon">SOON</span>}</button> })}</div>)}</nav>
      <div className="sidebar-bottom"><div className="backend-card"><span className={`connection-dot ${apiOnline ? 'on' : ''}`} /><div><b>{apiOnline ? 'API connected' : 'Demo environment'}</b><small>{apiOnline ? 'Local service ready' : 'Backend disconnected'}</small></div><button className="icon-button" title="Check service status" onClick={() => fetch(`${API_BASE}/health`).then((response) => setApiOnline(response.ok)).catch(() => setApiOnline(false))}><RotateCcw size={13} /></button></div><div className="profile-row"><div className="profile-avatar">SR</div><div><b>S. Researcher</b><small>Local session</small></div><button className="icon-button" title="Open settings" onClick={() => setPage('Settings')}><Settings2 size={15} /></button></div></div>
    </aside>
    {menuOpen && <button className="mobile-scrim" aria-label="Close navigation" onClick={() => setMenuOpen(false)} />}
    <main className="main-area"><header className="topbar"><div className="topbar-left"><button className="menu-button icon-button" onClick={() => setMenuOpen(true)} title="Open navigation"><Menu size={19} /></button><div className="breadcrumb"><span>Workspace</span><span className="breadcrumb-slash">/</span><b>{page}</b></div></div><div className="topbar-right"><span className="topbar-status"><i className={apiOnline ? 'connected' : ''} />{apiOnline ? 'API connected' : 'Demo mode'}</span><button className="help-button" title="Application information" onClick={() => setPage('Model Insights')}><CircleHelp size={17} /></button><div className="topbar-avatar">SR</div></div></header>
      <div className="page-content">{pageContent()}<footer className="page-footer"><span>EXOPLANET DISCOVERY AI <i /> LOCAL RESEARCH PREVIEW</span><span>No scientific results are generated by this interface.</span></footer></div>
    </main>
    {notice && <div className={`toast ${notice.includes('Unsupported') || notice.includes('exceeds') || notice.includes('requires') || notice.includes('needs') || notice.includes('failed') ? 'toast-error' : ''}`} role="status"><span>{notice.includes('Unsupported') || notice.includes('exceeds') || notice.includes('requires') || notice.includes('needs') || notice.includes('failed') ? <AlertCircle size={16} /> : <Check size={16} />}</span>{notice}<button onClick={() => setNotice('')} title="Dismiss"><X size={14} /></button></div>}
  </div>
}

function Stat({ label, value, note, icon: Icon, tone }: { label: string; value: string; note: string; icon: typeof Activity; tone: string }) {
  return <div className="stat-card"><div className={`stat-icon ${tone}`}><Icon size={17} /></div><div className="stat-label">{label}</div><div className="stat-value">{value}</div><div className="stat-note">{note}</div><span className="stat-indicator">—</span></div>
}
function PageHeader({ eyebrow, title, subtitle, actions }: { eyebrow: string; title: string; subtitle: string; actions?: React.ReactNode }) {
  return <div className="page-header"><div><div className="eyebrow">{eyebrow}</div><h1>{title}</h1><p className="subhead">{subtitle}</p></div>{actions && <div className="page-header-actions">{actions}</div>}</div>
}
function ActivityItem({ icon: Icon, color, title, time, detail }: { icon: typeof Activity; color: string; title: string; time: string; detail: string }) {
  return <div className="activity-item"><div className={`activity-icon ${color}`}><Icon size={14} /></div><div className="activity-copy"><b>{title}</b><small>{detail}</small></div><time>{time}</time></div>
}
function SourceRow({ icon: Icon, title, detail, status }: { icon: typeof Activity; title: string; detail: string; status: string }) {
  return <div className="source-row"><div className="source-icon"><Icon size={16} /></div><div><b>{title}</b><small>{detail}</small></div><span className="source-ready"><i />{status}</span></div>
}
function Info({ label, value }: { label: string; value: string }) {
  return <div className="info-item"><span>{label}</span><b>{value}</b></div>
}
function DataTable({ points }: { points: Point[] }) {
  return <div className="table-scroll"><table><thead><tr><th>ROW</th><th>TIME</th><th>FLUX</th><th>QUALITY</th></tr></thead><tbody>{points.map((point, index) => <tr key={`${point.time}-${index}`}><td className="mono">{String(index + 1).padStart(3, '0')}</td><td className="mono">{point.time.toFixed(4)}</td><td className="mono">{point.flux.toFixed(6)}</td><td><span className="quality-ok"><i />VALID</span></td></tr>)}</tbody></table></div>
}
function Metric({ label, value }: { label: string; value: string }) {
  return <div className="analyzer-metric"><span>{label}</span><b>{value}</b></div>
}
function Insight({ title, value, detail, icon: Icon }: { title: string; value: string; detail: string; icon: typeof Activity }) {
  return <div className="panel insight-card"><div className="insight-icon"><Icon size={17} /></div><span>{title}</span><b>{value}</b><small>{detail}</small></div>
}
function Setting({ title, detail, value }: { title: string; detail: string; value: string }) {
  return <div className="setting-row"><div><b>{title}</b><small>{detail}</small></div><span className="setting-value">{value}</span></div>
}
function ToggleSetting({ title, detail, value, onChange }: { title: string; detail: string; value: boolean; onChange: (value: boolean) => void }) {
  return <div className="setting-row"><div><b>{title}</b><small>{detail}</small></div><button className={`toggle ${value ? 'toggle-on' : ''}`} role="switch" aria-checked={value} aria-label={title} onClick={() => onChange(!value)}><i /></button></div>
}

export default App