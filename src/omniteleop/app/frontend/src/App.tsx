import { useState, useEffect } from 'react'
import dexmateLogo from '/assets/Dexmate_logo.png'
import { Settings, Moon, Sun } from 'lucide-react'
import { useStateStream } from './hooks/useStateStream'
import { TeleopControls } from './components/TeleopControls'
import { StatusPanel } from './components/StatusPanel'
import { CameraStream } from './components/CameraStream'
import { DebugPanel } from './components/DebugPanel'
import { Card, CardContent, CardHeader, CardTitle } from './components/ui/card'
import { type TeleopLaunchConfig } from './components/TeleopControls'

const DEFAULT_API = 'http://localhost:5006'

type SidePanel = 'settings' | 'debug'

function loadPersisted(key: string, fallback: string) {
  try { return localStorage.getItem(key) ?? fallback } catch { return fallback }
}
function persist(key: string, value: string) {
  try { localStorage.setItem(key, value) } catch { /* noop */ }
}

export default function App() {
  const [apiUrl, setApiUrl]     = useState(() => loadPersisted('apiUrl', DEFAULT_API))
  const [apiInput, setApiInput] = useState(apiUrl)
  const [loading, setLoading]   = useState(false)
  const [actionError, setActionError] = useState<string | null>(null)
  const [sidePanel, setSidePanel] = useState<SidePanel | null>(null)
  const [dark, setDark] = useState(() => loadPersisted('theme', 'light') === 'dark')

  useEffect(() => {
    document.documentElement.classList.toggle('dark', dark)
    persist('theme', dark ? 'dark' : 'light')
  }, [dark])

  const [showDiagnosis, setShowDiagnosis] = useState(false)

  const { state, connected } = useStateStream(apiUrl)
  const {
    controlState,
    isRecording,
    episodeId,
    softwareEstop,
    recordMode,
    teleopRunning,
    errorState,
    activeComponents,
  } = state

  // Auto-open diagnosis modal when state transitions to DIAGNOSIS
  useEffect(() => {
    if (controlState === 'DIAGNOSIS') setShowDiagnosis(true)
    else setShowDiagnosis(false)
  }, [controlState])


  function applyApiUrl(url: string) {
    let trimmed = url.trim().replace(/\/$/, '')
    // If no protocol is given, assume http:// so WebSocket conversion works correctly.
    if (trimmed && !/^https?:\/\//i.test(trimmed)) {
      trimmed = `http://${trimmed}`
    }
    // If no port is present, append the default backend port.
    try {
      const parsed = new URL(trimmed)
      if (!parsed.port) trimmed = `${parsed.protocol}//${parsed.hostname}:5006${parsed.pathname === '/' ? '' : parsed.pathname}`
    } catch { /* invalid URL – pass through and let the connection fail gracefully */ }
    setApiUrl(trimmed)
    setApiInput(trimmed)
    persist('apiUrl', trimmed)
  }

  async function api(method: string, path: string, body?: unknown) {
    setLoading(true)
    setActionError(null)
    try {
      const res = await fetch(`${apiUrl}${path}`, {
        method,
        headers: body ? { 'Content-Type': 'application/json' } : {},
        body: body ? JSON.stringify(body) : undefined,
      })
      if (!res.ok) {
        const text = await res.text()
        throw new Error(`${res.status}: ${text}`)
      }
    } catch (e) {
      setActionError(e instanceof Error ? e.message : String(e))
    } finally {
      setLoading(false)
    }
  }

  const handleTeleopStart  = (cfg: TeleopLaunchConfig) => api('POST', '/teleop/start', cfg)
  const handleTeleopStop   = () => api('POST', '/teleop/stop')
  const handleRecordStart  = () => api('POST', '/record/start', { metadata: {} })
  const handleRecordSave   = () => api('POST', '/record/stop',  { is_success: true })
  const handleRecordDiscard= () => api('POST', '/record/stop',  { is_success: false })
  const handleEstop        = () => api('POST', '/robots/estop')
  const handleClearEstop   = () => api('DELETE', '/robots/estop')

  const cameraActive = controlState === 'ACTIVE' || controlState === 'RECORD' || controlState === 'ALIGN'

  return (
    <div className="min-h-screen bg-muted dark:bg-background">

      {/* ---- Top bar ---- */}
      <header className="sticky top-0 z-20 bg-white dark:bg-muted border-b border-border">
        <div className="max-w-screen-xl mx-auto px-6 h-12 flex items-center gap-3">
          {/* Logo + title */}
          <img
            src={dexmateLogo}
            alt="Dexmate"
            className="h-7 w-auto shrink-0"
          />
          <div className="flex flex-col leading-none">
            <span className="font-semibold text-sm tracking-tight">OmniTeleop</span>
          </div>

          <div className="ml-auto flex items-center gap-3">
            <span
              title={connected ? 'Backend connected' : 'Backend disconnected'}
              className={`h-2 w-2 rounded-full ${connected ? 'bg-emerald-500' : 'bg-red-400'}`}
            />
            {/* Dark mode toggle */}
            <button
              onClick={() => setDark(d => !d)}
              className="p-1.5 rounded hover:bg-muted transition-colors"
              title={dark ? 'Switch to light mode' : 'Switch to dark mode'}
            >
              {dark ? <Sun className="h-4 w-4 text-muted-foreground" /> : <Moon className="h-4 w-4 text-muted-foreground" />}
            </button>
            <div className="flex items-center rounded-md border border-border overflow-hidden">
              <button
                onClick={() => setSidePanel(p => p === 'settings' ? null : 'settings')}
                className={`p-1.5 transition-colors ${sidePanel === 'settings' ? 'bg-muted' : 'hover:bg-muted'}`}
                aria-label="Settings"
                title="Settings"
              >
                <Settings className="h-4 w-4 text-muted-foreground" />
              </button>
              <button
                onClick={() => setSidePanel(p => p === 'debug' ? null : 'debug')}
                className={`px-2.5 h-7 text-xs font-mono transition-colors border-l border-border ${sidePanel === 'debug' ? 'bg-muted text-foreground' : 'text-muted-foreground hover:bg-muted'}`}
                aria-label="Debug"
                title="Debug logs"
              >
                debug
              </button>
            </div>
          </div>
        </div>
      </header>

      <main className="max-w-screen-xl mx-auto px-6 py-6 space-y-4">

        {/* ---- Settings / Debug panel ---- */}
        {sidePanel === 'settings' && (
          <Card className="border-dashed">
            <CardHeader><CardTitle>Settings</CardTitle></CardHeader>
            <CardContent>
              <div className="space-y-1 max-w-sm">
                <label className="text-xs font-medium text-muted-foreground">Backend URL</label>
                <div className="flex gap-2">
                  <input
                    className="flex-1 h-8 px-2.5 rounded-md border border-input bg-background text-sm focus:outline-none focus:ring-1 focus:ring-ring"
                    value={apiInput}
                    onChange={e => setApiInput(e.target.value)}
                    onKeyDown={e => e.key === 'Enter' && applyApiUrl(apiInput)}
                    placeholder="https://localhost:5006"
                  />
                  <button
                    onClick={() => applyApiUrl(apiInput)}
                    className="px-3 h-8 rounded-md bg-primary text-primary-foreground text-xs font-medium hover:bg-primary/90"
                  >
                    Apply
                  </button>
                </div>
              </div>
            </CardContent>
          </Card>
        )}
        {/* ---- Error banner ---- */}
        {actionError && (
          <div className="flex items-start gap-2 px-4 py-3 bg-red-50 border border-red-200 rounded-lg text-sm text-red-700">
            <span className="font-medium">Error:</span>
            <span className="flex-1">{actionError}</span>
            <button onClick={() => setActionError(null)} className="ml-2 text-red-400 hover:text-red-600">✕</button>
          </div>
        )}

        {/* ---- Main 3-column grid ---- */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-4 items-center">

          {/* Left column */}
          <div className="flex flex-col gap-4 lg:col-span-1">
            {/* E-Stop at top of left column */}
            {!softwareEstop ? (
              <button onClick={handleEstop} disabled={loading}
                className="w-full py-2.5 rounded-lg bg-red-600 hover:bg-red-700 text-white font-bold text-sm tracking-wide flex items-center justify-center gap-2 shadow-md transition-colors">
                <svg className="h-4 w-4" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5"><path d="M12 9v4m0 4h.01M10.29 3.86L1.82 18a2 2 0 001.71 3h16.94a2 2 0 001.71-3L13.71 3.86a2 2 0 00-3.42 0z"/></svg>
                Emergency Stop
              </button>
            ) : (
              <button onClick={handleClearEstop} disabled={loading}
                className="w-full py-2.5 rounded-lg border-2 border-emerald-500 text-emerald-600 dark:text-emerald-400 font-bold text-sm tracking-wide flex items-center justify-center gap-2 hover:bg-emerald-50 dark:hover:bg-emerald-950 transition-colors">
                <svg className="h-4 w-4" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5"><path d="M9 12l2 2 4-4m5.618-4.016A11.955 11.955 0 0112 2.944a11.955 11.955 0 01-8.618 3.04A12.02 12.02 0 003 9c0 5.591 3.824 10.29 9 11.622 5.176-1.332 9-6.03 9-11.622 0-1.042-.133-2.052-.382-3.016z"/></svg>
                Clear E-Stop
              </button>
            )}
            <StatusPanel
              controlState={controlState}
              teleopRunning={teleopRunning}
              recordMode={recordMode}
              softwareEstop={softwareEstop}
              activeComponents={activeComponents}
            />
            <TeleopControls
              controlState={controlState}
              teleopRunning={teleopRunning}
              isRecording={isRecording}
              episodeId={episodeId}
              softwareEstop={softwareEstop}
              recordMode={recordMode}
              loading={loading}
              onTeleopStart={handleTeleopStart}
              onTeleopStop={handleTeleopStop}
              onRecordStart={handleRecordStart}
              onRecordSave={handleRecordSave}
              onRecordDiscard={handleRecordDiscard}
            />
          </div>

          {/* Right columns: camera grid */}
          <div className="lg:col-span-2 flex flex-col gap-4">
            {(['camera_1', 'camera_2'] as const).map(id => (
              <div key={id} className="flex-1">
                <CameraStream
                  cameraId={id}
                  cameraName={id.replace(/_/g, ' ').toUpperCase()}
                  apiUrl={apiUrl}
                  active={cameraActive}
                />
              </div>
            ))}
          </div>

        </div>

        {/* ---- Debug panel (bottom) ---- */}
        {sidePanel === 'debug' && (
          <Card className="border-dashed bg-zinc-950 text-zinc-100">
            <CardHeader className="pb-2">
              <CardTitle className="text-zinc-100 font-mono text-sm">Debug — Process Logs</CardTitle>
            </CardHeader>
            <CardContent>
              <DebugPanel apiUrl={apiUrl} />
            </CardContent>
          </Card>
        )}

      </main>

      {/* ---- Diagnosis modal ---- */}
      {showDiagnosis && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 backdrop-blur-sm">
          <div className="bg-background border border-border rounded-xl shadow-2xl w-full max-w-lg mx-4 overflow-hidden">
            {/* Header */}
            <div className="flex items-center gap-3 px-5 py-4 border-b border-border bg-amber-50 dark:bg-amber-950/40">
              <svg className="h-5 w-5 text-amber-600 shrink-0" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5">
                <path d="M12 9v4m0 4h.01M10.29 3.86L1.82 18a2 2 0 001.71 3h16.94a2 2 0 001.71-3L13.71 3.86a2 2 0 00-3.42 0z"/>
              </svg>
              <span className="font-bold text-amber-800 dark:text-amber-200 text-sm tracking-wide">DIAGNOSIS — Teleop cannot start</span>
              <button
                onClick={() => setShowDiagnosis(false)}
                className="ml-auto text-amber-500 hover:text-amber-700 dark:hover:text-amber-300 text-lg leading-none"
              >✕</button>
            </div>
            {/* Body */}
            <div className="px-5 py-4 space-y-4 text-sm max-h-[60vh] overflow-y-auto">
              {errorState?.message && (
                <p className="text-foreground font-medium">{errorState.message}</p>
              )}

              {/* Diagnostic checks: pass/fail table */}
              {errorState?.diagnostic_checks && Object.keys(errorState.diagnostic_checks).length > 0 && (
                <div>
                  <p className="text-xs font-semibold text-muted-foreground uppercase tracking-wide mb-1.5">Diagnostic checks</p>
                  <div className="rounded-md border border-border overflow-hidden">
                    {Object.entries(errorState.diagnostic_checks).map(([check, passed]) => (
                      <div key={check} className="flex items-center justify-between px-3 py-1.5 odd:bg-muted/40 text-xs">
                        <span className="font-mono text-foreground">{check.replace(/_/g, ' ')}</span>
                        {passed ? (
                          <span className="text-emerald-600 dark:text-emerald-400 font-semibold">✓ pass</span>
                        ) : (
                          <span className="text-red-600 dark:text-red-400 font-semibold">✗ fail</span>
                        )}
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {/* Component errors */}
              {errorState?.component_errors && Object.keys(errorState.component_errors).length > 0 && (
                <div>
                  <p className="text-xs font-semibold text-muted-foreground uppercase tracking-wide mb-1.5">Component errors</p>
                  <div className="rounded-md border border-border overflow-hidden">
                    {Object.entries(errorState.component_errors).map(([comp, detail]) => (
                      <div key={comp} className="px-3 py-1.5 odd:bg-muted/40 text-xs">
                        <span className="font-mono font-semibold text-red-600 dark:text-red-400">{comp}</span>
                        <span className="ml-2 text-muted-foreground">{JSON.stringify(detail)}</span>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {/* Missing topics (BOOT state) */}
              {errorState?.missing_topics && errorState.missing_topics.length > 0 && (
                <div>
                  <p className="text-xs font-semibold text-muted-foreground uppercase tracking-wide mb-1.5">Missing topics</p>
                  <ul className="space-y-0.5">
                    {errorState.missing_topics.map(t => (
                      <li key={t} className="font-mono text-xs text-red-600 dark:text-red-400 bg-red-50 dark:bg-red-950/40 px-2 py-0.5 rounded">{t}</li>
                    ))}
                  </ul>
                </div>
              )}

              {/* Out-of-limit joints (ALIGN state) */}
              {errorState?.out_of_limit && (() => {
                const { message: _m, ...arms } = errorState.out_of_limit as Record<string, unknown>
                return Object.keys(arms).length > 0 ? (
                  <div>
                    <p className="text-xs font-semibold text-muted-foreground uppercase tracking-wide mb-1.5">Out of limit joints</p>
                    {Object.entries(arms).map(([arm, joints]) => (
                      <div key={arm} className="mb-1">
                        <p className="text-xs font-semibold capitalize mb-0.5">{arm.replace(/_/g, ' ')}</p>
                        <ul className="space-y-0.5">
                          {(joints as string[]).map((j, i) => (
                            <li key={i} className="font-mono text-xs text-amber-700 dark:text-amber-300 bg-amber-50 dark:bg-amber-950/40 px-2 py-0.5 rounded">{j}</li>
                          ))}
                        </ul>
                      </div>
                    ))}
                  </div>
                ) : null
              })()}

              {!errorState && (
                <p className="text-muted-foreground italic">No diagnostic details available.</p>
              )}
            </div>
            {/* Footer */}
            <div className="px-5 py-3 border-t border-border flex justify-end">
              <button
                onClick={() => setShowDiagnosis(false)}
                className="px-4 py-1.5 rounded-md bg-muted hover:bg-muted/80 text-sm font-medium transition-colors"
              >Dismiss</button>
            </div>
          </div>
        </div>
      )}

    </div>
  )
}
