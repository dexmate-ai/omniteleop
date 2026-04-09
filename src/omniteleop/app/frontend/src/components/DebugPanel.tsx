import { useEffect, useRef, useState } from 'react'
import { httpToWs } from '../lib/utils'

const PROC_NAMES = ['arm_reader', 'joycon_reader', 'command_processor', 'robot_controller'] as const
type ProcName = typeof PROC_NAMES[number] | 'app_backend'

interface LogEntry {
  source: string
  line: string
}

export function DebugPanel({ apiUrl }: { apiUrl: string }) {
  const [logs, setLogs] = useState<LogEntry[]>([])
  const [tab, setTab] = useState<ProcName>('app_backend')
  const [connected, setConnected] = useState(false)
  const bottomRef = useRef<HTMLDivElement>(null)
  const autoScrollRef = useRef(true)

  useEffect(() => {
    let alive = true
    let ws: WebSocket | null = null
    let retryTimer: ReturnType<typeof setTimeout> | null = null

    function connect() {
      if (!alive) return
      ws = new WebSocket(`${httpToWs(apiUrl)}/ws/logs`)
      ws.onopen = () => { if (alive) setConnected(true) }
      ws.onmessage = (ev) => {
        if (!alive) return
        try {
          const entry: LogEntry = JSON.parse(ev.data as string)
          setLogs(prev => {
            const next = [...prev, entry]
            return next.length > 2000 ? next.slice(-2000) : next
          })
        } catch { /* ignore */ }
      }
      ws.onerror = () => setConnected(false)
      ws.onclose = () => {
        setConnected(false)
        if (alive) retryTimer = setTimeout(connect, 3000)
      }
    }

    connect()
    return () => {
      alive = false
      if (retryTimer) clearTimeout(retryTimer)
      ws?.close()
    }
  }, [apiUrl])

  // Auto-scroll to bottom when new logs arrive
  useEffect(() => {
    if (autoScrollRef.current) {
      bottomRef.current?.scrollIntoView({ behavior: 'instant' })
    }
  }, [logs])

  const filtered = logs.filter(e => e.source === tab)

  const counts: Record<string, number> = {}
  for (const e of logs) counts[e.source] = (counts[e.source] ?? 0) + 1

  function lineColor(line: string) {
    const l = line.toLowerCase()
    if (l.includes('error') || l.includes('exception') || l.includes('traceback')) return 'text-red-400'
    if (l.includes('warn')) return 'text-yellow-400'
    if (l.includes('success') || l.includes(' ok ') || l.includes('started')) return 'text-emerald-400'
    return 'text-gray-300'
  }

  return (
    <div className="flex flex-col h-full min-h-0">
      {/* Tab bar */}
      <div className="flex items-center gap-1 pb-2 border-b border-zinc-700 flex-wrap">
        {(['app_backend', ...PROC_NAMES] as ProcName[]).map(name => (
          <button
            key={name}
            onClick={() => setTab(name)}
            className={`px-2 py-0.5 rounded text-xs font-mono transition-colors ${
              tab === name
                ? 'bg-zinc-600 text-white'
                : 'text-zinc-400 hover:text-zinc-200 hover:bg-zinc-700/60'
            }`}
          >
            {name}
            {counts[name] ? (
              <span className="ml-1 text-zinc-500">{counts[name]}</span>
            ) : null}
          </button>
        ))}
        <div className="ml-auto flex items-center gap-2">
          <span className={`h-1.5 w-1.5 rounded-full ${connected ? 'bg-emerald-500' : 'bg-zinc-600'}`} />
          <button
            onClick={() => setLogs([])}
            className="text-xs text-zinc-500 hover:text-zinc-300 transition-colors"
          >
            clear
          </button>
        </div>
      </div>

      {/* Log output */}
      <div
        className="flex-1 overflow-y-auto font-mono text-xs leading-5 pt-2 min-h-0"
        style={{ maxHeight: '420px' }}
        onScroll={e => {
          const el = e.currentTarget
          autoScrollRef.current = el.scrollHeight - el.scrollTop - el.clientHeight < 40
        }}
      >
        {filtered.length === 0 ? (
          <p className="text-zinc-600 italic px-1">No log output yet.</p>
        ) : (
          filtered.map((entry, i) => (
            <div key={i} className="flex gap-2 px-1 hover:bg-zinc-800/40">
              <span className={lineColor(entry.line)}>{entry.line}</span>
            </div>
          ))
        )}
        <div ref={bottomRef} />
      </div>
    </div>
  )
}
