import { AlertTriangle, CheckCircle, Wifi, WifiOff, Activity } from 'lucide-react'
import type { ErrorStateData, ControlState } from '../types'
import { Card, CardContent, CardHeader, CardTitle } from './ui/card'

interface Props {
  controlState: ControlState
  errorState: ErrorStateData | null
}

export function DiagnosticsPanel({ controlState, errorState }: Props) {
  if (!errorState || controlState === 'ACTIVE' || controlState === 'RECORD') return null

  return (
    <Card className="border-amber-200 bg-amber-50/50">
      <CardHeader>
        <CardTitle className="flex items-center gap-2 text-amber-700">
          <AlertTriangle className="h-4 w-4" />
          Diagnostics — {controlState}
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-3 text-sm">

        {controlState === 'DEAD' && (
          <div className="flex items-center gap-2 text-red-600">
            <WifiOff className="h-4 w-4 shrink-0" />
            <span>Cannot reach Jetson at <code className="font-mono text-xs bg-red-100 px-1 rounded">{errorState.jetson_ip}</code></span>
          </div>
        )}

        {controlState === 'BOOT' && (
          <div className="space-y-2">
            {(errorState.missing_topics ?? []).length > 0 && (
              <div>
                <p className="font-medium text-red-600 mb-1 flex items-center gap-1.5">
                  <WifiOff className="h-3.5 w-3.5" /> Missing topics
                </p>
                <ul className="ml-5 space-y-0.5">
                  {(errorState.missing_topics ?? []).map(t => (
                    <li key={t} className="font-mono text-xs text-red-700">{t}</li>
                  ))}
                </ul>
              </div>
            )}
            {(errorState.found_topics ?? []).length > 0 && (
              <div>
                <p className="font-medium text-emerald-600 mb-1 flex items-center gap-1.5">
                  <Wifi className="h-3.5 w-3.5" /> Found topics
                </p>
                <ul className="ml-5 space-y-0.5">
                  {(errorState.found_topics ?? []).map(t => (
                    <li key={t} className="font-mono text-xs text-emerald-700">{t}</li>
                  ))}
                </ul>
              </div>
            )}
          </div>
        )}

        {controlState === 'DIAGNOSIS' && errorState.diagnostic_checks && (
          <div className="space-y-1.5">
            {Object.entries(errorState.diagnostic_checks).map(([key, val]) => {
              const ok = val === true || val === 'ok'
              return (
                <div key={key} className="flex items-center gap-2">
                  {ok
                    ? <CheckCircle className="h-3.5 w-3.5 text-emerald-500 shrink-0" />
                    : <AlertTriangle className="h-3.5 w-3.5 text-red-500 shrink-0" />}
                  <span className={ok ? 'text-emerald-700' : 'text-red-700'}>
                    {key.replace(/_/g, ' ')}
                  </span>
                  {!ok && typeof val === 'string' && (
                    <span className="text-xs text-muted-foreground truncate">— {val}</span>
                  )}
                </div>
              )
            })}
          </div>
        )}

        {controlState === 'ALIGN' && (
          <div className="flex items-center gap-2 text-sky-700">
            <Activity className="h-4 w-4 shrink-0" />
            <span>{errorState.message ?? 'Move exo arms to match robot joint positions'}</span>
          </div>
        )}

      </CardContent>
    </Card>
  )
}
