import { useState } from 'react'
import { Play, Square, Circle, Trash2, ChevronDown } from 'lucide-react'
import { Button } from './ui/button'
import { Card, CardContent, CardHeader, CardTitle } from './ui/card'
import type { ControlState } from '../types'
import { cn } from '../lib/utils'

export interface TeleopLaunchConfig {
  record_mode: boolean
  env: {
    ROBOT_NAME: string
    ROBOT_CONFIG: string
    ZENOH_CONFIG: string
  }
}

interface Props {
  controlState: ControlState
  teleopRunning: boolean
  isRecording: boolean
  episodeId: string | null
  softwareEstop: boolean
  recordMode: boolean
  loading: boolean
  onTeleopStart: (cfg: TeleopLaunchConfig) => void
  onTeleopStop: () => void
  onRecordStart: () => void
  onRecordSave: () => void
  onRecordDiscard: () => void
}

function Field({ label, value, onChange, placeholder }: {
  label: string
  value: string
  onChange: (v: string) => void
  placeholder?: string
}) {
  return (
    <div className="space-y-0.5">
      <label className="text-xs text-muted-foreground">{label}</label>
      <input
        className="w-full h-7 px-2 rounded border border-input bg-background text-xs focus:outline-none focus:ring-1 focus:ring-ring font-mono"
        value={value}
        onChange={e => onChange(e.target.value)}
        placeholder={placeholder}
      />
    </div>
  )
}

export function TeleopControls({
  controlState,
  teleopRunning,
  isRecording,
  episodeId,
  softwareEstop,
  recordMode,
  loading,
  onTeleopStart,
  onTeleopStop,
  onRecordStart,
  onRecordSave,
  onRecordDiscard,
}: Props) {
  const [showLaunchForm, setShowLaunchForm] = useState(false)
  const [cfg, setCfg] = useState<TeleopLaunchConfig>({
    record_mode: false,
    env: { ROBOT_NAME: '', ROBOT_CONFIG: '', ZENOH_CONFIG: '' },
  })

  const setEnv = (key: keyof TeleopLaunchConfig['env'], val: string) =>
    setCfg(c => ({ ...c, env: { ...c.env, [key]: val } }))

  const canStartTeleop = !teleopRunning && !softwareEstop
  const robotReady = controlState === 'ACTIVE' || controlState === 'RECORD'
  const canRecord = recordMode && robotReady && teleopRunning

  function handleStart() {
    onTeleopStart(cfg)
    setShowLaunchForm(false)
  }

  return (
    <Card>
      <CardHeader>
        <CardTitle>Controls</CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">

        {/* ---- Teleop start/stop ---- */}
        <div className="space-y-2">
          <p className="text-xs text-muted-foreground font-medium">Teleoperation</p>

          {/* Launch form — shown when about to start */}
          {showLaunchForm && (
            <div className="rounded-lg border border-border bg-muted/30 p-3 space-y-2">
              <Field
                label="ROBOT_NAME"
                value={cfg.env.ROBOT_NAME}
                onChange={v => setEnv('ROBOT_NAME', v)}
                placeholder="e.g. dm/robot1  (auto-detected if empty)"
              />
              <Field
                label="ROBOT_CONFIG"
                value={cfg.env.ROBOT_CONFIG}
                onChange={v => setEnv('ROBOT_CONFIG', v)}
                placeholder="e.g. vega_1_f5d6  (default: vega_1p_gripper)"
              />
              <Field
                label="ZENOH_CONFIG (certificate name)"
                value={cfg.env.ZENOH_CONFIG}
                onChange={v => setEnv('ZENOH_CONFIG', v)}
                placeholder="e.g. my_cert  (auto-detected if empty)"
              />
              <label className="flex items-center gap-2 cursor-pointer pt-1">
                <input
                  type="checkbox"
                  checked={cfg.record_mode}
                  onChange={e => setCfg(c => ({ ...c, record_mode: e.target.checked }))}
                  className="rounded"
                />
                <span className="text-xs font-medium">Enable recording</span>
              </label>
              <div className="flex gap-2 pt-1">
                <Button size="sm" onClick={handleStart} disabled={loading}
                  className="flex-1 bg-emerald-600 hover:bg-emerald-700 text-white gap-1.5">
                  <Play className="h-3 w-3 fill-current" /> Launch
                </Button>
                <Button size="sm" variant="ghost" onClick={() => setShowLaunchForm(false)}
                  className="text-muted-foreground">
                  Cancel
                </Button>
              </div>
            </div>
          )}

          {/* Start / Stop row */}
          {!showLaunchForm && (
            <div className="flex gap-2">
              <Button
                onClick={() => setShowLaunchForm(true)}
                disabled={loading || !canStartTeleop}
                className={cn('gap-2 flex-1', canStartTeleop ? 'bg-emerald-600 hover:bg-emerald-700 text-white' : '')}
                variant={canStartTeleop ? 'default' : 'outline'}
              >
                <Play className="h-3.5 w-3.5 fill-current" />
                Start
                <ChevronDown className="h-3 w-3 ml-auto opacity-60" />
              </Button>
              <Button
                onClick={onTeleopStop}
                disabled={loading || !teleopRunning}
                variant="outline"
                className="gap-2 flex-1"
              >
                <Square className="h-3.5 w-3.5 fill-current" />
                Stop
              </Button>
            </div>
          )}
        </div>

        {/* ---- Recording ---- */}
        <div className="space-y-1.5 pt-1 border-t">
          <div className="flex items-center justify-between">
            <p className="text-xs text-muted-foreground font-medium">Recording</p>
            {!recordMode && teleopRunning && (
              <span className="text-xs text-muted-foreground/60 italic">not enabled</span>
            )}
          </div>

          {isRecording && episodeId && (
            <div className="flex items-center gap-2 px-2.5 py-1.5 bg-red-50 border border-red-200 rounded-lg">
              <span className="relative flex h-2 w-2 shrink-0">
                <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-red-400 opacity-75" />
                <span className="relative inline-flex rounded-full h-2 w-2 bg-red-500" />
              </span>
              <span className="text-xs text-red-700 font-medium">Recording</span>
              <span className="ml-auto font-mono text-xs text-red-600 truncate max-w-[180px]">{episodeId}</span>
            </div>
          )}

          <div className={cn('flex flex-wrap gap-2', !recordMode && 'opacity-50 pointer-events-none select-none')}>
            <Button
              onClick={onRecordStart}
              disabled={loading || isRecording || !canRecord}
              variant={canRecord && !isRecording ? 'default' : 'outline'}
              className={cn('gap-1.5', canRecord && !isRecording ? 'bg-emerald-600 hover:bg-emerald-700 text-white' : '')}
              size="sm"
            >
              <Circle className="h-3 w-3 fill-current" /> Start
            </Button>
            <Button
              onClick={onRecordSave}
              disabled={loading || !isRecording}
              variant="secondary"
              size="sm"
              className="gap-1.5"
            >
              <Square className="h-3 w-3" /> Save
            </Button>
            <Button
              onClick={onRecordDiscard}
              disabled={loading || !isRecording}
              variant="outline"
              size="sm"
              className="gap-1.5 text-orange-600 border-orange-200 hover:bg-orange-50"
            >
              <Trash2 className="h-3 w-3" /> Discard
            </Button>
          </div>
        </div>

      </CardContent>
    </Card>
  )
}
