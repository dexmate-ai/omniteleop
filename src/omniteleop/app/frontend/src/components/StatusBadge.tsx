import { cn } from '../lib/utils'
import type { ControlState } from '../types'

const STATE_CONFIG: Record<ControlState, { label: string; dot: string; bg: string; text: string }> = {
  BOOT:      { label: 'Booting',    dot: 'bg-amber-400',  bg: 'bg-amber-50',   text: 'text-amber-700' },
  DIAGNOSIS: { label: 'Diagnosis',  dot: 'bg-orange-400', bg: 'bg-orange-50',  text: 'text-orange-700' },
  ALIGN:     { label: 'Aligning',   dot: 'bg-sky-400',    bg: 'bg-sky-50',     text: 'text-sky-700' },
  ACTIVE:    { label: 'Active',     dot: 'bg-emerald-400',bg: 'bg-emerald-50', text: 'text-emerald-700' },
  RECORD:    { label: 'Recording',  dot: 'bg-red-500 animate-pulse', bg: 'bg-red-50', text: 'text-red-700' },
  PAUSE:     { label: 'Paused',     dot: 'bg-yellow-400', bg: 'bg-yellow-50',  text: 'text-yellow-700' },
}

interface Props {
  state: ControlState
  className?: string
}

export function StatusBadge({ state, className }: Props) {
  const cfg = STATE_CONFIG[state] ?? STATE_CONFIG.BOOT
  return (
    <span className={cn('inline-flex items-center gap-1.5 rounded-full px-2.5 py-0.5 text-xs font-medium', cfg.bg, cfg.text, className)}>
      <span className={cn('h-1.5 w-1.5 rounded-full', cfg.dot)} />
      {cfg.label}
    </span>
  )
}
