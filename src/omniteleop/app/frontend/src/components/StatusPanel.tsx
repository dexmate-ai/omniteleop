import { Cpu } from 'lucide-react'
import { Card, CardContent, CardHeader, CardTitle } from './ui/card'
import { StatusBadge } from './StatusBadge'
import type { ControlState } from '../types'
import { cn } from '../lib/utils'

const COMPONENT_LABELS: Record<string, string> = {
  left_arm: 'Left Arm',
  right_arm: 'Right Arm',
  torso: 'Torso',
  head: 'Head',
  left_hand: 'Left Hand',
  right_hand: 'Right Hand',
  chassis: 'Chassis',
}

interface Props {
  controlState: ControlState
  teleopRunning: boolean
  recordMode: boolean
  softwareEstop: boolean
  activeComponents: string[]
}

export function StatusPanel({ controlState, teleopRunning, recordMode, softwareEstop, activeComponents }: Props) {
  const compLabels = activeComponents.map(c => COMPONENT_LABELS[c] ?? c)

  return (
    <Card className={cn('transition-colors', softwareEstop ? 'border-red-300 bg-red-50/40' : '')}>
      <CardHeader className="pb-2">
        <div className="flex items-center justify-between">
          <CardTitle>
            <span className="flex items-center gap-1.5"><Cpu className="h-3.5 w-3.5" />Status</span>
          </CardTitle>
          <StatusBadge state={controlState} />
        </div>
      </CardHeader>
      <CardContent>
        <div className="grid grid-cols-[auto_1fr] gap-x-4 gap-y-2 text-xs">
          <span className="text-muted-foreground self-center">Teleop</span>
          <span className={cn('flex items-center gap-1.5 font-medium justify-end', teleopRunning ? 'text-emerald-600' : 'text-zinc-500')}>
            <span className={cn('h-1.5 w-1.5 rounded-full shrink-0', teleopRunning ? 'bg-emerald-500' : 'bg-zinc-300')} />
            {teleopRunning ? 'Running' : 'Stopped'}
          </span>

          <span className="text-muted-foreground self-center">Record</span>
          <span className={cn('flex items-center gap-1.5 font-medium justify-end', recordMode ? 'text-sky-600' : 'text-zinc-500')}>
            <span className={cn('h-1.5 w-1.5 rounded-full shrink-0', recordMode ? 'bg-sky-500' : 'bg-zinc-300')} />
            {recordMode ? 'On' : 'Off'}
          </span>

          <span className="text-muted-foreground self-center">E-Stop</span>
          <span className={cn('flex items-center gap-1.5 font-medium justify-end', softwareEstop ? 'text-red-600' : 'text-zinc-500')}>
            <span className={cn('h-1.5 w-1.5 rounded-full shrink-0', softwareEstop ? 'bg-red-500' : 'bg-zinc-300')} />
            {softwareEstop ? 'Active' : 'Clear'}
          </span>

          <span className="text-muted-foreground self-center">Active Components</span>
          <span className="font-medium text-zinc-600 dark:text-zinc-300 text-right leading-snug">
            {compLabels.length > 0 ? compLabels.join(', ') : '—'}
          </span>
        </div>
      </CardContent>
    </Card>
  )
}
