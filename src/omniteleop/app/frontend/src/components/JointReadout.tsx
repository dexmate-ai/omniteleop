import { Card, CardContent, CardHeader, CardTitle } from './ui/card'

interface Props {
  robotObs: Record<string, number[]>
  exoObs: Record<string, number[]>
}

function JointBar({ value, label }: { value: number; label: string }) {
  // Clamp to [-π, π] range for display
  const pct = Math.round(((value + Math.PI) / (2 * Math.PI)) * 100)
  return (
    <div className="flex items-center gap-2 text-xs">
      <span className="w-4 text-muted-foreground font-mono shrink-0">{label}</span>
      <div className="flex-1 bg-muted rounded-full h-1.5 overflow-hidden">
        <div
          className="h-full bg-primary/60 rounded-full transition-all duration-75"
          style={{ width: `${Math.max(2, Math.min(100, pct))}%` }}
        />
      </div>
      <span className="w-10 text-right font-mono text-muted-foreground">{value.toFixed(2)}</span>
    </div>
  )
}

function ComponentBlock({ name, values }: { name: string; values: number[] }) {
  return (
    <div className="space-y-1">
      <p className="text-xs font-medium capitalize">{name.replace(/_/g, ' ')}</p>
      {values.map((v, i) => (
        <JointBar key={i} value={v} label={`j${i + 1}`} />
      ))}
    </div>
  )
}

export function JointReadout({ robotObs, exoObs }: Props) {
  const hasRobot = Object.keys(robotObs).length > 0
  const hasExo = Object.keys(exoObs).length > 0
  if (!hasRobot && !hasExo) return null

  return (
    <Card>
      <CardHeader>
        <CardTitle>Joint Readout</CardTitle>
      </CardHeader>
      <CardContent className="grid grid-cols-2 gap-x-6 gap-y-4">
        {hasRobot && Object.entries(robotObs).map(([name, values]) => (
          <ComponentBlock key={`robot-${name}`} name={name} values={values} />
        ))}
        {hasExo && Object.entries(exoObs).map(([name, values]) => (
          <ComponentBlock key={`exo-${name}`} name={`exo ${name}`} values={values} />
        ))}
      </CardContent>
    </Card>
  )
}
