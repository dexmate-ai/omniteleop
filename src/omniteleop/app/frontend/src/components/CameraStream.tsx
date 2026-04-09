import { useEffect, useRef, useState, useCallback } from 'react'
import { Card, CardContent, CardHeader, CardTitle } from './ui/card'
import { httpToWs } from '../lib/utils'

interface Props {
  cameraId: string
  cameraName: string
  apiUrl: string
  active: boolean  // only stream when robot is active/recording
}

export function CameraStream({ cameraId, cameraName, apiUrl, active }: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const wsRef = useRef<WebSocket | null>(null)
  const [streaming, setStreaming] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const disconnect = useCallback(() => {
    wsRef.current?.close()
    wsRef.current = null
    setStreaming(false)
  }, [])

  const connect = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) return
    wsRef.current?.close()

    const ws = new WebSocket(`${httpToWs(apiUrl)}/ws/sensors/stream/${cameraId}`)
    wsRef.current = ws
    ws.binaryType = 'arraybuffer'

    ws.onopen = () => { setStreaming(true); setError(null) }
    ws.onerror = () => { setError('Connection error'); setStreaming(false) }
    ws.onclose = () => { setStreaming(false) }

    ws.onmessage = (ev) => {
      const data: ArrayBuffer =
        ev.data instanceof ArrayBuffer
          ? ev.data
          : ev.data instanceof Blob
          ? (ev.data as Blob) as unknown as ArrayBuffer   // handled below
          : new Uint8Array(ev.data as ArrayLike<number>).buffer

      const draw = (buffer: ArrayBuffer) => {
        const blob = new Blob([buffer], { type: 'image/jpeg' })
        const url = URL.createObjectURL(blob)
        const img = new Image()
        img.onload = () => {
          const canvas = canvasRef.current
          if (!canvas) { URL.revokeObjectURL(url); return }
          canvas.width = img.width
          canvas.height = img.height
          canvas.getContext('2d')?.drawImage(img, 0, 0)
          URL.revokeObjectURL(url)
        }
        img.onerror = () => URL.revokeObjectURL(url)
        img.src = url
      }

      if (ev.data instanceof Blob) {
        ev.data.arrayBuffer().then(draw)
      } else {
        draw(data)
      }
    }
  }, [apiUrl, cameraId])

  useEffect(() => {
    if (active) connect()
    else disconnect()
  }, [active, connect, disconnect])

  useEffect(() => () => disconnect(), [cameraId, disconnect])

  return (
    <Card className="overflow-hidden flex flex-col h-full">
      <CardHeader className="py-3 px-4 shrink-0">
        <div className="flex items-center justify-between">
          <CardTitle>{cameraName}</CardTitle>
          {streaming
            ? <span className="text-xs font-medium text-emerald-600">● Live</span>
            : <span className="text-xs text-muted-foreground">{active ? 'Connecting…' : 'Inactive'}</span>}
        </div>
      </CardHeader>
      <CardContent className="p-0 flex-1 flex flex-col">
        <div className="relative bg-zinc-950 flex-1">
          <canvas ref={canvasRef} className="w-full h-full object-contain" />
          {!streaming && (
            <div className="absolute inset-0 flex items-center justify-center text-zinc-500 text-sm">
              {error ?? (active ? 'Connecting…' : 'Waiting for robot')}
            </div>
          )}
        </div>
        <p className="px-4 py-1.5 text-xs text-muted-foreground font-mono">{cameraId}</p>
      </CardContent>
    </Card>
  )
}
