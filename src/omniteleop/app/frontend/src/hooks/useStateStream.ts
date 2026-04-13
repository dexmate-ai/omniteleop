import { useEffect, useRef, useState } from 'react'
import type { RecorderState, WsStateMessage, ControlState, ErrorStateData } from '../types'
import { httpToWs } from '../lib/utils'

const DEFAULT_STATE: RecorderState = {
  controlState: 'BOOT',
  isRecording: false,
  episodeId: null,
  softwareEstop: false,
  recordMode: false,
  teleopRunning: false,
  errorState: null,
  robotObservations: {},
  exoObservations: {},
  activeComponents: [],
}

function parseMessage(msg: WsStateMessage): RecorderState {
  const topics = Object.values(msg.robots ?? {})[0] ?? []

  const get = (topic: string) => topics.find(t => t.topic === topic)?.data

  const controlState = (get('control_state') as ControlState) ?? 'DEAD'
  const episodeId = (get('episode_id') as string) || null
  const isRecording = controlState === 'RECORD'
  const recordMode = (get('record_mode') as boolean) ?? false
  const teleopRunning = (get('teleop_running') as boolean) ?? false

  const estopData = get('estop_state') as { software_estop?: boolean } | null
  const softwareEstop = estopData?.software_estop ?? controlState === 'PAUSE'

  const errorState = (get('error_state') as ErrorStateData) ?? null
  const activeComponents = (get('active_components') as string[]) ?? []

  const robotObservations: Record<string, number[]> = {}
  const exoObservations: Record<string, number[]> = {}

  for (const t of topics) {
    if (t.topic.startsWith('observation/state/')) {
      const key = t.topic.replace('observation/state/', '')
      robotObservations[key] = t.data as number[]
    } else if (t.topic.startsWith('observation/exo/')) {
      const key = t.topic.replace('observation/exo/', '')
      exoObservations[key] = t.data as number[]
    }
  }

  return { controlState, isRecording, episodeId, softwareEstop, recordMode, teleopRunning, errorState, robotObservations, exoObservations, activeComponents }
}

export function useStateStream(apiUrl: string) {
  const [state, setState] = useState<RecorderState>(DEFAULT_STATE)
  const [connected, setConnected] = useState(false)
  const wsRef = useRef<WebSocket | null>(null)
  const retryRef = useRef<ReturnType<typeof setTimeout> | null>(null)

  useEffect(() => {
    let alive = true

    function connect() {
      if (!alive) return
      let ws: WebSocket
      try {
        ws = new WebSocket(`${httpToWs(apiUrl)}/ws/state`)
      } catch {
        // Invalid URL — retry after delay rather than crashing.
        if (alive) retryRef.current = setTimeout(connect, 2000)
        return
      }
      wsRef.current = ws

      ws.onopen = () => {
        if (!alive) { ws.close(); return }
        setConnected(true)
      }

      ws.onmessage = (ev) => {
        if (!alive) return
        try {
          const msg: WsStateMessage = JSON.parse(ev.data as string)
          setState(parseMessage(msg))
        } catch {
          // malformed frame – ignore
        }
      }

      ws.onerror = () => {
        setConnected(false)
      }

      ws.onclose = () => {
        setConnected(false)
        if (alive) {
          retryRef.current = setTimeout(connect, 2000)
        }
      }
    }

    connect()

    return () => {
      alive = false
      if (retryRef.current) clearTimeout(retryRef.current)
      wsRef.current?.close()
    }
  }, [apiUrl])

  return { state, connected }
}
