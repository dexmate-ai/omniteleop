// Control states reported by the backend
export type ControlState =
  | 'BOOT'
  | 'DIAGNOSIS'
  | 'ALIGN'
  | 'ACTIVE'
  | 'RECORD'
  | 'PAUSE'

// One item in the robots array streamed via /ws/state
export interface StateTopic {
  topic: string
  data_type: string
  data: unknown
}

// The top-level WebSocket message from /ws/state
export interface WsStateMessage {
  timestamp_posix: number
  robots: Record<string, StateTopic[]>
}

// Parsed and normalised state used in the UI
export interface RecorderState {
  controlState: ControlState
  isRecording: boolean
  episodeId: string | null
  softwareEstop: boolean
  recordMode: boolean       // backend was started with --record-mode
  teleopRunning: boolean    // teleop subprocess stack is active
  errorState: ErrorStateData | null
  robotObservations: Record<string, number[]>
  exoObservations: Record<string, number[]>
  activeComponents: string[]
}

export interface ErrorStateData {
  state: string
  message?: string
missing_topics?: string[]
  found_topics?: string[]
  diagnostic_checks?: Record<string, unknown>
  component_errors?: Record<string, unknown>
  out_of_limit?: Record<string, unknown>
}

export interface Sensor {
  id: string
  data_type: string
}
