"use client"

import { useEffect, useMemo, useRef, useState, useCallback } from "react"

type NodeData = {
  id: string
  x: number
  y: number
  floor: string
  label: string
}

type EdgeData = {
  source: string
  target: string
  weight: number
  path: Array<[number, number]>
  isStair: boolean
}

type PathfindingMapProps = {
  onPathComputed?: (instructions: string[], distance: number) => void
  autoThreatNode?: { id: string; floor?: string; label?: string }
  resetSignal?: number
}

const GEXF_PATH = "/BMHS_FloorPlan_combined.gexf"
const IMAGE_PATH = "/BMHS_FloorPlan.JPG"

const LOCATION_CONTEXT: Record<string, string> = {
  ENTRY: "Entry",
  CLASS: "Classroom",
  TP: "Teacher Planning",
  SPED: "Special Education",
  HUB: "Interdisciplinary Hub",
  LAB: "Teaching Lab",
  FIT: "Fitness",
  TH: "Theater",
  TRK: "Track",
  MC: "Media Center",
  ART: "Art",
  MUSIC: "Music",
  BBT: "Black-Box Theater",
  MK: "Makerspace",
  TSHOP: "Theater Shop",
  GYM: "Gymnasium",
  LOCK: "Locker Rooms",
  KIT: "Kitchen/Servery",
  DINING: "Dining Commons",
  PREK: "Pre-K",
}

const FRIENDLY_BASE_NAMES = [
  "Atrium Corner",
  "Main Hall",
  "Commons Bend",
  "Library Nook",
  "Science Wing",
  "Gym Lobby",
  "Art Walk",
  "Music Hall",
  "East Stair",
  "West Stair",
  "North Stair",
  "South Stair",
  "Media Center",
  "Lab Crossway",
  "Hub Junction",
  "Dining Entry",
  "Courtyard Edge",
  "Bridge Hall",
  "Theater Walk",
  "Locker Row",
  "Fitness Hall",
  "Makerspace Turn",
  "Admin Lobby",
]

function friendlyLabel(rawLabel: string, floor: string, idx: number): string {
  const areaMatch = rawLabel.match(/^([A-Za-z]+)_F\d+/)
  if (areaMatch) {
    const code = areaMatch[1].toUpperCase()
    if (LOCATION_CONTEXT[code]) {
      return `${LOCATION_CONTEXT[code]} (${rawLabel})`
    }
  }
  if (rawLabel.includes("Node") || rawLabel.startsWith("Floor")) {
    const base = FRIENDLY_BASE_NAMES[idx % FRIENDLY_BASE_NAMES.length]
    return `${base} (${floor})`
  }
  return rawLabel
}

function prettyLabel(nodeId: string, nodesById: Map<string, NodeData>): string {
  return nodesById.get(nodeId)?.label || nodeId
}

function parseGexf(text: string): { nodes: NodeData[]; edges: EdgeData[] } {
  const parser = new DOMParser()
  const xml = parser.parseFromString(text, "application/xml")

  const nodeAttrMap = new Map<string, string>()
  xml.querySelectorAll('attributes[class="node"] attribute').forEach((attr) => {
    const id = attr.getAttribute("id")
    const title = attr.getAttribute("title")
    if (id && title) nodeAttrMap.set(id, title)
  })

  const edgeAttrMap = new Map<string, string>()
  xml.querySelectorAll('attributes[class="edge"] attribute').forEach((attr) => {
    const id = attr.getAttribute("id")
    const title = attr.getAttribute("title")
    if (id && title) edgeAttrMap.set(id, title)
  })

  const nodes: NodeData[] = []
  const nodeElements = Array.from(xml.querySelectorAll("node"))
  nodeElements.forEach((node, idx) => {
    const id = node.getAttribute("id") || ""
    const rawLabel = node.getAttribute("label") || id
    const attrValues = new Map<string, string>()
    node.querySelectorAll("attvalue").forEach((av) => {
      const key = av.getAttribute("for") || ""
      const title = nodeAttrMap.get(key) || key
      const val = av.getAttribute("value") || ""
      attrValues.set(title, val)
    })
    const floorName = attrValues.get("floor") || "Unknown"
    const label = friendlyLabel(rawLabel, floorName, idx)
    nodes.push({
      id,
      label,
      x: Number.parseFloat(attrValues.get("pos_x") || "0"),
      y: Number.parseFloat(attrValues.get("pos_y") || "0"),
      floor: floorName,
    })
  })

  const edges: EdgeData[] = []
  xml.querySelectorAll("edge").forEach((edge) => {
    const source = edge.getAttribute("source") || ""
    const target = edge.getAttribute("target") || ""
    const weight = Number.parseFloat(edge.getAttribute("weight") || "1")
    const attrValues = new Map<string, string>()
    edge.querySelectorAll("attvalue").forEach((av) => {
      const key = av.getAttribute("for") || ""
      const title = edgeAttrMap.get(key) || key
      const val = av.getAttribute("value") || ""
      attrValues.set(title, val)
    })
    let path: Array<[number, number]> = []
    try {
      const raw = attrValues.get("path") || "[]"
      const parsed = JSON.parse(raw)
      if (Array.isArray(parsed)) {
        path = parsed.map((p: [number, number]) => [Number(p[0]), Number(p[1])])
      }
    } catch {
      path = []
    }
    edges.push({
      source,
      target,
      weight,
      path,
      isStair: (attrValues.get("is_stair") || "false") === "true",
    })
  })

  return { nodes, edges }
}

function dijkstra(
  nodes: NodeData[],
  edges: EdgeData[],
  start: string,
  goal: string,
): { path: string[]; distance: number } | null {
  const neighbors = new Map<string, Array<{ to: string; weight: number }>>()
  edges.forEach((e) => {
    neighbors.set(e.source, [...(neighbors.get(e.source) || []), { to: e.target, weight: e.weight }])
    neighbors.set(e.target, [...(neighbors.get(e.target) || []), { to: e.source, weight: e.weight }])
  })

  const dist = new Map<string, number>()
  const prev = new Map<string, string | null>()
  nodes.forEach((n) => {
    dist.set(n.id, Infinity)
    prev.set(n.id, null)
  })
  dist.set(start, 0)

  const unvisited = new Set(nodes.map((n) => n.id))

  while (unvisited.size > 0) {
    let u: string | null = null
    let minDist = Infinity
    unvisited.forEach((id) => {
      const d = dist.get(id) ?? Infinity
      if (d < minDist) {
        minDist = d
        u = id
      }
    })

    if (u === null) break
    unvisited.delete(u)
    if (u === goal) break

    const neigh = neighbors.get(u) || []
    for (const { to, weight } of neigh) {
      if (!unvisited.has(to)) continue
      const alt = (dist.get(u) ?? Infinity) + weight
      if (alt < (dist.get(to) ?? Infinity)) {
        dist.set(to, alt)
        prev.set(to, u)
      }
    }
  }

  if (!prev.has(goal) || (dist.get(goal) ?? Infinity) === Infinity) return null

  const path: string[] = []
  let cur: string | null = goal
  while (cur) {
    path.unshift(cur)
    cur = prev.get(cur) || null
  }
  return { path, distance: dist.get(goal) ?? 0 }
}

function buildInstructions(
  path: string[],
  nodesById: Map<string, NodeData>,
  edgesByKey: Map<string, EdgeData>,
): string[] {
  if (path.length === 0) return []
  const instructions: string[] = []
  instructions.push(`Start at ${prettyLabel(path[0], nodesById)}`)
  for (let i = 1; i < path.length; i++) {
    const from = path[i - 1]
    const to = path[i]
    const key = `${from}|${to}`
    const edge = edgesByKey.get(key) || edgesByKey.get(`${to}|${from}`)
    const toLabel = prettyLabel(to, nodesById)
    const fromNode = nodesById.get(from)
    const toNode = nodesById.get(to)
    if (edge?.isStair && fromNode && toNode && fromNode.floor !== toNode.floor) {
      const goingUp = Number.parseInt(toNode.floor.replace(/\D+/g, "")) > Number.parseInt(fromNode.floor.replace(/\D+/g, ""))
      instructions.push(goingUp ? `Go up the staircase to ${toLabel}` : `Go down the staircase to ${toLabel}`)
    } else {
      instructions.push(`Move to ${toLabel}`)
    }
  }
  instructions.push(`Arrive at ${prettyLabel(path[path.length - 1], nodesById)}`)
  return instructions
}

export default function PathfindingMap({ onPathComputed, autoThreatNode, resetSignal }: PathfindingMapProps) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null)
  const containerRef = useRef<HTMLDivElement | null>(null)
  const [image, setImage] = useState<HTMLImageElement | null>(null)
  const [nodes, setNodes] = useState<NodeData[]>([])
  const [edges, setEdges] = useState<EdgeData[]>([])
  const [path, setPath] = useState<string[]>([])
  const [distance, setDistance] = useState(0)
  const [instructions, setInstructions] = useState<string[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [currentFloor, setCurrentFloor] = useState<string | null>(null)
  const [assignMode, setAssignMode] = useState<"officer" | "threat" | null>(null)
  const [officerNode, setOfficerNode] = useState<string | null>(null)
  const [threatNode, setThreatNode] = useState<string | null>(null)

  const nodesById = useMemo(() => new Map(nodes.map((n) => [n.id, n])), [nodes])
  const edgesByKey = useMemo(() => {
    const m = new Map<string, EdgeData>()
    edges.forEach((e) => {
      m.set(`${e.source}|${e.target}`, e)
      m.set(`${e.target}|${e.source}`, e)
    })
    return m
  }, [edges])

  const floors = useMemo(() => Array.from(new Set(nodes.map((n) => n.floor))).sort(), [nodes])
  const activeFloor = currentFloor || floors[0] || null

  useEffect(() => {
    const img = new Image()
    img.src = IMAGE_PATH
    img.onload = () => setImage(img)
    img.onerror = () => setError("Failed to load floor plan image")

    fetch(GEXF_PATH)
      .then((res) => res.text())
      .then((txt) => {
        const parsed = parseGexf(txt)
        setNodes(parsed.nodes)
        setEdges(parsed.edges)
        if (parsed.nodes.length > 0) {
          const floorList = Array.from(new Set(parsed.nodes.map((n) => n.floor))).sort()
          setCurrentFloor(floorList[0] || null)
        }
        setLoading(false)
      })
      .catch(() => setError("Failed to load map data"))
  }, [])

  useEffect(() => {
    const handleResize = () => draw()
    window.addEventListener("resize", handleResize)
    return () => window.removeEventListener("resize", handleResize)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [image, nodes, edges, path, activeFloor])

  const computePath = useCallback((startId: string, endId: string) => {
    const result = dijkstra(nodes, edges, startId, endId)
    if (!result) {
      setPath([])
      setDistance(0)
      setInstructions(["No path found"])
      if (onPathComputed) onPathComputed(["No path found"], 0)
      return
    }
    setPath(result.path)
    setDistance(result.distance)
    const instr = buildInstructions(result.path, nodesById, edgesByKey)
    setInstructions(instr)
    if (onPathComputed) onPathComputed(instr, result.distance)
  }, [nodes, edges, nodesById, edgesByKey, onPathComputed])

  const draw = () => {
    const canvas = canvasRef.current
    const container = containerRef.current
    if (!canvas || !container || !image) return

    const containerWidth = container.clientWidth
    const aspect = image.height ? image.height / image.width : 1
    const cssWidth = containerWidth
    const cssHeight = Math.max(containerWidth * aspect, 420)
    const dpr = window.devicePixelRatio || 1
    canvas.width = Math.floor(cssWidth * dpr)
    canvas.height = Math.floor(cssHeight * dpr)
    canvas.style.width = `${cssWidth}px`
    canvas.style.height = `${cssHeight}px`

    const ctx = canvas.getContext("2d")
    if (!ctx) return

    ctx.setTransform(dpr, 0, 0, dpr, 0, 0)
    ctx.imageSmoothingEnabled = true
    // @ts-ignore imageSmoothingQuality may not exist in all contexts
    ctx.imageSmoothingQuality = "high"

    ctx.clearRect(0, 0, cssWidth, cssHeight)

    const scale = cssWidth / image.width
    const offsetY = (cssHeight - image.height * scale) / 2

    ctx.drawImage(image, 0, offsetY, image.width * scale, image.height * scale)

    ctx.lineJoin = "round"
    ctx.lineCap = "round"

    // draw all floors; fade non-active
    edges.forEach((edge) => {
      if (edge.isStair) return
      const src = nodesById.get(edge.source)
      const tgt = nodesById.get(edge.target)
      if (!src || !tgt) return
      if (src.floor != tgt.floor) return
      const isActive = activeFloor ? src.floor === activeFloor : true
      const pts = edge.path.length
        ? edge.path
        : [src, tgt].map((n) => [n.x, n.y] as [number, number])
      if (!pts.length) return
      ctx.beginPath()
      pts.forEach(([x, y], idx) => {
        const px = x * scale
        const py = y * scale + offsetY
        if (idx === 0) ctx.moveTo(px, py)
        else ctx.lineTo(px, py)
      })
      ctx.strokeStyle = isActive ? "rgba(160,160,160,0.9)" : "rgba(160,160,160,0.25)"
      ctx.lineWidth = isActive ? 4 : 2
      ctx.stroke()
      if (isActive) {
        ctx.strokeStyle = "rgba(140,140,140,1)"
        ctx.lineWidth = 2
        ctx.stroke()
      }
    })

    const pathEdgeSet = new Set<string>()
    for (let i = 1; i < path.length; i++) {
      pathEdgeSet.add(`${path[i - 1]}|${path[i]}`)
      pathEdgeSet.add(`${path[i]}|${path[i - 1]}`)
    }

    edges.forEach((edge) => {
      if (!pathEdgeSet.has(`${edge.source}|${edge.target}`)) return
      if (edge.isStair) return
      const src = nodesById.get(edge.source)
      const tgt = nodesById.get(edge.target)
      if (!src || !tgt) return
      if (src.floor != tgt.floor) return
      const isActive = activeFloor ? src.floor === activeFloor : true
      const pts = edge.path.length
        ? edge.path
        : [src, tgt].map((n) => [n.x, n.y] as [number, number])
      if (!pts.length) return
      ctx.beginPath()
      pts.forEach(([x, y], idx) => {
        const px = x * scale
        const py = y * scale + offsetY
        if (idx === 0) ctx.moveTo(px, py)
        else ctx.lineTo(px, py)
      })
      ctx.strokeStyle = isActive ? "rgba(200,0,0,0.9)" : "rgba(200,0,0,0.3)"
      ctx.lineWidth = isActive ? 3 : 2
      ctx.stroke()
    })

    nodes.forEach((node) => {
      const px = node.x * scale
      const py = node.y * scale + offsetY
      const isActive = activeFloor ? node.floor === activeFloor : true
      ctx.beginPath()
      ctx.fillStyle = isActive ? "rgba(240,240,240,0.9)" : "rgba(240,240,240,0.3)"
      ctx.arc(px, py, 8, 0, Math.PI * 2)
      ctx.fill()
      ctx.beginPath()
      ctx.fillStyle = isActive ? "rgba(0,90,255,0.95)" : "rgba(0,90,255,0.35)"
      ctx.arc(px, py, 5, 0, Math.PI * 2)
      ctx.fill()
    })

    if (officerNode) {
      const n = nodesById.get(officerNode)
      if (n) {
        const px = n.x * scale
        const py = n.y * scale + offsetY
        ctx.beginPath()
        ctx.fillStyle = "rgba(0,200,0,0.9)"
        ctx.arc(px, py, 10, 0, Math.PI * 2)
        ctx.fill()
      }
    }
    if (threatNode) {
      const n = nodesById.get(threatNode)
      if (n) {
        const px = n.x * scale
        const py = n.y * scale + offsetY
        ctx.beginPath()
        ctx.fillStyle = "rgba(200,0,0,0.9)"
        ctx.arc(px, py, 10, 0, Math.PI * 2)
        ctx.fill()
      }
    }
  }

  useEffect(() => {
    draw()
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [image, nodes, edges, path, activeFloor, officerNode, threatNode])

  // Auto-assign threat node from backend signals
  useEffect(() => {
    if (autoThreatNode?.id && autoThreatNode.id !== threatNode) {
      setThreatNode(autoThreatNode.id)
      if (officerNode) {
        computePath(officerNode, autoThreatNode.id)
      }
    }
  }, [autoThreatNode, threatNode, officerNode, computePath])

  // External reset clears selections and path
  useEffect(() => {
    if (resetSignal === undefined) return
    setThreatNode(null)
    setOfficerNode(null)
    setPath([])
    setDistance(0)
    setInstructions([])
  }, [resetSignal])

  const handleClick = (event: React.MouseEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current
    const container = containerRef.current
    if (!canvas || !container || !image) return

    const rect = canvas.getBoundingClientRect()
    const clickX = event.clientX - rect.left
    const clickY = event.clientY - rect.top

    const cssWidth = Number(canvas.style.width.replace('px','')) || canvas.width
    const cssHeight = Number(canvas.style.height.replace('px','')) || canvas.height
    const scale = cssWidth / image.width
    const offsetY = (cssHeight - image.height * scale) / 2
    const imgX = clickX / scale
    const imgY = (clickY - offsetY) / scale

    const nodesOnFloor = activeFloor ? nodes.filter((n) => n.floor === activeFloor) : nodes

    let nearest: NodeData | null = null
    let minD = Infinity
    const clickRadius = 30
    nodesOnFloor.forEach((n) => {
      const d = Math.hypot(n.x - imgX, n.y - imgY)
      if (d < minD) {
        minD = d
        nearest = n
      }
    })
    if (!nearest || minD > clickRadius) return

    if (assignMode === "officer") {
      setOfficerNode(nearest.id)
      setAssignMode(null)
      if (threatNode) {
        computePath(nearest.id, threatNode)
      }
      return
    }
    if (assignMode === "threat") {
      setThreatNode(nearest.id)
      setAssignMode(null)
      if (officerNode) {
        computePath(officerNode, nearest.id)
      }
      return
    }
  }

  const officerLabel = officerNode ? prettyLabel(officerNode, nodesById) : "Not set"
  const threatLabel = threatNode ? prettyLabel(threatNode, nodesById) : "Not set"

  return (
    <div ref={containerRef} className="w-full h-full">
      <div className="relative w-full h-full min-h-[420px] rounded-lg overflow-hidden border bg-white">
        {loading && (
          <div className="absolute inset-0 flex items-center justify-center text-sm text-slate-500">
            Loading map...
          </div>
        )}
        {error && (
          <div className="absolute inset-0 flex items-center justify-center text-sm text-red-600">
            {error}
          </div>
        )}
        <canvas
          ref={canvasRef}
          className="w-full h-full"
          onClick={handleClick}
          aria-label="Interactive floor map"
        />

        {floors.length > 1 && (
          <div className="absolute top-3 right-3 bg-white/90 backdrop-blur-sm rounded-md px-2 py-1 text-xs text-slate-700 shadow-sm flex gap-1">
            {floors.map((f) => (
              <button
                key={f}
                onClick={() => setCurrentFloor(f)}
                className={`px-2 py-1 rounded ${f === activeFloor ? "bg-emerald-100 text-emerald-700" : "bg-slate-100 text-slate-600"}`}
              >
                {f}
              </button>
            ))}
          </div>
        )}

        <div className="absolute left-0 right-0 bottom-0 h-[40%] bg-white/95 backdrop-blur-sm border-t border-slate-200 shadow-sm text-xs text-slate-700">
          <div className="h-full px-4 py-3 flex gap-4 overflow-hidden">
            <div className="flex-1 min-w-[200px] h-full space-y-2 overflow-y-auto">
              <div className="text-[11px] uppercase tracking-wide text-slate-500 font-semibold">Map Controls</div>
              <div className="space-y-1">
                <div className="flex items-start gap-1">
                  <span className="text-slate-400">•</span>
                  <span>Choose an assignment button, then click a node.</span>
                </div>
                <div className="flex items-start gap-1">
                  <span className="text-slate-400">•</span>
                  <span>Assign both officer and threat to generate a path.</span>
                </div>
              </div>
            </div>

            <div className="flex-1 min-w-[220px] h-full space-y-2 overflow-y-auto">
              <div className="text-[11px] uppercase tracking-wide text-slate-500 font-semibold">Officer Controls</div>
              <div className="space-y-2">
                <button
                  className={`w-full px-2 py-1 rounded text-left border ${assignMode === "officer" ? "border-emerald-500 bg-emerald-50" : "border-slate-200"}`}
                  onClick={() => setAssignMode("officer")}
                >
                  Manually Assign Officer Location
                </button>
                <button
                  className={`w-full px-2 py-1 rounded text-left border ${assignMode === "threat" ? "border-red-500 bg-red-50" : "border-slate-200"}`}
                  onClick={() => setAssignMode("threat")}
                >
                  Manually Assign Threat Location
                </button>
                <div className="text-[11px] text-slate-600">
                  Officer: <span className="font-semibold">{officerLabel}</span>
                </div>
                <div className="text-[11px] text-slate-600">
                  Threat: <span className="font-semibold text-red-600">{threatLabel}</span>
                </div>
              </div>
            </div>

            <div className="flex-1 min-w-[220px] h-full space-y-2 overflow-y-auto">
              <div className="text-[11px] uppercase tracking-wide text-slate-500 font-semibold">Navigation Instructions</div>
              {instructions.length === 0 ? (
                <div className="text-slate-500">No path computed</div>
              ) : (
                <div className="space-y-1 max-h-full overflow-y-auto overflow-x-auto pr-1">
                  {instructions.map((instruction, idx) => (
                    <div key={idx} className="flex items-start gap-1">
                      <span className="text-slate-400">{idx + 1}.</span>
                      <span>{instruction}</span>
                    </div>
                  ))}
                </div>
              )}
              {distance > 0 && (
                <div className="text-[11px] text-emerald-700 font-semibold">
                  Distance: {distance.toFixed(1)}
                </div>
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
