"use client"

import { useEffect, useMemo, useRef, useState } from "react"

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
}

const GEXF_PATH = "/BMHS_FloorPlan_combined.gexf"
const IMAGE_PATH = "/BMHS_FloorPlan.JPG"

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
  xml.querySelectorAll("node").forEach((node) => {
    const id = node.getAttribute("id") || ""
    const label = node.getAttribute("label") || id
    const attrValues = new Map<string, string>()
    node.querySelectorAll("attvalue").forEach((av) => {
      const key = av.getAttribute("for") || ""
      const title = nodeAttrMap.get(key) || key
      const val = av.getAttribute("value") || ""
      attrValues.set(title, val)
    })
    nodes.push({
      id,
      label,
      x: Number.parseFloat(attrValues.get("pos_x") || "0"),
      y: Number.parseFloat(attrValues.get("pos_y") || "0"),
      floor: attrValues.get("floor") || "Unknown",
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
  instructions.push(`Start at ${nodesById.get(path[0])?.label || path[0]}`)
  for (let i = 1; i < path.length; i++) {
    const from = path[i - 1]
    const to = path[i]
    const key = `${from}|${to}`
    const edge = edgesByKey.get(key) || edgesByKey.get(`${to}|${from}`)
    const toLabel = nodesById.get(to)?.label || to
    const fromNode = nodesById.get(from)
    const toNode = nodesById.get(to)
    if (edge?.isStair && fromNode && toNode && fromNode.floor !== toNode.floor) {
      const goingUp = Number.parseInt(toNode.floor.replace(/\D+/g, "")) > Number.parseInt(fromNode.floor.replace(/\D+/g, ""))
      instructions.push(goingUp ? `Go up the staircase to ${toLabel}` : `Go down the staircase to ${toLabel}`)
    } else {
      instructions.push(`Move to ${toLabel}`)
    }
  }
  instructions.push(`Arrive at ${nodesById.get(path[path.length - 1])?.label || path[path.length - 1]}`)
  return instructions
}

export default function PathfindingMap({ onPathComputed }: PathfindingMapProps) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null)
  const containerRef = useRef<HTMLDivElement | null>(null)
  const [image, setImage] = useState<HTMLImageElement | null>(null)
  const [nodes, setNodes] = useState<NodeData[]>([])
  const [edges, setEdges] = useState<EdgeData[]>([])
  const [selected, setSelected] = useState<{ start?: string; end?: string }>({})
  const [path, setPath] = useState<string[]>([])
  const [distance, setDistance] = useState(0)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [currentFloor, setCurrentFloor] = useState<string | null>(null)

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
        if (!currentFloor && parsed.nodes.length > 0) {
          const floorList = Array.from(new Set(parsed.nodes.map((n) => n.floor))).sort()
          setCurrentFloor(floorList[0] || null)
        }
        setLoading(false)
      })
      .catch(() => setError("Failed to load map data"))
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  useEffect(() => {
    const handleResize = () => draw()
    window.addEventListener("resize", handleResize)
    return () => window.removeEventListener("resize", handleResize)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [image, nodes, edges, path, selected, activeFloor])

  const draw = () => {
    const canvas = canvasRef.current
    const container = containerRef.current
    if (!canvas || !container || !image) return

    const containerWidth = container.clientWidth
    const aspect = image.height ? image.height / image.width : 1
    canvas.width = containerWidth
    canvas.height = Math.max(containerWidth * aspect, 420)

    const ctx = canvas.getContext("2d")
    if (!ctx) return

    ctx.clearRect(0, 0, canvas.width, canvas.height)

    const scale = canvas.width / image.width
    const offsetY = (canvas.height - image.height * scale) / 2

    ctx.drawImage(image, 0, offsetY, image.width * scale, image.height * scale)

    ctx.lineJoin = "round"
    ctx.lineCap = "round"

    const nodesOnFloor = activeFloor ? nodes.filter((n) => n.floor === activeFloor) : nodes
    const nodeSet = new Set(nodesOnFloor.map((n) => n.id))
    const edgesOnFloor = edges.filter(
      (edge) => nodeSet.has(edge.source) && nodeSet.has(edge.target) && !edge.isStair,
    )

    edgesOnFloor.forEach((edge) => {
      const pts = edge.path.length
        ? edge.path
        : [nodesById.get(edge.source), nodesById.get(edge.target)]
            .filter(Boolean)
            .map((n) => [n!.x, n!.y])
      if (!pts.length) return
      ctx.beginPath()
      pts.forEach(([x, y], idx) => {
        const px = x * scale
        const py = y * scale + offsetY
        if (idx === 0) ctx.moveTo(px, py)
        else ctx.lineTo(px, py)
      })
      ctx.strokeStyle = "rgba(160,160,160,0.9)"
      ctx.lineWidth = 4
      ctx.stroke()
      ctx.strokeStyle = "rgba(140,140,140,1)"
      ctx.lineWidth = 2
      ctx.stroke()
    })

    const pathEdgeSet = new Set<string>()
    for (let i = 1; i < path.length; i++) {
      pathEdgeSet.add(`${path[i - 1]}|${path[i]}`)
      pathEdgeSet.add(`${path[i]}|${path[i - 1]}`)
    }

    edgesOnFloor.forEach((edge) => {
      if (!pathEdgeSet.has(`${edge.source}|${edge.target}`)) return
      const pts = edge.path.length
        ? edge.path
        : [nodesById.get(edge.source), nodesById.get(edge.target)]
            .filter(Boolean)
            .map((n) => [n!.x, n!.y])
      if (!pts.length) return
      ctx.beginPath()
      pts.forEach(([x, y], idx) => {
        const px = x * scale
        const py = y * scale + offsetY
        if (idx === 0) ctx.moveTo(px, py)
        else ctx.lineTo(px, py)
      })
      ctx.strokeStyle = "rgba(0,0,180,0.5)"
      ctx.lineWidth = 7
      ctx.stroke()
      ctx.strokeStyle = "rgba(200,0,0,0.9)"
      ctx.lineWidth = 3
      ctx.stroke()
    })

    nodesOnFloor.forEach((node) => {
      const px = node.x * scale
      const py = node.y * scale + offsetY
      ctx.beginPath()
      ctx.fillStyle = "rgba(240,240,240,0.9)"
      ctx.arc(px, py, 8, 0, Math.PI * 2)
      ctx.fill()
      ctx.beginPath()
      ctx.fillStyle = "rgba(0,90,255,0.95)"
      ctx.arc(px, py, 5, 0, Math.PI * 2)
      ctx.fill()
    })

    if (selected.start) {
      const n = nodesById.get(selected.start)
      if (n && (!activeFloor || n.floor === activeFloor)) {
        const px = n.x * scale
        const py = n.y * scale + offsetY
        ctx.beginPath()
        ctx.fillStyle = "rgba(0,200,0,0.9)"
        ctx.arc(px, py, 10, 0, Math.PI * 2)
        ctx.fill()
      }
    }
    if (selected.end) {
      const n = nodesById.get(selected.end)
      if (n && (!activeFloor || n.floor === activeFloor)) {
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
  }, [image, nodes, edges, selected, path, activeFloor])

  const handleClick = (event: React.MouseEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current
    const container = containerRef.current
    if (!canvas || !container || !image) return

    const rect = canvas.getBoundingClientRect()
    const clickX = event.clientX - rect.left
    const clickY = event.clientY - rect.top

    const scale = canvas.width / image.width
    const offsetY = (canvas.height - image.height * scale) / 2
    const imgX = clickX / scale
    const imgY = (clickY - offsetY) / scale

    const nodesOnFloor = activeFloor ? nodes.filter((n) => n.floor === activeFloor) : nodes

    let nearest: NodeData | null = null
    let minD = Infinity
    nodesOnFloor.forEach((n) => {
      const d = Math.hypot(n.x - imgX, n.y - imgY)
      if (d < minD) {
        minD = d
        nearest = n
      }
    })
    if (!nearest || minD > 25) return

    if (!selected.start || (selected.start && selected.end)) {
      setSelected({ start: nearest.id, end: undefined })
      setPath([])
      if (onPathComputed) onPathComputed([], 0)
      return
    }

    const startId = selected.start
    const endId = nearest.id
    setSelected({ start: startId, end: endId })

    const result = dijkstra(nodes, edges, startId, endId)
    if (!result) {
      setPath([])
      setDistance(0)
      if (onPathComputed) onPathComputed(["No path found"], 0)
      return
    }

    setPath(result.path)
    setDistance(result.distance)
    const instructions = buildInstructions(result.path, nodesById, edgesByKey)
    if (onPathComputed) onPathComputed(instructions, result.distance)
  }

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
        <div className="absolute bottom-3 left-3 bg-white/90 backdrop-blur-sm rounded-md px-3 py-2 text-xs text-slate-700 shadow-sm">
          <div className="font-semibold mb-1">Map controls</div>
          <div>• Click once to set start</div>
          <div>• Click again to set destination</div>
        </div>
        {path.length > 0 && (
          <div className="absolute bottom-3 right-3 bg-white/90 backdrop-blur-sm rounded-md px-3 py-2 text-xs text-slate-700 shadow-sm">
            <div className="font-semibold mb-1">Path length: {distance.toFixed(1)}</div>
            <div>{path.length} steps</div>
          </div>
        )}
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
      </div>
    </div>
  )
}
