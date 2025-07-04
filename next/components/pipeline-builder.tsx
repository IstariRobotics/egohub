"use client"

import { useCallback } from "react"
import {
  ReactFlow,
  ReactFlowProvider,
  MiniMap,
  Controls,
  Background,
  addEdge,
  type Connection,
  type Edge,
  type Node,
  useNodesState,
  useEdgesState,
} from "@xyflow/react"

import "@xyflow/react/dist/style.css"

const initialNodes: Node[] = [
  {
    id: "dataset",
    position: { x: 0, y: 50 },
    data: { label: "Load Dataset" },
    type: "input",
  },
  {
    id: "transform",
    position: { x: 250, y: 50 },
    data: { label: "Transform\n(eg. NormalizeRig)" },
  },
  {
    id: "output",
    position: { x: 500, y: 50 },
    data: { label: "Save ↴" },
    type: "output",
  },
]

const initialEdges: Edge[] = [
  { id: "e1", source: "dataset", target: "transform", animated: true },
  { id: "e2", source: "transform", target: "output", animated: true },
]

export default function PipelineBuilder() {
  //   OLD (delete)
  //   const [nodes, setNodes] = React.useState<Node[]>(initialNodes)
  //   const [edges, setEdges] = React.useState<Edge[]>(initialEdges)

  const [nodes, setNodes, onNodesChange] = useNodesState(initialNodes)
  const [edges, setEdges, onEdgesChange] = useEdgesState(initialEdges)

  const onConnect = useCallback((params: Edge | Connection) => setEdges((eds) => addEdge(params, eds)), [])

  return (
    <div className="h-[80vh] w-full border rounded-md">
      <ReactFlowProvider>
        <ReactFlow
          nodes={nodes}
          edges={edges}
          onNodesChange={onNodesChange}
          onEdgesChange={onEdgesChange}
          onConnect={onConnect}
          fitView
        >
          <MiniMap />
          <Controls />
          <Background gap={16} />
        </ReactFlow>
      </ReactFlowProvider>
    </div>
  )
}
