"use client"

import type React from "react"
import { useCallback, useState, useRef } from "react"
import ReactFlow, {
  type Node,
  addEdge,
  useNodesState,
  useEdgesState,
  type Connection,
  ReactFlowProvider,
  Controls,
  Background,
  MiniMap,
  Panel,
} from "reactflow"
import "reactflow/dist/style.css"

import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import { Play, Save, Upload, Settings } from "lucide-react"
import { NodePalette } from "./node-palette"
import { WorkflowControls } from "./workflow-controls"
import { ExecutionMonitor } from "./execution-monitor"
import { nodeTypes, initialNodes, initialEdges } from "./pipeline-config"

export function PipelineEditor() {
  const [nodes, setNodes, onNodesChange] = useNodesState(initialNodes)
  const [edges, setEdges, onEdgesChange] = useEdgesState(initialEdges)
  const [isExecuting, setIsExecuting] = useState(false)
  const [executionLogs, setExecutionLogs] = useState<string[]>([])
  const [selectedNode, setSelectedNode] = useState<Node | null>(null)
  const reactFlowWrapper = useRef<HTMLDivElement>(null)

  const onConnect = useCallback((params: Connection) => setEdges((eds) => addEdge(params, eds)), [setEdges])

  const onNodeClick = useCallback((_: React.MouseEvent, node: Node) => {
    setSelectedNode(node)
  }, [])

  const onDragOver = useCallback((event: React.DragEvent) => {
    event.preventDefault()
    event.dataTransfer.dropEffect = "move"
  }, [])

  const onDrop = useCallback(
    (event: React.DragEvent) => {
      event.preventDefault()

      const type = event.dataTransfer.getData("application/reactflow")
      if (typeof type === "undefined" || !type) {
        return
      }

      const position = {
        x: event.clientX - 250,
        y: event.clientY - 100,
      }

      const newNode: Node = {
        id: `${type}-${Date.now()}`,
        type,
        position,
        data: {
          label: `${type} node`,
          config: {},
        },
      }

      setNodes((nds) => nds.concat(newNode))
    },
    [setNodes],
  )

  const executeWorkflow = async () => {
    setIsExecuting(true)
    setExecutionLogs([])

    // Simulate workflow execution
    const steps = [
      "Validating workflow structure...",
      "Loading dataset from source...",
      "Applying coordinate transformations...",
      "Running hand tracking model...",
      "Normalizing pose data...",
      "Training VAE model...",
      "Encoding dataset with VAE...",
      "Training policy model...",
      "Saving results...",
      "Workflow completed successfully!",
    ]

    for (let i = 0; i < steps.length; i++) {
      await new Promise((resolve) => setTimeout(resolve, 1000))
      setExecutionLogs((prev) => [...prev, `[${new Date().toLocaleTimeString()}] ${steps[i]}`])
    }

    setIsExecuting(false)
  }

  const saveWorkflow = () => {
    const workflow = {
      nodes,
      edges,
      metadata: {
        name: "EgoHub Pipeline",
        created: new Date().toISOString(),
        version: "1.0.0",
      },
    }

    const blob = new Blob([JSON.stringify(workflow, null, 2)], {
      type: "application/json",
    })
    const url = URL.createObjectURL(blob)
    const a = document.createElement("a")
    a.href = url
    a.download = "egohub-pipeline.json"
    a.click()
    URL.revokeObjectURL(url)
  }

  return (
    <div className="flex h-screen bg-gray-50">
      {/* Node Palette */}
      <div className="w-64 bg-white border-r border-gray-200 flex flex-col">
        <div className="p-4 border-b border-gray-200">
          <h2 className="text-lg font-semibold text-gray-900">Pipeline Builder</h2>
          <p className="text-sm text-gray-600 mt-1">Drag nodes to build your workflow</p>
        </div>
        <NodePalette />
      </div>

      {/* Main Editor */}
      <div className="flex-1 flex flex-col">
        {/* Toolbar */}
        <div className="bg-white border-b border-gray-200 p-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-2">
              <Button onClick={executeWorkflow} disabled={isExecuting} className="bg-green-600 hover:bg-green-700">
                <Play className="w-4 h-4 mr-2" />
                {isExecuting ? "Executing..." : "Run Pipeline"}
              </Button>
              <Button variant="outline" onClick={saveWorkflow}>
                <Save className="w-4 h-4 mr-2" />
                Save
              </Button>
              <Button variant="outline">
                <Upload className="w-4 h-4 mr-2" />
                Load
              </Button>
            </div>
            <div className="flex items-center space-x-2">
              <Badge variant="secondary">
                {nodes.length} nodes, {edges.length} connections
              </Badge>
              <Button variant="outline" size="sm">
                <Settings className="w-4 h-4" />
              </Button>
            </div>
          </div>
        </div>

        {/* Flow Editor */}
        <div className="flex-1" ref={reactFlowWrapper}>
          <ReactFlowProvider>
            <ReactFlow
              nodes={nodes}
              edges={edges}
              onNodesChange={onNodesChange}
              onEdgesChange={onEdgesChange}
              onConnect={onConnect}
              onNodeClick={onNodeClick}
              onDrop={onDrop}
              onDragOver={onDragOver}
              nodeTypes={nodeTypes}
              fitView
              className="bg-gray-50"
            >
              <Background />
              <Controls />
              <MiniMap />
              <Panel position="top-right">
                <WorkflowControls isExecuting={isExecuting} onExecute={executeWorkflow} />
              </Panel>
            </ReactFlow>
          </ReactFlowProvider>
        </div>
      </div>

      {/* Execution Monitor */}
      <div className="w-80 bg-white border-l border-gray-200">
        <ExecutionMonitor logs={executionLogs} isExecuting={isExecuting} selectedNode={selectedNode} />
      </div>
    </div>
  )
}
