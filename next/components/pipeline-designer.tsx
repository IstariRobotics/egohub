"use client"

import type React from "react"

import { useCallback, useState, useRef } from "react"
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
  Panel,
} from "@xyflow/react"

import "@xyflow/react/dist/style.css"

import { Button } from "@/components/ui/button"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Separator } from "@/components/ui/separator"
import { ScrollArea } from "@/components/ui/scroll-area"
import { Play, Save, Trash2, Database, Brain, FileOutput, Hand, Eye, RotateCcw } from "lucide-react"

import { CustomNode } from "./custom-node"

const nodeTypes = {
  custom: CustomNode,
}

const initialNodes: Node[] = [
  {
    id: "1",
    type: "custom",
    position: { x: 100, y: 100 },
    data: {
      label: "Load Dataset",
      nodeType: "source",
      config: {
        path: "/data/egodex-v1.h5",
        format: "hdf5",
      },
    },
  },
  {
    id: "2",
    type: "custom",
    position: { x: 400, y: 100 },
    data: {
      label: "Hand Tracking",
      nodeType: "transform",
      config: {
        model: "mediapipe/hands",
        confidence: 0.8,
      },
    },
  },
  {
    id: "3",
    type: "custom",
    position: { x: 700, y: 100 },
    data: {
      label: "Save Output",
      nodeType: "output",
      config: {
        format: "hdf5",
        path: "/output/processed.h5",
      },
    },
  },
]

const initialEdges: Edge[] = [
  { id: "e1-2", source: "1", target: "2", animated: true },
  { id: "e2-3", source: "2", target: "3", animated: true },
]

const nodePalette = [
  { type: "source", label: "Load Dataset", icon: Database, color: "blue" },
  { type: "transform", label: "Hand Tracking", icon: Hand, color: "green" },
  { type: "transform", label: "Pose Transform", icon: RotateCcw, color: "green" },
  { type: "transform", label: "Depth Estimation", icon: Eye, color: "green" },
  { type: "model", label: "Train VAE", icon: Brain, color: "purple" },
  { type: "model", label: "Train Policy", icon: Brain, color: "purple" },
  { type: "output", label: "Save HDF5", icon: FileOutput, color: "orange" },
  { type: "output", label: "Export Model", icon: FileOutput, color: "orange" },
]

interface PipelineDesignerProps {
  dataset: any
}

export function PipelineDesigner({ dataset }: PipelineDesignerProps) {
  const [nodes, setNodes, onNodesChange] = useNodesState(initialNodes)
  const [edges, setEdges, onEdgesChange] = useEdgesState(initialEdges)
  const [selectedNode, setSelectedNode] = useState<Node | null>(null)
  const [isExecuting, setIsExecuting] = useState(false)
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

      const nodeData = event.dataTransfer.getData("application/reactflow")
      if (!nodeData) return

      const { type, label } = JSON.parse(nodeData)
      const position = {
        x: event.clientX - 300,
        y: event.clientY - 100,
      }

      const newNode: Node = {
        id: `${Date.now()}`,
        type: "custom",
        position,
        data: {
          label,
          nodeType: type,
          config: {},
        },
      }

      setNodes((nds) => nds.concat(newNode))
    },
    [setNodes],
  )

  const deleteNode = useCallback(() => {
    if (selectedNode) {
      setNodes((nds) => nds.filter((node) => node.id !== selectedNode.id))
      setEdges((eds) => eds.filter((edge) => edge.source !== selectedNode.id && edge.target !== selectedNode.id))
      setSelectedNode(null)
    }
  }, [selectedNode, setNodes, setEdges])

  const updateNodeConfig = useCallback(
    (config: any) => {
      if (selectedNode) {
        setNodes((nds) =>
          nds.map((node) => (node.id === selectedNode.id ? { ...node, data: { ...node.data, config } } : node)),
        )
        setSelectedNode({ ...selectedNode, data: { ...selectedNode.data, config } })
      }
    },
    [selectedNode, setNodes],
  )

  const executePipeline = async () => {
    setIsExecuting(true)
    // Simulate execution
    await new Promise((resolve) => setTimeout(resolve, 3000))
    setIsExecuting(false)
  }

  return (
    <div className="flex h-full">
      {/* Node Palette */}
      <div className="w-64 bg-white border-r border-gray-200 flex flex-col">
        <div className="p-4 border-b border-gray-200">
          <h3 className="text-sm font-medium text-gray-900 mb-2">Node Palette</h3>
          <p className="text-xs text-gray-600">Drag nodes to the canvas</p>
        </div>

        <ScrollArea className="flex-1 p-4">
          <div className="space-y-2">
            {nodePalette.map((item, index) => (
              <div
                key={index}
                className="flex items-center p-2 rounded-lg border border-gray-200 cursor-move hover:bg-gray-50 transition-colors"
                draggable
                onDragStart={(event) => {
                  event.dataTransfer.setData(
                    "application/reactflow",
                    JSON.stringify({ type: item.type, label: item.label }),
                  )
                  event.dataTransfer.effectAllowed = "move"
                }}
              >
                <item.icon className="w-4 h-4 text-gray-600 mr-2" />
                <span className="text-sm text-gray-900">{item.label}</span>
                <Badge variant="secondary" className={`ml-auto text-xs bg-${item.color}-100 text-${item.color}-800`}>
                  {item.type}
                </Badge>
              </div>
            ))}
          </div>
        </ScrollArea>
      </div>

      {/* Main Canvas */}
      <div className="flex-1 flex flex-col">
        {/* Toolbar */}
        <div className="bg-white border-b border-gray-200 p-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-2">
              <Button onClick={executePipeline} disabled={isExecuting} className="bg-green-600 hover:bg-green-700">
                <Play className="w-4 h-4 mr-2" />
                {isExecuting ? "Running..." : "Run Pipeline"}
              </Button>
              <Button variant="outline">
                <Save className="w-4 h-4 mr-2" />
                Save
              </Button>
            </div>
            <div className="flex items-center space-x-2">
              <Badge variant="secondary">
                {nodes.length} nodes, {edges.length} connections
              </Badge>
            </div>
          </div>
        </div>

        {/* Flow Canvas */}
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
                <Card className="w-64">
                  <CardHeader className="pb-2">
                    <CardTitle className="text-sm">Pipeline Status</CardTitle>
                  </CardHeader>
                  <CardContent>
                    <Badge variant={isExecuting ? "default" : "secondary"}>{isExecuting ? "Running" : "Ready"}</Badge>
                  </CardContent>
                </Card>
              </Panel>
            </ReactFlow>
          </ReactFlowProvider>
        </div>
      </div>

      {/* Properties Panel */}
      <div className="w-80 bg-white border-l border-gray-200 flex flex-col">
        <div className="p-4 border-b border-gray-200">
          <h3 className="text-sm font-medium text-gray-900 mb-2">Node Properties</h3>
          {selectedNode && (
            <div className="flex items-center justify-between">
              <Badge variant="outline">{selectedNode.data.nodeType}</Badge>
              <Button
                variant="outline"
                size="sm"
                onClick={deleteNode}
                className="text-red-600 hover:text-red-700 bg-transparent"
              >
                <Trash2 className="w-3 h-3" />
              </Button>
            </div>
          )}
        </div>

        <div className="flex-1 p-4">
          {selectedNode ? (
            <div className="space-y-4">
              <div>
                <Label className="text-xs text-gray-600">Node Label</Label>
                <Input
                  value={selectedNode.data.label}
                  className="mt-1"
                  onChange={(e) => {
                    const updatedNode = {
                      ...selectedNode,
                      data: { ...selectedNode.data, label: e.target.value },
                    }
                    setSelectedNode(updatedNode)
                    setNodes((nds) => nds.map((node) => (node.id === selectedNode.id ? updatedNode : node)))
                  }}
                />
              </div>

              <Separator />

              {selectedNode.data.nodeType === "transform" && (
                <div className="space-y-3">
                  <div>
                    <Label className="text-xs text-gray-600">Model</Label>
                    <Select
                      value={selectedNode.data.config?.model || ""}
                      onValueChange={(value) => updateNodeConfig({ ...selectedNode.data.config, model: value })}
                    >
                      <SelectTrigger className="mt-1">
                        <SelectValue placeholder="Select model" />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="mediapipe/hands">MediaPipe Hands</SelectItem>
                        <SelectItem value="openpose">OpenPose</SelectItem>
                        <SelectItem value="custom/hand-v1">Custom Hand Model</SelectItem>
                      </SelectContent>
                    </Select>
                  </div>
                  <div>
                    <Label className="text-xs text-gray-600">Confidence Threshold</Label>
                    <Input
                      type="number"
                      step="0.1"
                      min="0"
                      max="1"
                      value={selectedNode.data.config?.confidence || 0.8}
                      className="mt-1"
                      onChange={(e) =>
                        updateNodeConfig({
                          ...selectedNode.data.config,
                          confidence: Number.parseFloat(e.target.value),
                        })
                      }
                    />
                  </div>
                </div>
              )}

              {selectedNode.data.nodeType === "model" && (
                <div className="space-y-3">
                  <div>
                    <Label className="text-xs text-gray-600">Epochs</Label>
                    <Input
                      type="number"
                      value={selectedNode.data.config?.epochs || 100}
                      className="mt-1"
                      onChange={(e) =>
                        updateNodeConfig({
                          ...selectedNode.data.config,
                          epochs: Number.parseInt(e.target.value),
                        })
                      }
                    />
                  </div>
                  <div>
                    <Label className="text-xs text-gray-600">Learning Rate</Label>
                    <Input
                      type="number"
                      step="0.0001"
                      value={selectedNode.data.config?.learningRate || 0.001}
                      className="mt-1"
                      onChange={(e) =>
                        updateNodeConfig({
                          ...selectedNode.data.config,
                          learningRate: Number.parseFloat(e.target.value),
                        })
                      }
                    />
                  </div>
                </div>
              )}

              {selectedNode.data.nodeType === "output" && (
                <div className="space-y-3">
                  <div>
                    <Label className="text-xs text-gray-600">Output Path</Label>
                    <Input
                      value={selectedNode.data.config?.path || ""}
                      className="mt-1"
                      onChange={(e) =>
                        updateNodeConfig({
                          ...selectedNode.data.config,
                          path: e.target.value,
                        })
                      }
                    />
                  </div>
                  <div>
                    <Label className="text-xs text-gray-600">Format</Label>
                    <Select
                      value={selectedNode.data.config?.format || "hdf5"}
                      onValueChange={(value) => updateNodeConfig({ ...selectedNode.data.config, format: value })}
                    >
                      <SelectTrigger className="mt-1">
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="hdf5">HDF5</SelectItem>
                        <SelectItem value="json">JSON</SelectItem>
                        <SelectItem value="pickle">Pickle</SelectItem>
                      </SelectContent>
                    </Select>
                  </div>
                </div>
              )}
            </div>
          ) : (
            <p className="text-sm text-gray-500">Select a node to edit its properties</p>
          )}
        </div>
      </div>
    </div>
  )
}
