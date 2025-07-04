"use client"

import type React from "react"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { ScrollArea } from "@/components/ui/scroll-area"
import {
  Database,
  ReplaceIcon as Transform,
  Brain,
  Save,
  Camera,
  Hand,
  Eye,
  BarChart3,
  FileText,
  Layers,
} from "lucide-react"

const nodeCategories = [
  {
    title: "Data Sources",
    color: "bg-blue-100 text-blue-800",
    nodes: [
      { type: "dataSource", label: "Load Dataset", icon: Database, description: "Load raw dataset from file or URL" },
      { type: "dataSource", label: "Camera Stream", icon: Camera, description: "Real-time camera input" },
    ],
  },
  {
    title: "Transforms",
    color: "bg-green-100 text-green-800",
    nodes: [
      {
        type: "transform",
        label: "Coordinate Transform",
        icon: Transform,
        description: "Convert between coordinate systems",
      },
      { type: "transform", label: "Hand Tracking", icon: Hand, description: "Estimate hand poses using ML models" },
      { type: "transform", label: "Depth Estimation", icon: Eye, description: "Generate depth maps from RGB" },
      {
        type: "transform",
        label: "Pose Normalization",
        icon: Layers,
        description: "Normalize poses to canonical format",
      },
    ],
  },
  {
    title: "Models",
    color: "bg-purple-100 text-purple-800",
    nodes: [
      { type: "model", label: "Train VAE", icon: Brain, description: "Train variational autoencoder" },
      { type: "model", label: "Encode Dataset", icon: BarChart3, description: "Encode data using trained VAE" },
      { type: "model", label: "Train Policy", icon: Brain, description: "Train policy model" },
    ],
  },
  {
    title: "Outputs",
    color: "bg-orange-100 text-orange-800",
    nodes: [
      { type: "output", label: "Save HDF5", icon: Save, description: "Save processed data to HDF5" },
      { type: "output", label: "Export Model", icon: FileText, description: "Export trained model" },
      { type: "output", label: "Visualize", icon: Eye, description: "Visualize data with Rerun" },
    ],
  },
]

export function NodePalette() {
  const onDragStart = (event: React.DragEvent, nodeType: string) => {
    event.dataTransfer.setData("application/reactflow", nodeType)
    event.dataTransfer.effectAllowed = "move"
  }

  return (
    <ScrollArea className="flex-1 p-4">
      <div className="space-y-4">
        {nodeCategories.map((category) => (
          <Card key={category.title} className="border-gray-200">
            <CardHeader className="pb-2">
              <CardTitle className="text-sm font-medium text-gray-700">{category.title}</CardTitle>
            </CardHeader>
            <CardContent className="space-y-2">
              {category.nodes.map((node, index) => (
                <div
                  key={index}
                  className="flex items-center p-2 rounded-lg border border-gray-200 cursor-move hover:bg-gray-50 transition-colors"
                  draggable
                  onDragStart={(event) => onDragStart(event, node.type)}
                >
                  <node.icon className="w-4 h-4 text-gray-600 mr-2 flex-shrink-0" />
                  <div className="flex-1 min-w-0">
                    <div className="text-sm font-medium text-gray-900 truncate">{node.label}</div>
                    <div className="text-xs text-gray-500 truncate">{node.description}</div>
                  </div>
                  <Badge variant="secondary" className={`text-xs ${category.color} ml-2`}>
                    {category.title.slice(0, -1)}
                  </Badge>
                </div>
              ))}
            </CardContent>
          </Card>
        ))}
      </div>
    </ScrollArea>
  )
}
