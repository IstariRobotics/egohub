"use client"

import { Handle, Position } from "@xyflow/react"
import { Card, CardContent } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Database, Workflow, Brain, FileOutput, Settings } from "lucide-react"

const getNodeIcon = (nodeType: string) => {
  switch (nodeType) {
    case "source":
      return Database
    case "transform":
      return Workflow
    case "model":
      return Brain
    case "output":
      return FileOutput
    default:
      return Settings
  }
}

const getNodeColor = (nodeType: string) => {
  switch (nodeType) {
    case "source":
      return "border-blue-200 bg-blue-50"
    case "transform":
      return "border-green-200 bg-green-50"
    case "model":
      return "border-purple-200 bg-purple-50"
    case "output":
      return "border-orange-200 bg-orange-50"
    default:
      return "border-gray-200 bg-gray-50"
  }
}

const getBadgeColor = (nodeType: string) => {
  switch (nodeType) {
    case "source":
      return "bg-blue-100 text-blue-800"
    case "transform":
      return "bg-green-100 text-green-800"
    case "model":
      return "bg-purple-100 text-purple-800"
    case "output":
      return "bg-orange-100 text-orange-800"
    default:
      return "bg-gray-100 text-gray-800"
  }
}

export function CustomNode({ data, selected }: { data: any; selected?: boolean }) {
  const Icon = getNodeIcon(data.nodeType)

  return (
    <Card className={`w-48 ${getNodeColor(data.nodeType)} ${selected ? "ring-2 ring-blue-500" : ""}`}>
      <CardContent className="p-3">
        <div className="flex items-center space-x-2 mb-2">
          <Icon className="w-4 h-4" />
          <Badge variant="secondary" className={`text-xs ${getBadgeColor(data.nodeType)}`}>
            {data.nodeType}
          </Badge>
        </div>
        <div className="text-sm font-medium text-gray-900 mb-1">{data.label}</div>
        {data.config && Object.keys(data.config).length > 0 && (
          <div className="text-xs text-gray-600">
            {Object.entries(data.config)
              .slice(0, 2)
              .map(([key, value]) => (
                <div key={key}>
                  {key}: {String(value)}
                </div>
              ))}
          </div>
        )}
      </CardContent>

      {data.nodeType !== "source" && <Handle type="target" position={Position.Left} className="bg-gray-400" />}
      {data.nodeType !== "output" && <Handle type="source" position={Position.Right} className="bg-gray-400" />}
    </Card>
  )
}
