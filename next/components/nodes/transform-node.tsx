"use client"
import { Handle, Position } from "reactflow"
import { Card, CardContent, CardHeader } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { ReplaceIcon as Transform, Settings } from "lucide-react"

export function TransformNode({ data }: { data: any }) {
  return (
    <Card className="w-48 border-green-200 bg-green-50">
      <CardHeader className="pb-2">
        <div className="flex items-center space-x-2">
          <Transform className="w-4 h-4 text-green-600" />
          <Badge variant="secondary" className="bg-green-100 text-green-800 text-xs">
            Transform
          </Badge>
        </div>
      </CardHeader>
      <CardContent className="pt-0">
        <div className="text-sm font-medium text-gray-900 mb-1">{data.label || "Transform Data"}</div>
        <div className="text-xs text-gray-600 flex items-center">
          <Settings className="w-3 h-3 mr-1" />
          Configure parameters
        </div>
      </CardContent>
      <Handle type="target" position={Position.Left} className="bg-green-500" />
      <Handle type="source" position={Position.Right} className="bg-green-500" />
    </Card>
  )
}
