"use client"
import { Handle, Position } from "reactflow"
import { Card, CardContent, CardHeader } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Brain, Zap } from "lucide-react"

export function ModelNode({ data }: { data: any }) {
  return (
    <Card className="w-48 border-purple-200 bg-purple-50">
      <CardHeader className="pb-2">
        <div className="flex items-center space-x-2">
          <Brain className="w-4 h-4 text-purple-600" />
          <Badge variant="secondary" className="bg-purple-100 text-purple-800 text-xs">
            Model
          </Badge>
        </div>
      </CardHeader>
      <CardContent className="pt-0">
        <div className="text-sm font-medium text-gray-900 mb-1">{data.label || "Train Model"}</div>
        <div className="text-xs text-gray-600 flex items-center">
          <Zap className="w-3 h-3 mr-1" />
          GPU accelerated
        </div>
      </CardContent>
      <Handle type="target" position={Position.Left} className="bg-purple-500" />
      <Handle type="source" position={Position.Right} className="bg-purple-500" />
    </Card>
  )
}
