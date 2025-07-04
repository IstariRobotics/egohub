"use client"
import { Handle, Position } from "reactflow"
import { Card, CardContent, CardHeader } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Database, Folder } from "lucide-react"

export function DataSourceNode({ data }: { data: any }) {
  return (
    <Card className="w-48 border-blue-200 bg-blue-50">
      <CardHeader className="pb-2">
        <div className="flex items-center space-x-2">
          <Database className="w-4 h-4 text-blue-600" />
          <Badge variant="secondary" className="bg-blue-100 text-blue-800 text-xs">
            Data Source
          </Badge>
        </div>
      </CardHeader>
      <CardContent className="pt-0">
        <div className="text-sm font-medium text-gray-900 mb-1">{data.label || "Load Dataset"}</div>
        <div className="text-xs text-gray-600 flex items-center">
          <Folder className="w-3 h-3 mr-1" />
          /path/to/dataset.h5
        </div>
      </CardContent>
      <Handle type="source" position={Position.Right} className="bg-blue-500" />
    </Card>
  )
}
