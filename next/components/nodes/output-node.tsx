"use client"
import { Handle, Position } from "reactflow"
import { Card, CardContent, CardHeader } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Save, FileCheck } from "lucide-react"

export function OutputNode({ data }: { data: any }) {
  return (
    <Card className="w-48 border-orange-200 bg-orange-50">
      <CardHeader className="pb-2">
        <div className="flex items-center space-x-2">
          <Save className="w-4 h-4 text-orange-600" />
          <Badge variant="secondary" className="bg-orange-100 text-orange-800 text-xs">
            Output
          </Badge>
        </div>
      </CardHeader>
      <CardContent className="pt-0">
        <div className="text-sm font-medium text-gray-900 mb-1">{data.label || "Save Output"}</div>
        <div className="text-xs text-gray-600 flex items-center">
          <FileCheck className="w-3 h-3 mr-1" />
          Ready to export
        </div>
      </CardContent>
      <Handle type="target" position={Position.Left} className="bg-orange-500" />
    </Card>
  )
}
