"use client"
import { Button } from "@/components/ui/button"
import { Card, CardContent } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Play, Square, RotateCcw } from "lucide-react"

interface WorkflowControlsProps {
  isExecuting: boolean
  onExecute: () => void
}

export function WorkflowControls({ isExecuting, onExecute }: WorkflowControlsProps) {
  return (
    <Card className="w-64">
      <CardContent className="p-3">
        <div className="flex items-center justify-between mb-2">
          <Badge variant="outline" className="text-xs">
            Pipeline Status
          </Badge>
          <Badge variant={isExecuting ? "default" : "secondary"} className={isExecuting ? "bg-green-600" : ""}>
            {isExecuting ? "Running" : "Ready"}
          </Badge>
        </div>
        <div className="flex space-x-2">
          <Button size="sm" onClick={onExecute} disabled={isExecuting} className="flex-1">
            <Play className="w-3 h-3 mr-1" />
            Run
          </Button>
          <Button size="sm" variant="outline" disabled={!isExecuting}>
            <Square className="w-3 h-3" />
          </Button>
          <Button size="sm" variant="outline">
            <RotateCcw className="w-3 h-3" />
          </Button>
        </div>
      </CardContent>
    </Card>
  )
}
