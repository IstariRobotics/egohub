"use client"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { ScrollArea } from "@/components/ui/scroll-area"
import { Separator } from "@/components/ui/separator"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Activity, CheckCircle, XCircle, Clock, Settings, Terminal } from "lucide-react"
import type { Node } from "reactflow"

interface ExecutionMonitorProps {
  logs: string[]
  isExecuting: boolean
  selectedNode: Node | null
}

export function ExecutionMonitor({ logs, isExecuting, selectedNode }: ExecutionMonitorProps) {
  return (
    <div className="flex flex-col h-full">
      {/* Node Configuration */}
      <Card className="m-4 mb-2">
        <CardHeader className="pb-2">
          <CardTitle className="text-sm font-medium flex items-center">
            <Settings className="w-4 h-4 mr-2" />
            Node Configuration
          </CardTitle>
        </CardHeader>
        <CardContent>
          {selectedNode ? (
            <div className="space-y-3">
              <div>
                <Label className="text-xs text-gray-600">Node Type</Label>
                <Badge variant="outline" className="mt-1">
                  {selectedNode.type}
                </Badge>
              </div>
              <div>
                <Label className="text-xs text-gray-600">Node ID</Label>
                <Input value={selectedNode.id} className="h-8 text-xs" readOnly />
              </div>
              {selectedNode.type === "transform" && (
                <>
                  <div>
                    <Label className="text-xs text-gray-600">Model</Label>
                    <Input placeholder="huggingface/hand-pose-v1" className="h-8 text-xs" />
                  </div>
                  <div>
                    <Label className="text-xs text-gray-600">Confidence Threshold</Label>
                    <Input type="number" placeholder="0.8" className="h-8 text-xs" />
                  </div>
                </>
              )}
              {selectedNode.type === "model" && (
                <>
                  <div>
                    <Label className="text-xs text-gray-600">Epochs</Label>
                    <Input type="number" placeholder="100" className="h-8 text-xs" />
                  </div>
                  <div>
                    <Label className="text-xs text-gray-600">Learning Rate</Label>
                    <Input type="number" placeholder="0.001" className="h-8 text-xs" />
                  </div>
                </>
              )}
              <Button size="sm" className="w-full">
                Update Configuration
              </Button>
            </div>
          ) : (
            <p className="text-xs text-gray-500">Select a node to configure</p>
          )}
        </CardContent>
      </Card>

      <Separator className="mx-4" />

      {/* Execution Status */}
      <Card className="m-4 mt-2 mb-2">
        <CardHeader className="pb-2">
          <CardTitle className="text-sm font-medium flex items-center">
            <Activity className="w-4 h-4 mr-2" />
            Execution Status
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="flex items-center space-x-2">
            {isExecuting ? (
              <>
                <Clock className="w-4 h-4 text-yellow-500 animate-spin" />
                <Badge variant="secondary" className="bg-yellow-100 text-yellow-800">
                  Running
                </Badge>
              </>
            ) : logs.length > 0 ? (
              <>
                <CheckCircle className="w-4 h-4 text-green-500" />
                <Badge variant="secondary" className="bg-green-100 text-green-800">
                  Completed
                </Badge>
              </>
            ) : (
              <>
                <XCircle className="w-4 h-4 text-gray-400" />
                <Badge variant="secondary" className="bg-gray-100 text-gray-600">
                  Ready
                </Badge>
              </>
            )}
          </div>
        </CardContent>
      </Card>

      <Separator className="mx-4" />

      {/* Execution Logs */}
      <Card className="m-4 mt-2 flex-1 flex flex-col">
        <CardHeader className="pb-2">
          <CardTitle className="text-sm font-medium flex items-center">
            <Terminal className="w-4 h-4 mr-2" />
            Execution Logs
          </CardTitle>
        </CardHeader>
        <CardContent className="flex-1 flex flex-col">
          <ScrollArea className="flex-1 h-0">
            <div className="space-y-1">
              {logs.length === 0 ? (
                <p className="text-xs text-gray-500">No logs yet. Run the pipeline to see execution details.</p>
              ) : (
                logs.map((log, index) => (
                  <div key={index} className="text-xs font-mono text-gray-700 p-1 rounded bg-gray-50">
                    {log}
                  </div>
                ))
              )}
            </div>
          </ScrollArea>
        </CardContent>
      </Card>
    </div>
  )
}
