"use client"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Slider } from "@/components/ui/slider"
import { Play, SkipBack, SkipForward, Eye, Settings, Maximize, Download } from "lucide-react"

interface RerunViewerProps {
  dataset: any
}

export function RerunViewer({ dataset }: RerunViewerProps) {
  return (
    <div className="h-full flex flex-col">
      {/* Viewer Controls */}
      <div className="bg-white border-b border-gray-200 p-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-4">
            <div className="flex items-center space-x-2">
              <Button variant="outline" size="sm">
                <SkipBack className="w-4 h-4" />
              </Button>
              <Button variant="outline" size="sm">
                <Play className="w-4 h-4" />
              </Button>
              <Button variant="outline" size="sm">
                <SkipForward className="w-4 h-4" />
              </Button>
            </div>

            <div className="flex items-center space-x-2">
              <span className="text-sm text-gray-600">Sequence:</span>
              <Select defaultValue="seq-001">
                <SelectTrigger className="w-32">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="seq-001">Sequence 001</SelectItem>
                  <SelectItem value="seq-002">Sequence 002</SelectItem>
                  <SelectItem value="seq-003">Sequence 003</SelectItem>
                </SelectContent>
              </Select>
            </div>

            <div className="flex items-center space-x-2">
              <span className="text-sm text-gray-600">Frame:</span>
              <span className="text-sm font-mono">1247 / 2450</span>
            </div>
          </div>

          <div className="flex items-center space-x-2">
            <Badge variant="secondary" className="bg-green-100 text-green-800">
              Connected
            </Badge>
            <Button variant="outline" size="sm">
              <Settings className="w-4 h-4" />
            </Button>
            <Button variant="outline" size="sm">
              <Maximize className="w-4 h-4" />
            </Button>
          </div>
        </div>

        {/* Timeline Scrubber */}
        <div className="mt-4">
          <Slider defaultValue={[50]} max={100} step={1} className="w-full" />
        </div>
      </div>

      {/* Main Viewer */}
      <div className="flex-1 flex">
        {/* 3D Viewer */}
        <div className="flex-1 bg-gray-900 relative">
          <div className="absolute inset-0 flex items-center justify-center">
            <div className="text-center text-white">
              <Eye className="w-16 h-16 mx-auto mb-4 opacity-50" />
              <h3 className="text-lg font-medium mb-2">Rerun 3D Viewer</h3>
              <p className="text-sm opacity-75 mb-4">Interactive 3D visualization of {dataset.name}</p>
              <div className="space-y-2 text-xs opacity-60">
                <p>• Hand poses and trajectories</p>
                <p>• Object interactions</p>
                <p>• Camera viewpoints</p>
                <p>• Coordinate frames</p>
              </div>
            </div>
          </div>

          {/* Overlay Controls */}
          <div className="absolute top-4 left-4 space-y-2">
            <Badge className="bg-black/50 text-white">RGB Camera</Badge>
            <Badge className="bg-black/50 text-white">Hand Poses</Badge>
            <Badge className="bg-black/50 text-white">Objects</Badge>
          </div>
        </div>

        {/* Side Panel */}
        <div className="w-80 bg-white border-l border-gray-200 flex flex-col">
          <div className="p-4 border-b border-gray-200">
            <h3 className="text-sm font-medium text-gray-900 mb-3">Visualization Layers</h3>
            <div className="space-y-2">
              {[
                { name: "RGB Camera", enabled: true, color: "blue" },
                { name: "Hand Poses", enabled: true, color: "green" },
                { name: "Object Meshes", enabled: false, color: "purple" },
                { name: "Depth Maps", enabled: false, color: "orange" },
                { name: "Trajectories", enabled: true, color: "red" },
              ].map((layer) => (
                <div key={layer.name} className="flex items-center justify-between">
                  <div className="flex items-center space-x-2">
                    <div className={`w-3 h-3 rounded-full bg-${layer.color}-500`} />
                    <span className="text-sm">{layer.name}</span>
                  </div>
                  <input type="checkbox" checked={layer.enabled} className="rounded" />
                </div>
              ))}
            </div>
          </div>

          <div className="p-4 border-b border-gray-200">
            <h3 className="text-sm font-medium text-gray-900 mb-3">Current Frame Data</h3>
            <div className="space-y-2 text-xs">
              <div className="flex justify-between">
                <span className="text-gray-600">Timestamp:</span>
                <span className="font-mono">12.47s</span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-600">Hand confidence:</span>
                <span className="font-mono">0.94</span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-600">Objects detected:</span>
                <span className="font-mono">3</span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-600">Camera pose:</span>
                <span className="font-mono">Valid</span>
              </div>
            </div>
          </div>

          <div className="p-4 flex-1">
            <h3 className="text-sm font-medium text-gray-900 mb-3">Export Options</h3>
            <div className="space-y-2">
              <Button variant="outline" size="sm" className="w-full justify-start bg-transparent">
                <Download className="w-4 h-4 mr-2" />
                Export Frame
              </Button>
              <Button variant="outline" size="sm" className="w-full justify-start bg-transparent">
                <Download className="w-4 h-4 mr-2" />
                Export Sequence
              </Button>
              <Button variant="outline" size="sm" className="w-full justify-start bg-transparent">
                <Eye className="w-4 h-4 mr-2" />
                Open in Rerun
              </Button>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
