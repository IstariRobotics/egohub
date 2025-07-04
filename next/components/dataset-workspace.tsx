"use client"

import { useState } from "react"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Database, Workflow, BarChart3, Eye, Settings, Play, RefreshCw, FileText, Clock, HardDrive } from "lucide-react"

import { PipelineDesigner } from "./pipeline-designer"
import { DatasetOverview } from "./dataset-overview"
import { RerunViewer } from "./rerun-viewer"
import { DatasetAnalytics } from "./dataset-analytics"

// Mock dataset data
const datasets = [
  {
    id: "egodex-v1",
    name: "EgoDex v1.0",
    status: "processed",
    size: "2.3 GB",
    sequences: 1247,
    lastModified: "2024-01-15",
    description: "Egocentric dexterous manipulation dataset with hand poses and object interactions",
  },
  {
    id: "aria-pilot",
    name: "Aria Pilot Study",
    status: "raw",
    size: "890 MB",
    sequences: 342,
    lastModified: "2024-01-10",
    description: "Pilot data collection using Aria glasses for egocentric vision",
  },
  {
    id: "kitchen-tasks",
    name: "Kitchen Tasks",
    status: "processing",
    size: "1.8 GB",
    sequences: 856,
    lastModified: "2024-01-12",
    description: "Kitchen manipulation tasks with multi-modal sensor data",
  },
]

export function DatasetWorkspace() {
  const [selectedDataset, setSelectedDataset] = useState(datasets[0])

  const getStatusColor = (status: string) => {
    switch (status) {
      case "processed":
        return "bg-green-100 text-green-800"
      case "processing":
        return "bg-yellow-100 text-yellow-800"
      case "raw":
        return "bg-blue-100 text-blue-800"
      default:
        return "bg-gray-100 text-gray-800"
    }
  }

  return (
    <div className="flex h-screen bg-gray-50">
      {/* Sidebar */}
      <div className="w-80 bg-white border-r border-gray-200 flex flex-col">
        <div className="p-4 border-b border-gray-200">
          <h1 className="text-xl font-bold text-gray-900 mb-2">EgoHub Workspace</h1>
          <Select
            value={selectedDataset.id}
            onValueChange={(value) => {
              const dataset = datasets.find((d) => d.id === value)
              if (dataset) setSelectedDataset(dataset)
            }}
          >
            <SelectTrigger>
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              {datasets.map((dataset) => (
                <SelectItem key={dataset.id} value={dataset.id}>
                  <div className="flex items-center space-x-2">
                    <Database className="w-4 h-4" />
                    <span>{dataset.name}</span>
                  </div>
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>

        {/* Dataset Info */}
        <div className="p-4 border-b border-gray-200">
          <div className="space-y-3">
            <div className="flex items-center justify-between">
              <span className="text-sm font-medium text-gray-700">Status</span>
              <Badge className={getStatusColor(selectedDataset.status)}>{selectedDataset.status}</Badge>
            </div>
            <div className="flex items-center justify-between">
              <span className="text-sm font-medium text-gray-700">Size</span>
              <span className="text-sm text-gray-600">{selectedDataset.size}</span>
            </div>
            <div className="flex items-center justify-between">
              <span className="text-sm font-medium text-gray-700">Sequences</span>
              <span className="text-sm text-gray-600">{selectedDataset.sequences}</span>
            </div>
            <div className="flex items-center justify-between">
              <span className="text-sm font-medium text-gray-700">Modified</span>
              <span className="text-sm text-gray-600">{selectedDataset.lastModified}</span>
            </div>
          </div>
          <p className="text-xs text-gray-500 mt-3">{selectedDataset.description}</p>
        </div>

        {/* Quick Actions */}
        <div className="p-4 border-b border-gray-200">
          <h3 className="text-sm font-medium text-gray-700 mb-3">Quick Actions</h3>
          <div className="space-y-2">
            <Button variant="outline" size="sm" className="w-full justify-start bg-transparent">
              <Play className="w-4 h-4 mr-2" />
              Run Pipeline
            </Button>
            <Button variant="outline" size="sm" className="w-full justify-start bg-transparent">
              <RefreshCw className="w-4 h-4 mr-2" />
              Refresh Data
            </Button>
            <Button variant="outline" size="sm" className="w-full justify-start bg-transparent">
              <Eye className="w-4 h-4 mr-2" />
              Open in Rerun
            </Button>
          </div>
        </div>

        {/* Recent Activity */}
        <div className="p-4 flex-1">
          <h3 className="text-sm font-medium text-gray-700 mb-3">Recent Activity</h3>
          <div className="space-y-2">
            <div className="flex items-start space-x-2 text-xs">
              <Clock className="w-3 h-3 text-gray-400 mt-0.5" />
              <div>
                <p className="text-gray-600">Pipeline completed</p>
                <p className="text-gray-400">2 hours ago</p>
              </div>
            </div>
            <div className="flex items-start space-x-2 text-xs">
              <HardDrive className="w-3 h-3 text-gray-400 mt-0.5" />
              <div>
                <p className="text-gray-600">Data processed</p>
                <p className="text-gray-400">5 hours ago</p>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Main Content */}
      <div className="flex-1 flex flex-col">
        <div className="bg-white border-b border-gray-200 p-4">
          <div className="flex items-center justify-between">
            <div>
              <h2 className="text-lg font-semibold text-gray-900">{selectedDataset.name}</h2>
              <p className="text-sm text-gray-600">Dataset workspace and pipeline designer</p>
            </div>
            <div className="flex items-center space-x-2">
              <Button variant="outline" size="sm">
                <Settings className="w-4 h-4 mr-2" />
                Settings
              </Button>
            </div>
          </div>
        </div>

        <Tabs defaultValue="overview" className="flex-1 flex flex-col">
          <TabsList className="grid w-full grid-cols-4 bg-white border-b border-gray-200 rounded-none h-12">
            <TabsTrigger value="overview" className="flex items-center space-x-2">
              <FileText className="w-4 h-4" />
              <span>Overview</span>
            </TabsTrigger>
            <TabsTrigger value="pipeline" className="flex items-center space-x-2">
              <Workflow className="w-4 h-4" />
              <span>Pipeline</span>
            </TabsTrigger>
            <TabsTrigger value="analytics" className="flex items-center space-x-2">
              <BarChart3 className="w-4 h-4" />
              <span>Analytics</span>
            </TabsTrigger>
            <TabsTrigger value="viewer" className="flex items-center space-x-2">
              <Eye className="w-4 h-4" />
              <span>3D Viewer</span>
            </TabsTrigger>
          </TabsList>

          <TabsContent value="overview" className="flex-1 p-6">
            <DatasetOverview dataset={selectedDataset} />
          </TabsContent>

          <TabsContent value="pipeline" className="flex-1 p-0">
            <PipelineDesigner dataset={selectedDataset} />
          </TabsContent>

          <TabsContent value="analytics" className="flex-1 p-6">
            <DatasetAnalytics dataset={selectedDataset} />
          </TabsContent>

          <TabsContent value="viewer" className="flex-1 p-0">
            <RerunViewer dataset={selectedDataset} />
          </TabsContent>
        </Tabs>
      </div>
    </div>
  )
}
