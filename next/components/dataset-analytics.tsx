"use client"

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { BarChart3, TrendingUp, Activity } from "lucide-react"

interface DatasetAnalyticsProps {
  dataset: any
}

export function DatasetAnalytics({ dataset }: DatasetAnalyticsProps) {
  return (
    <div className="space-y-6">
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {/* Data Quality Metrics */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center space-x-2">
              <BarChart3 className="w-5 h-5" />
              <span>Data Quality</span>
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="flex justify-between items-center">
              <span className="text-sm">Hand tracking accuracy</span>
              <Badge className="bg-green-100 text-green-800">94.2%</Badge>
            </div>
            <div className="flex justify-between items-center">
              <span className="text-sm">Frame completeness</span>
              <Badge className="bg-green-100 text-green-800">98.7%</Badge>
            </div>
            <div className="flex justify-between items-center">
              <span className="text-sm">Pose estimation confidence</span>
              <Badge className="bg-yellow-100 text-yellow-800">87.3%</Badge>
            </div>
            <div className="flex justify-between items-center">
              <span className="text-sm">Annotation coverage</span>
              <Badge className="bg-blue-100 text-blue-800">76.1%</Badge>
            </div>
          </CardContent>
        </Card>

        {/* Performance Metrics */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center space-x-2">
              <Activity className="w-5 h-5" />
              <span>Performance</span>
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="flex justify-between items-center">
              <span className="text-sm">Processing speed</span>
              <Badge className="bg-green-100 text-green-800">2.4 fps</Badge>
            </div>
            <div className="flex justify-between items-center">
              <span className="text-sm">Memory usage</span>
              <Badge className="bg-yellow-100 text-yellow-800">8.2 GB</Badge>
            </div>
            <div className="flex justify-between items-center">
              <span className="text-sm">GPU utilization</span>
              <Badge className="bg-green-100 text-green-800">87%</Badge>
            </div>
            <div className="flex justify-between items-center">
              <span className="text-sm">Pipeline efficiency</span>
              <Badge className="bg-green-100 text-green-800">91.5%</Badge>
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Data Distribution */}
      <Card>
        <CardHeader>
          <CardTitle>Data Distribution</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="h-64 flex items-center justify-center border-2 border-dashed border-gray-300 rounded-lg">
            <div className="text-center">
              <BarChart3 className="w-12 h-12 text-gray-400 mx-auto mb-2" />
              <p className="text-sm text-gray-600">Interactive charts would be rendered here</p>
              <p className="text-xs text-gray-500">Showing sequence length, hand pose distributions, etc.</p>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Model Performance */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center space-x-2">
            <TrendingUp className="w-5 h-5" />
            <span>Model Performance Trends</span>
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="h-48 flex items-center justify-center border-2 border-dashed border-gray-300 rounded-lg">
            <div className="text-center">
              <TrendingUp className="w-12 h-12 text-gray-400 mx-auto mb-2" />
              <p className="text-sm text-gray-600">Training loss curves and validation metrics</p>
              <p className="text-xs text-gray-500">VAE reconstruction loss, policy performance, etc.</p>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  )
}
