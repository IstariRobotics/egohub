"use client"

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Progress } from "@/components/ui/progress"
import { BarChart3, Clock, HardDrive, Users, Camera, Hand, Eye, FileText } from "lucide-react"

interface DatasetOverviewProps {
  dataset: any
}

export function DatasetOverview({ dataset }: DatasetOverviewProps) {
  return (
    <div className="space-y-6">
      {/* Dataset Stats */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Total Sequences</CardTitle>
            <BarChart3 className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{dataset.sequences}</div>
            <p className="text-xs text-muted-foreground">+12% from last month</p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Storage Used</CardTitle>
            <HardDrive className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{dataset.size}</div>
            <p className="text-xs text-muted-foreground">73% of allocated space</p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Processing Time</CardTitle>
            <Clock className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">2.4h</div>
            <p className="text-xs text-muted-foreground">Average per sequence</p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Contributors</CardTitle>
            <Users className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">8</div>
            <p className="text-xs text-muted-foreground">Active researchers</p>
          </CardContent>
        </Card>
      </div>

      {/* Data Streams */}
      <Card>
        <CardHeader>
          <CardTitle>Available Data Streams</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div className="space-y-3">
              <div className="flex items-center justify-between">
                <div className="flex items-center space-x-2">
                  <Camera className="w-4 h-4 text-blue-600" />
                  <span className="text-sm font-medium">RGB Camera</span>
                </div>
                <Badge variant="secondary" className="bg-green-100 text-green-800">
                  Available
                </Badge>
              </div>
              <Progress value={100} className="h-2" />

              <div className="flex items-center justify-between">
                <div className="flex items-center space-x-2">
                  <Hand className="w-4 h-4 text-green-600" />
                  <span className="text-sm font-medium">Hand Poses</span>
                </div>
                <Badge variant="secondary" className="bg-green-100 text-green-800">
                  Available
                </Badge>
              </div>
              <Progress value={87} className="h-2" />
            </div>

            <div className="space-y-3">
              <div className="flex items-center justify-between">
                <div className="flex items-center space-x-2">
                  <Eye className="w-4 h-4 text-purple-600" />
                  <span className="text-sm font-medium">Depth Maps</span>
                </div>
                <Badge variant="secondary" className="bg-yellow-100 text-yellow-800">
                  Processing
                </Badge>
              </div>
              <Progress value={45} className="h-2" />

              <div className="flex items-center justify-between">
                <div className="flex items-center space-x-2">
                  <FileText className="w-4 h-4 text-orange-600" />
                  <span className="text-sm font-medium">Annotations</span>
                </div>
                <Badge variant="secondary" className="bg-blue-100 text-blue-800">
                  Manual
                </Badge>
              </div>
              <Progress value={23} className="h-2" />
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Recent Processing */}
      <Card>
        <CardHeader>
          <CardTitle>Recent Processing Jobs</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="space-y-4">
            <div className="flex items-center justify-between p-3 border rounded-lg">
              <div>
                <p className="text-sm font-medium">Hand pose estimation</p>
                <p className="text-xs text-gray-600">MediaPipe model on sequences 1200-1247</p>
              </div>
              <Badge className="bg-green-100 text-green-800">Completed</Badge>
            </div>

            <div className="flex items-center justify-between p-3 border rounded-lg">
              <div>
                <p className="text-sm font-medium">Depth map generation</p>
                <p className="text-xs text-gray-600">MiDaS model on RGB frames</p>
              </div>
              <Badge className="bg-yellow-100 text-yellow-800">Running</Badge>
            </div>

            <div className="flex items-center justify-between p-3 border rounded-lg">
              <div>
                <p className="text-sm font-medium">VAE training</p>
                <p className="text-xs text-gray-600">Latent space encoding for policy learning</p>
              </div>
              <Badge className="bg-gray-100 text-gray-800">Queued</Badge>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  )
}
