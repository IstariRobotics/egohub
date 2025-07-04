import type { Node, Edge } from "reactflow"
import { DataSourceNode } from "./nodes/data-source-node"
import { TransformNode } from "./nodes/transform-node"
import { ModelNode } from "./nodes/model-node"
import { OutputNode } from "./nodes/output-node"

export const nodeTypes = {
  dataSource: DataSourceNode,
  transform: TransformNode,
  model: ModelNode,
  output: OutputNode,
}

export const initialNodes: Node[] = [
  {
    id: "data-1",
    type: "dataSource",
    position: { x: 50, y: 100 },
    data: { label: "EgoDex Dataset" },
  },
  {
    id: "transform-1",
    type: "transform",
    position: { x: 350, y: 100 },
    data: { label: "Coordinate Transform" },
  },
  {
    id: "transform-2",
    type: "transform",
    position: { x: 650, y: 100 },
    data: { label: "Hand Tracking" },
  },
  {
    id: "model-1",
    type: "model",
    position: { x: 950, y: 100 },
    data: { label: "Train VAE" },
  },
  {
    id: "model-2",
    type: "model",
    position: { x: 350, y: 300 },
    data: { label: "Encode Dataset" },
  },
  {
    id: "model-3",
    type: "model",
    position: { x: 650, y: 300 },
    data: { label: "Train Policy" },
  },
  {
    id: "output-1",
    type: "output",
    position: { x: 950, y: 300 },
    data: { label: "Export Model" },
  },
]

export const initialEdges: Edge[] = [
  { id: "e1-2", source: "data-1", target: "transform-1" },
  { id: "e2-3", source: "transform-1", target: "transform-2" },
  { id: "e3-4", source: "transform-2", target: "model-1" },
  { id: "e4-5", source: "model-1", target: "model-2" },
  { id: "e5-6", source: "model-2", target: "model-3" },
  { id: "e6-7", source: "model-3", target: "output-1" },
]
