import { BaseEdge, getBezierPath } from "@xyflow/react";
import type { EdgeProps } from "@xyflow/react";

export interface KnowledgeEdgeData {
  [key: string]: unknown;
  tone: "reference" | "similar" | "citation-path" | "association";
  active?: boolean;
}

export function KnowledgeEdge({
  id,
  sourceX,
  sourceY,
  targetX,
  targetY,
  sourcePosition,
  targetPosition,
  markerEnd,
  style,
  data,
}: EdgeProps) {
  const edgeData = (data ?? {}) as unknown as KnowledgeEdgeData;
  const [path] = getBezierPath({
    sourceX,
    sourceY,
    targetX,
    targetY,
    sourcePosition,
    targetPosition,
    curvature: edgeData.active ? 0.36 : 0.28,
  });

  return (
    <>
      <path
        d={path}
        className={`kg-edge kg-edge-underlay tone-${edgeData.tone} ${edgeData.active ? "active" : "idle"}`}
      />
      <BaseEdge
        id={id}
        path={path}
        markerEnd={markerEnd}
        style={style}
        className={`kg-edge kg-edge-main tone-${edgeData.tone} ${edgeData.active ? "active" : "idle"}`}
      />
    </>
  );
}
