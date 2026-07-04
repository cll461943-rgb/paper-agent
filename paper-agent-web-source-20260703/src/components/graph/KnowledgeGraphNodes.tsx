import { Handle, Position } from "@xyflow/react";
import type { NodeProps } from "@xyflow/react";
import type { CSSProperties } from "react";

export interface KnowledgeNodeData {
  [key: string]: unknown;
  title: string;
  subtitle?: string;
  meta?: string;
  badge?: string;
  tone: "paper" | "topic" | "method" | "dataset";
  highlighted?: boolean;
  linked?: boolean;
  muted?: boolean;
  selected?: boolean;
  accent?: string;
}

function HiddenHandles() {
  return (
    <>
      <Handle type="target" position={Position.Top} className="kg-handle" />
      <Handle type="source" position={Position.Bottom} className="kg-handle" />
      <Handle type="source" position={Position.Left} className="kg-handle" />
      <Handle type="target" position={Position.Right} className="kg-handle" />
    </>
  );
}

function nodeStyle(accent: string) {
  return { "--kg-accent": accent } as CSSProperties;
}

export function PaperNode({ data, selected }: NodeProps) {
  const content = data as unknown as KnowledgeNodeData;
  const isSelected = selected || content.selected;

  return (
    <div
      className={`kg-node kg-node-paper ${isSelected ? "selected" : ""} ${content.highlighted ? "highlighted" : ""} ${content.linked ? "linked" : ""} ${content.muted ? "muted" : ""}`}
      style={nodeStyle(content.accent ?? "#4f6df2")}
    >
      <HiddenHandles />
      <span className="kg-node-badge">{content.badge ?? "Paper"}</span>
      <strong>{content.title}</strong>
      {isSelected && content.subtitle ? <p>{content.subtitle}</p> : null}
    </div>
  );
}

export function TopicNode({ data, selected }: NodeProps) {
  const content = data as unknown as KnowledgeNodeData;
  const isSelected = selected || content.selected;

  return (
    <div
      className={`kg-node kg-node-topic ${isSelected ? "selected" : ""} ${content.linked ? "linked" : ""} ${content.muted ? "muted" : ""}`}
      style={nodeStyle(content.accent ?? "#3558d8")}
    >
      <HiddenHandles />
      <span className="kg-node-badge">{content.badge ?? "Topic"}</span>
      <strong>{content.title}</strong>
    </div>
  );
}

export function MetaNode({ data, selected }: NodeProps) {
  const content = data as unknown as KnowledgeNodeData;
  const isSelected = selected || content.selected;
  const fallbackAccent = content.tone === "dataset" ? "#1d8a68" : "#ad7024";

  return (
    <div
      className={`kg-node kg-node-meta kg-node-${content.tone} ${isSelected ? "selected" : ""} ${content.linked ? "linked" : ""} ${content.muted ? "muted" : ""}`}
      style={nodeStyle(content.accent ?? fallbackAccent)}
    >
      <HiddenHandles />
      <span className="kg-node-badge">{content.badge ?? content.tone}</span>
      <strong>{content.title}</strong>
    </div>
  );
}
