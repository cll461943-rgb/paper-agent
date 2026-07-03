import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useParams } from "react-router-dom";
import { ArrowUpRight, Radar } from "lucide-react";
import {
  applyNodeChanges,
  Background,
  Controls,
  MarkerType,
  Panel as FlowPanel,
  Position,
  ReactFlow,
} from "@xyflow/react";
import type { Edge, Node, NodeChange, NodeMouseHandler, XYPosition } from "@xyflow/react";
import { getGraph } from "../lib/api";
import type { GraphEdge, GraphNode, GraphResponse } from "../types/api";
import { PageHeader, Panel, StatusPill } from "../components/Common";
import { MetaNode, PaperNode, TopicNode } from "../components/graph/KnowledgeGraphNodes";
import type { KnowledgeNodeData } from "../components/graph/KnowledgeGraphNodes";
import { KnowledgeEdge } from "../components/graph/KnowledgeGraphEdges";
import type { KnowledgeEdgeData } from "../components/graph/KnowledgeGraphEdges";

const nodeTypes = {
  paper: PaperNode,
  topic: TopicNode,
  method: MetaNode,
  dataset: MetaNode,
};

const edgeTypes = {
  knowledge: KnowledgeEdge,
};

const clusterAccentPalette = ["#4f6df2", "#d0812f", "#1e8a62", "#2f7ab6", "#9d6c1e", "#b55074"];

const typeTone: Record<GraphNode["type"], "success" | "warning" | "info" | "muted"> = {
  paper: "info",
  method: "warning",
  dataset: "success",
  author: "muted",
  topic: "muted",
};

const typeLabel: Record<GraphNode["type"], string> = {
  paper: "Paper",
  method: "Method",
  dataset: "Dataset",
  author: "Author",
  topic: "Topic",
};

const relationLabel: Record<GraphEdge["type"], string> = {
  reference: "Reference path",
  similar: "Shared evidence",
  "citation-path": "Citation bridge",
  association: "Semantic link",
};

function truncate(value: string, length: number) {
  return value.length > length ? `${value.slice(0, length)}...` : value;
}

function formatDisplayLabel(value: string) {
  return value
    .replace(/\$([^$]+)\$/g, (_, inner: string) => inner.replace(/[\\{}]/g, "").replace(/\^/g, ""))
    .replace(/\s+/g, " ")
    .trim();
}

function edgeStroke(type: GraphEdge["type"]) {
  switch (type) {
    case "reference":
      return "#2f6fed";
    case "similar":
      return "#16835d";
    case "citation-path":
      return "#cf7a17";
    default:
      return "#74839a";
  }
}

function buildLayout(nodes: GraphNode[], clusters: GraphResponse["clusters"]) {
  const visible = nodes.filter((node) => ["paper", "topic", "method", "dataset"].includes(node.type));
  const papers = visible.filter((node) => node.type === "paper");
  const topics = visible.filter((node) => node.type === "topic");
  const methods = visible.filter((node) => node.type === "method");
  const datasets = visible.filter((node) => node.type === "dataset");
  const highlighted = papers.find((node) => node.highlighted) ?? papers[0];
  const map = new Map<string, { x: number; y: number }>();
  const centerX = 520;
  const centerY = 330;
  const clusterTopicById = new Map(
    clusters.map((cluster, index) => {
      const fallbackTopicId = `topic-${index + 1}`;
      return [cluster.id, visible.find((node) => node.id === fallbackTopicId)?.id ?? fallbackTopicId] as const;
    }),
  );

  if (topics.length) {
    const root = topics.find((node) => node.id === "topic-root") ?? topics[0];
    map.set(root.id, { x: centerX - 148, y: centerY - 4 });
    topics
      .filter((node) => node.id !== root.id)
      .forEach((node, index, rest) => {
        const count = Math.max(rest.length - 1, 1);
        const angle = -2.66 + (index / count) * 1.55;
        map.set(node.id, {
          x: centerX + Math.cos(angle) * 250,
          y: centerY + Math.sin(angle) * 152,
        });
      });
  }

  methods.forEach((node, index) => {
    const angle = 2.58 + index * 0.22;
    map.set(node.id, {
      x: centerX + Math.cos(angle) * 306,
      y: centerY + Math.sin(angle) * 164,
    });
  });

  datasets.forEach((node, index) => {
    const angle = 0.45 + index * 0.23;
    map.set(node.id, {
      x: centerX + Math.cos(angle) * 298,
      y: centerY + Math.sin(angle) * 164,
    });
  });

  if (highlighted) {
    map.set(highlighted.id, { x: centerX + 24, y: centerY + 18 });
  }

  const clusterOrder = clusters.map((cluster) => cluster.id);
  const fallbackPapers = papers.filter((paper) => paper.id !== highlighted?.id);

  clusterOrder.forEach((clusterId, clusterIndex) => {
    const group = fallbackPapers.filter((paper) => paper.clusterId === clusterId);
    const topicId = clusterTopicById.get(clusterId);
    const topicPoint = topicId ? map.get(topicId) : undefined;
    const baseAngle = topicPoint ? Math.atan2(topicPoint.y - centerY, topicPoint.x - centerX) : -2 + clusterIndex * 0.9;
    const anchorX = topicPoint ? topicPoint.x + Math.cos(baseAngle) * 126 : centerX + Math.cos(baseAngle) * 184;
    const anchorY = topicPoint ? topicPoint.y + Math.sin(baseAngle) * 74 : centerY + Math.sin(baseAngle) * 132;

    group.forEach((paper, paperIndex) => {
      const spread = group.length === 1 ? 0 : Math.min(0.92, 0.38 + group.length * 0.12);
      const localAngle = baseAngle + Math.PI / 2 - spread / 2 + (paperIndex / Math.max(group.length - 1, 1)) * spread;
      const localRadius = 24 + paperIndex * 16;
      map.set(paper.id, {
        x: anchorX + Math.cos(localAngle) * localRadius,
        y: anchorY + Math.sin(localAngle) * localRadius,
      });
    });
  });

  fallbackPapers
    .filter((paper) => !map.has(paper.id))
    .forEach((paper, index) => {
      const angle = -0.18 + index * 0.45;
      map.set(paper.id, {
        x: centerX + Math.cos(angle) * 132,
        y: centerY + Math.sin(angle) * 92,
      });
    });

  return map;
}

export function GraphPage() {
  const { jobId = "search_20260630_1042" } = useParams();
  const [graph, setGraph] = useState<GraphResponse | null>(null);
  const [selectedId, setSelectedId] = useState("");
  const [activeTypes, setActiveTypes] = useState<GraphNode["type"][]>(["paper", "topic", "method", "dataset"]);
  const [flowNodes, setFlowNodes] = useState<Node<KnowledgeNodeData>[]>([]);
  const draggedPositionsRef = useRef<Record<string, XYPosition>>({});

  useEffect(() => {
    void getGraph(jobId).then((nextGraph) => {
      setGraph(nextGraph);
      setSelectedId(nextGraph.selected_node_id ?? "");
    });
  }, [jobId]);

  const selectedNode = useMemo(() => graph?.nodes.find((node) => node.id === selectedId), [graph, selectedId]);
  const visibleNodes = useMemo(() => graph?.nodes.filter((node) => activeTypes.includes(node.type)) ?? [], [activeTypes, graph]);
  const visibleIds = useMemo(() => new Set(visibleNodes.map((node) => node.id)), [visibleNodes]);
  const visibleEdges = useMemo(
    () => graph?.edges.filter((edge) => visibleIds.has(edge.source) && visibleIds.has(edge.target)) ?? [],
    [graph, visibleIds],
  );

  const typeCounts = useMemo(
    () =>
      visibleNodes.reduce<Record<string, number>>((counts, node) => {
        counts[node.type] = (counts[node.type] ?? 0) + 1;
        return counts;
      }, {}),
    [visibleNodes],
  );

  const clusterPalette = useMemo(
    () =>
      new Map((graph?.clusters ?? []).map((cluster, index) => [cluster.id, clusterAccentPalette[index % clusterAccentPalette.length]])),
    [graph],
  );

  const connectedIds = useMemo(() => {
    const ids = new Set<string>();
    if (!selectedId) {
      return ids;
    }
    ids.add(selectedId);
    visibleEdges.forEach((edge) => {
      if (edge.source === selectedId) {
        ids.add(edge.target);
      }
      if (edge.target === selectedId) {
        ids.add(edge.source);
      }
    });
    return ids;
  }, [selectedId, visibleEdges]);

  const computedFlowNodes = useMemo<Node<KnowledgeNodeData>[]>(() => {
    if (!graph) {
      return [];
    }

    const layout = buildLayout(visibleNodes, graph.clusters);

    return visibleNodes.map((node) => {
      const position = layout.get(node.id) ?? { x: node.x, y: node.y };
      const meta = (node.meta ?? {}) as Record<string, unknown>;
      const subtitleBits = [node.year ? String(node.year) : "", typeof meta.venue === "string" ? meta.venue : ""].filter(Boolean);
      const cleanTitle = formatDisplayLabel(node.label);
      const accent =
        node.type === "topic" && node.id === "topic-root"
          ? "#3558d8"
          : node.type === "method"
            ? "#ad7024"
            : node.type === "dataset"
              ? "#1d8a68"
              : clusterPalette.get(node.clusterId ?? "") ?? "#4f6df2";
      const hasSelection = Boolean(selectedId);
      const isConnected = connectedIds.has(node.id);
      const isFocused = node.id === selectedId || node.highlighted;

      return {
        id: node.id,
        type: node.type === "topic" ? "topic" : node.type === "paper" ? "paper" : node.type,
        position: draggedPositionsRef.current[node.id] ?? position,
        draggable: true,
        selectable: true,
        data: {
          title:
            node.type === "paper"
              ? truncate(cleanTitle, isFocused ? 54 : 22)
              : node.type === "topic"
                ? truncate(cleanTitle, 22)
                : truncate(cleanTitle, 16),
          subtitle: isFocused && subtitleBits.length ? subtitleBits.join(" / ") : undefined,
          meta: undefined,
          badge: typeLabel[node.type],
          tone: node.type === "paper" || node.type === "topic" || node.type === "method" || node.type === "dataset" ? node.type : "topic",
          highlighted: node.highlighted,
          linked: hasSelection && isConnected && node.id !== selectedId,
          muted: hasSelection && !isConnected,
          accent,
        },
        sourcePosition: node.type === "method" ? Position.Right : node.type === "dataset" ? Position.Left : Position.Bottom,
        targetPosition: node.type === "method" ? Position.Right : node.type === "dataset" ? Position.Left : Position.Top,
      } satisfies Node<KnowledgeNodeData>;
    });
  }, [clusterPalette, connectedIds, graph, selectedId, visibleNodes]);

  useEffect(() => {
    setFlowNodes(computedFlowNodes);
  }, [computedFlowNodes]);

  const flowEdges = useMemo<Edge[]>(() => {
    return visibleEdges.map((edge) => ({
      id: edge.id,
      source: edge.source,
      target: edge.target,
      type: "knowledge",
      animated: false,
      zIndex: edge.source === selectedId || edge.target === selectedId ? 10 : 1,
      markerEnd: {
        type: MarkerType.ArrowClosed,
        width: 12,
        height: 12,
        color: edgeStroke(edge.type),
      },
      style: {
        stroke: edgeStroke(edge.type),
        strokeWidth:
          edge.source === selectedId || edge.target === selectedId
            ? edge.type === "reference"
              ? 2.4
              : 2.05
            : edge.type === "reference"
              ? 1.75
              : 1.24,
        strokeDasharray: edge.type === "citation-path" ? "8 8" : edge.type === "similar" ? "3 7" : edge.type === "association" ? "0" : undefined,
        opacity: edge.source === selectedId || edge.target === selectedId ? 0.95 : selectedId ? 0.1 : edge.type === "reference" ? 0.46 : 0.26,
      },
      interactionWidth: 28,
      data: {
        tone: edge.type,
        active: edge.source === selectedId || edge.target === selectedId,
      } satisfies KnowledgeEdgeData,
    }));
  }, [selectedId, visibleEdges]);

  const relationRows = useMemo(
    () =>
      visibleEdges
        .filter((edge) => edge.source === selectedId || edge.target === selectedId)
        .map((edge) => ({
          edge,
          node: graph?.nodes.find((node) => node.id === (edge.source === selectedId ? edge.target : edge.source)),
        }))
        .filter((row) => row.node),
    [graph, selectedId, visibleEdges],
  );

  const onNodeClick = useMemo<NodeMouseHandler<Node<KnowledgeNodeData>>>(
    () => (_, node) => {
      setSelectedId(node.id);
    },
    [],
  );

  const onNodesChange = useCallback((changes: NodeChange<Node<KnowledgeNodeData>>[]) => {
    setFlowNodes((current) => {
      const next = applyNodeChanges(changes, current);
      for (const node of next) {
        draggedPositionsRef.current[node.id] = node.position;
      }
      return next;
    });
  }, []);

  if (!graph) {
    return <PageHeader eyebrow="Knowledge Graph" title="Loading graph" description="Rebuilding semantic neighborhoods and evidence paths." />;
  }

  const meta = (selectedNode?.meta ?? {}) as Record<string, unknown>;

  function toggleType(type: GraphNode["type"]) {
    setActiveTypes((current) => (current.includes(type) ? current.filter((item) => item !== type) : [...current, type]));
  }

  return (
    <>
      <PageHeader eyebrow="Knowledge Graph" title="Knowledge graph" description={graph.query} />

      <div className="graph-grid graph-grid-refined graph-grid-simple">
        <Panel title="Network graph" meta={`${flowNodes.length} nodes / ${flowEdges.length} links`} className="graph-panel graph-panel-refined kg-panel">
          <div className="kg-flow-wrap kg-flow-wrap-simple">
            <ReactFlow
              nodes={flowNodes}
              edges={flowEdges}
              nodeTypes={nodeTypes}
              edgeTypes={edgeTypes}
              fitView
              fitViewOptions={{ padding: 0.1 }}
              minZoom={0.56}
              maxZoom={1.6}
              onNodeClick={onNodeClick}
              onNodesChange={onNodesChange}
              onPaneClick={() => setSelectedId("")}
              nodesDraggable
              nodesConnectable={false}
              elementsSelectable
              onlyRenderVisibleElements
              proOptions={{ hideAttribution: true }}
            >
              <Background gap={24} size={1} color="#dce5f0" />
              <Controls showInteractive={false} />
              <FlowPanel position="bottom-right" className="kg-layer-panel kg-layer-panel-simple kg-layer-panel-quiet">
                <div className="kg-overlay-head">
                  <Radar size={15} />
                  <strong>Layers</strong>
                </div>
                <div className="kg-layer-buttons">
                  {(["paper", "topic", "method", "dataset"] as GraphNode["type"][]).map((type) => (
                    <button className={activeTypes.includes(type) ? "active" : ""} key={type} type="button" onClick={() => toggleType(type)}>
                      <span>{typeLabel[type]}</span>
                      <small>{typeCounts[type] ?? 0}</small>
                    </button>
                  ))}
                </div>
              </FlowPanel>
            </ReactFlow>
          </div>
        </Panel>

        <Panel
          title="Evidence focus"
          meta={selectedNode ? `${typeLabel[selectedNode.type]}${selectedNode.year ? ` / ${selectedNode.year}` : ""}` : "Select a node"}
          className="graph-detail-panel"
        >
          {selectedNode ? (
            <div className="node-detail node-detail-refined">
              <div className="node-detail-top">
                <StatusPill tone={typeTone[selectedNode.type]} label={typeLabel[selectedNode.type]} />
                {selectedNode.year ? <span>{selectedNode.year}</span> : null}
              </div>

              <h2>{formatDisplayLabel(selectedNode.label)}</h2>

              <div className="graph-detail-meta graph-detail-meta-simple">
                <div>
                  <span>Cluster</span>
                  <strong>{graph.clusters.find((cluster) => cluster.id === selectedNode.clusterId)?.label ?? selectedNode.clusterId ?? "Core layer"}</strong>
                </div>
                <div>
                  <span>Connections</span>
                  <strong>{relationRows.length}</strong>
                </div>
              </div>

              {typeof meta.authors === "string" && meta.authors ? (
                <div className="graph-detail-block">
                  <span>Authors</span>
                  <p>{String(meta.authors)}</p>
                </div>
              ) : null}

              {meta.venue ? (
                <div className="graph-detail-block">
                  <span>Venue / Source</span>
                  <p>{String(meta.venue)}</p>
                </div>
              ) : null}

              {meta.summary ? (
                <div className="graph-detail-block">
                  <span>Theme summary</span>
                  <p>{String(meta.summary)}</p>
                </div>
              ) : null}

              {meta.abstract ? (
                <div className="graph-detail-block">
                  <span>Abstract</span>
                  <p style={{ textAlign: "justify", lineHeight: "1.4", fontSize: "12px", color: "#475569", margin: "4px 0" }}>{String(meta.abstract)}</p>
                </div>
              ) : null}

              {meta.score !== undefined ? (
                <div className="graph-detail-block">
                  <span>Ranking signal</span>
                  <p>{Number(meta.score).toFixed(3)}</p>
                </div>
              ) : null}

              {selectedNode.type === "paper" && (
                <div style={{ display: "flex", gap: "8px", margin: "12px 0", flexWrap: "wrap" }}>
                  {meta.url && (
                    <a className="button primary small" href={String(meta.url)} target="_blank" rel="noreferrer" style={{ fontSize: "11px", padding: "4px 8px" }}>
                      Open Link
                    </a>
                  )}
                  {meta.doi && (
                    <a className="button secondary small" href={`https://doi.org/${String(meta.doi)}`} target="_blank" rel="noreferrer" style={{ fontSize: "11px", padding: "4px 8px" }}>
                      Open DOI
                    </a>
                  )}
                </div>
              )}

              {selectedNode.type === "paper" && meta.references && Array.isArray(meta.references) && meta.references.length > 0 ? (
                <div className="graph-detail-block">
                  <span>References ({meta.references.length})</span>
                  <ul className="scrollbar-thin" style={{ maxHeight: "110px", overflowY: "auto", paddingLeft: "16px", fontSize: "11px", color: "#64748b", display: "grid", gap: "4px", margin: "4px 0" }}>
                    {meta.references.slice(0, 10).map((ref: any, idx: number) => (
                      <li key={idx} style={{ overflowWrap: "anywhere" }}>{String(ref)}</li>
                    ))}
                    {meta.references.length > 10 ? <li style={{ listStyleType: "none", color: "#94a3b8" }}>...and {meta.references.length - 10} more</li> : null}
                  </ul>
                </div>
              ) : null}

              <div className="edge-list edge-list-refined">
                <strong>Connected nodes</strong>
                {relationRows.slice(0, 6).map(({ edge, node }) =>
                  node ? (
                    <button key={edge.id} type="button" onClick={() => setSelectedId(node.id)}>
                      <div>
                        <span>{relationLabel[edge.type]}</span>
                        <strong>{truncate(formatDisplayLabel(node.label), 40)}</strong>
                      </div>
                      <StatusPill tone={typeTone[node.type]} label={typeLabel[node.type]} />
                    </button>
                  ) : null,
                )}
                {!relationRows.length ? <p className="graph-empty-note">This node currently has no visible links under the active layers.</p> : null}
              </div>
            </div>
          ) : (
            <div className="graph-detail-empty">
              <p>Click a node to inspect its neighborhood.</p>
            </div>
          )}
        </Panel>
      </div>
    </>
  );
}
