export type AllowedNodeType =
  | 'n8n-nodes-base.httpRequest'
  | 'n8n-nodes-base.set'
  | 'n8n-nodes-base.if'
  | 'n8n-nodes-base.function'
  | 'n8n-nodes-base.googleSheets'
  | 'n8n-nodes-base.openAi'
  | 'n8n-nodes-base.webhook';

export interface NodePosition {
  x: number;
  y: number;
}

export interface NodeCredentials {
  [key: string]: `{{CREDENTIALS.${string}}}`;
}

export interface WorkflowNode {
  id: string;
  name: string;
  type: AllowedNodeType;
  position: NodePosition;
  parameters: Record<string, unknown>;
  credentials?: NodeCredentials;
  notes?: string;
}

export interface ConnectionReference {
  node: string;
  type: 'main';
  index: number;
}

export interface ConnectionMap {
  main: ConnectionReference[][];
}

export interface Workflow {
  nodes: WorkflowNode[];
  connections: Record<string, ConnectionMap>;
  active: boolean;
  settings: Record<string, never>;
  meta: Record<string, unknown>;
  version: number;
}

export interface GenerateOptions {
  spec: string;
  outputPath?: string;
  model?: string;
  temperature?: number;
}
