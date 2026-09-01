export type ViewName = "overview" | "analysis" | "tasks" | "manage";

export interface WorkerHealth {
  online: boolean;
  status: string;
  worker_id?: string;
  heartbeat_at?: string;
  current_task_id?: number | null;
}

export interface HealthPayload {
  name: string;
  version: string;
  status: string;
  worker: WorkerHealth;
}

export interface TaskItem {
  id: string;
  displayId: string;
  course: string;
  video: string;
  status: string;
  progress: number;
  mode: string;
  createdAt: string;
  startedAt: string;
  endedAt: string;
  errorMessage: string;
  workerId: string;
  heartbeatAt: string;
  attemptCount: number;
}

export interface SegmentItem {
  label: string;
  start: number;
  total: number;
  classCount: number;
  dominantBehavior: string;
  dominantBehaviorKey: string;
  dominantBehaviorRate: number;
  reviewCueCount: number;
  reviewCueRate: number;
  reviewPriority: string;
  reviewReason: string;
  requiresReview: boolean;
  distribution: Record<string, number>;
  counts: Record<string, number>;
}

export interface DimensionItem {
  name: string;
  score: number;
  evidence: string;
}

export interface AgentItem {
  name: string;
  focus: string;
  status: string;
  finding: string;
  evidence: string;
  color: string;
}

export interface ReportItem {
  id: number;
  course: string;
  teacher: string;
  className: string;
  classroom: string;
  lessonDate: string;
  lessonSection: string;
  videoName: string;
  duration: number;
  totalCount: number;
  classCount: number;
  dominantBehavior: string;
  dominantBehaviorKey: string;
  dominantBehaviorRate: number;
  reviewCueCount: number;
  reviewCueRate: number;
  reviewSegmentCount: number;
  evidenceCompleteness: number;
  evidenceStatus: string;
  reviewFocus: string;
  distribution: Record<string, number>;
  counts: Record<string, number>;
  dimensions: DimensionItem[];
  segments: SegmentItem[];
  warnings: string[];
  agents: AgentItem[];
  consensus: string;
  suggestion: string;
}

export interface DashboardPayload {
  schemaVersion: number;
  generatedAt: string;
  summary: {
    videoCount: number;
    taskCount: number;
    completedCount: number;
    reportCount: number;
    reviewSegmentCount: number;
  };
  reports: ReportItem[];
  tasks: TaskItem[];
  model: {
    name: string;
    family: string;
    weight: string;
    config: string;
    ready: boolean;
  };
}

export interface VideoItem {
  id: number;
  course_name: string;
  teacher_name: string;
  class_name: string;
  video_name: string;
  analysis_status: string;
  upload_time: string;
  size_mb?: number;
}

export interface FrameItem {
  frameId: number;
  timestamp: number;
  timeLabel: string;
  targetCount: number;
  classCount: number;
  reviewCueCount: number;
  reviewCueRate: number;
  averageConfidence: number;
  dominantBehavior: string;
  imageUrl: string;
  cleanImageUrl: string;
  behaviorCounts: Record<string, number>;
  detections: Array<{
    id: number;
    label: string;
    labelText: string;
    confidence: number;
    isReviewCue: boolean;
    box: { x1: number; y1: number; x2: number; y2: number };
  }>;
}

export interface FramePayload {
  taskId: number;
  availableFrameCount: number;
  highlightFrameId: number | null;
  selectionRule: string;
  resolution: { width: number; height: number };
  frames: FrameItem[];
}

export interface DiagnosisPayload {
  llm_diagnosis?: {
    enabled?: boolean;
    model?: string;
    content?: string;
    summary?: string;
    reason?: string;
    error?: string;
  };
  agent_report?: { full_report?: string; summary?: string; suggestion?: string };
  multi_agent?: { agents?: AgentItem[]; consensus?: string };
}

export interface OcclusionItem {
  id: number;
  occlusion_type: "S-S" | "S-O" | "";
  confidence: number;
  reason: string;
  upper_body_visibility: string;
  blocker: string;
}

export interface OcclusionReview {
  enabled: boolean;
  used_llm: boolean;
  model?: string;
  reason?: string;
  reviewed_count?: number;
  items: OcclusionItem[];
  summary: {
    occluded_count: number;
    ss_count: number;
    so_count: number;
    hg_count?: number;
  };
}

export interface TeacherReview {
  task_id: number;
  owner: string;
  due: string;
  actions: string;
  status: "待提交" | "已提交" | "复评中" | "已完成";
  review_conclusion: "尚未复核" | "与课堂任务一致" | "需要持续关注" | "证据不足，无法判断";
  context_notes: string;
  updated_at: string;
}

export interface ModelFileItem {
  name: string;
  path: string;
  family?: string;
  is_default: boolean;
  size_mb?: number;
  size_kb?: number;
}

export interface ModelInfo {
  model_name: string;
  model_path: string;
  config_path: string;
  repo_path: string;
  model_family: string;
  exists: boolean;
  config_exists: boolean;
  repo_exists: boolean;
  repo_engine_ready: boolean;
  runtime_available: boolean;
  inference_supported: boolean;
  runtime_message: string;
  device: string;
  deim_missing_dependencies: string[];
  available_models: ModelFileItem[];
  available_configs: ModelFileItem[];
}
