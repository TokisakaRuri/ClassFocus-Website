import {
  Activity,
  AlertTriangle,
  ArrowRight,
  BarChart3,
  BrainCircuit,
  Check,
  CheckCircle2,
  ChevronDown,
  CircleDot,
  Clock3,
  ClipboardCheck,
  Cpu,
  Database,
  Eye,
  FileChartColumn,
  FileDown,
  Film,
  FileCog,
  Gauge,
  HardDrive,
  LayoutDashboard,
  LoaderCircle,
  Menu,
  MonitorCheck,
  Play,
  RefreshCw,
  Search,
  Save,
  ServerCog,
  Settings2,
  ShieldCheck,
  Sparkles,
  SquareStack,
  Trash2,
  UploadCloud,
  Users,
  Video,
  X,
  XCircle,
  Zap,
} from "lucide-react";
import { FormEvent, useCallback, useEffect, useId, useMemo, useRef, useState } from "react";
import type { CSSProperties } from "react";
import {
  bootstrapSession,
  cancelTask,
  createAnalysis,
  deleteTask,
  generateDiagnosis,
  getDashboard,
  getFrames,
  getHealth,
  getModelInfo,
  getTeacherReview,
  getVideos,
  reviewFrameOcclusion,
  saveTeacherReview,
  uploadModel,
  uploadModelConfig,
  uploadVideo,
} from "./api";
import type {
  DashboardPayload,
  DiagnosisPayload,
  FramePayload,
  HealthPayload,
  ModelInfo,
  OcclusionReview,
  ReportItem,
  TaskItem,
  TeacherReview,
  VideoItem,
  ViewName,
} from "./types";

const NAV_ITEMS: Array<{ id: ViewName; label: string; detail: string; icon: typeof LayoutDashboard }> = [
  { id: "overview", label: "课堂总览", detail: "运行与证据概况", icon: LayoutDashboard },
  { id: "analysis", label: "行为分析", detail: "时序与关键帧", icon: BarChart3 },
  { id: "tasks", label: "任务中心", detail: "队列、进度与历史", icon: SquareStack },
  { id: "manage", label: "数据管理", detail: "上传与模型配置", icon: Settings2 },
];

const BEHAVIORS = [
  { key: "listening", label: "听课", color: "#4c7dff" },
  { key: "writing", label: "书写", color: "#2abd86" },
  { key: "reading", label: "阅读", color: "#8b68e8" },
  { key: "using phone", label: "使用手机", color: "#f4a340" },
  { key: "bowing the head", label: "低头", color: "#e86d77" },
  { key: "sleeping", label: "睡觉", color: "#697386" },
];

const BEHAVIOR_CONTEXT = [
  ["听课", "可能对应视听参与，但面向教师不能证明已进行认知加工。"],
  ["书写", "可能是记笔记或练习，也可能与当前任务无关，需结合教学环节。"],
  ["阅读", "可能阅读课程材料或无关内容，需结合关键帧与任务说明。"],
  ["低头", "可能是书写、阅读、查看材料、疲劳或注意转移，不能单独定性。"],
  ["使用手机", "可能用于扫码答题或资料查询，也可能与学习无关，需教师复核。"],
  ["睡觉", "需核对持续性、遮挡、姿态混淆与误检，再决定是否关注。"],
] as const;

const DIAGNOSIS_TITLES = ["行为证据汇总", "情境一致性分析", "教学反思建议"] as const;

const STATUS_TEXT: Record<string, string> = {
  completed: "已完成",
  running: "分析中",
  waiting: "队列等待",
  failed: "失败",
  canceled: "已取消",
  canceling: "正在取消",
  uploaded: "已上传",
};

const VIEW_COPY: Record<ViewName, { eyebrow: string; title: string; description: string }> = {
  overview: { eyebrow: "CLASSROOM OPERATIONS", title: "课堂分析总览", description: "从任务运行到课堂证据，一处掌握当前状态。" },
  analysis: { eyebrow: "EVIDENCE REVIEW", title: "课堂行为分析", description: "聚焦可观察行为、时序变化与可追溯关键帧。" },
  tasks: { eyebrow: "INFERENCE QUEUE", title: "分析任务中心", description: "独立 Worker 持久执行，API 重启不会丢失等待任务。" },
  manage: { eyebrow: "LOCAL DATA", title: "数据与模型管理", description: "视频、检测参数和模型状态均保留在本机。" },
};

function formatDuration(seconds: number): string {
  if (!seconds) return "—";
  const minutes = Math.floor(seconds / 60);
  const rest = Math.round(seconds % 60);
  return `${minutes}分 ${rest}秒`;
}

function formatDateTime(value: string): string {
  if (!value) return "—";
  return value.replace("T", " ").slice(0, 19);
}

function behaviorColor(label: string): string {
  return BEHAVIORS.find((item) => item.key === label)?.color || "#4c7dff";
}

function splitDiagnosis(diagnosis: DiagnosisPayload | null, report: ReportItem) {
  const content = diagnosis?.llm_diagnosis?.enabled ? diagnosis.llm_diagnosis.content || "" : "";
  const positions = DIAGNOSIS_TITLES.map((title) => ({ title, index: content.indexOf(title) })).filter((item) => item.index >= 0);
  if (positions.length === DIAGNOSIS_TITLES.length) {
    return positions.map((item, index) => {
      const start = item.index + item.title.length;
      const end = positions[index + 1]?.index ?? content.length;
      return { title: item.title, body: content.slice(start, end).replace(/^[\s：:*#-]+|[\s*]+$/g, "") || "暂无生成内容" };
    });
  }
  const agents = diagnosis?.multi_agent?.agents || report.agents || [];
  return DIAGNOSIS_TITLES.map((title, index) => ({
    title,
    body: agents[index]?.finding || agents[index]?.evidence || (index === 2 ? report.suggestion : report.consensus) || "需教师结合课堂情境复核。",
  }));
}

function cx(...values: Array<string | false | null | undefined>): string {
  return values.filter(Boolean).join(" ");
}

type SelectChoice = {
  value: string;
  label: string;
};

function SelectMenu({ value, options, onChange, ariaLabel, className }: {
  value: string;
  options: SelectChoice[];
  onChange: (value: string) => void;
  ariaLabel: string;
  className?: string;
}) {
  const [open, setOpen] = useState(false);
  const selectedIndex = Math.max(0, options.findIndex((option) => option.value === value));
  const [activeIndex, setActiveIndex] = useState(selectedIndex);
  const rootRef = useRef<HTMLDivElement>(null);
  const listId = useId();
  const selected = options[selectedIndex];

  useEffect(() => setActiveIndex(selectedIndex), [selectedIndex]);

  useEffect(() => {
    if (!open) return;
    const closeOnOutsidePress = (event: PointerEvent) => {
      if (!rootRef.current?.contains(event.target as Node)) setOpen(false);
    };
    document.addEventListener("pointerdown", closeOnOutsidePress);
    return () => document.removeEventListener("pointerdown", closeOnOutsidePress);
  }, [open]);

  const commit = (index: number) => {
    const option = options[index];
    if (!option) return;
    onChange(option.value);
    setOpen(false);
  };

  const handleKeyDown = (event: React.KeyboardEvent<HTMLButtonElement>) => {
    if (!options.length) return;
    if (event.key === "ArrowDown" || event.key === "ArrowUp") {
      event.preventDefault();
      const direction = event.key === "ArrowDown" ? 1 : -1;
      setOpen(true);
      setActiveIndex((current) => (current + direction + options.length) % options.length);
      return;
    }
    if (event.key === "Home" || event.key === "End") {
      event.preventDefault();
      setOpen(true);
      setActiveIndex(event.key === "Home" ? 0 : options.length - 1);
      return;
    }
    if (event.key === "Enter" || event.key === " ") {
      event.preventDefault();
      if (open) commit(activeIndex);
      else setOpen(true);
      return;
    }
    if (event.key === "Escape" && open) {
      event.preventDefault();
      setOpen(false);
    }
  };

  return (
    <div className={cx("select-menu", open && "is-open", className)} ref={rootRef}>
      <button
        type="button"
        className="select-trigger"
        aria-label={ariaLabel}
        aria-haspopup="listbox"
        aria-expanded={open}
        aria-controls={listId}
        aria-activedescendant={open ? `${listId}-option-${activeIndex}` : undefined}
        onClick={() => setOpen((current) => {
          if (!current) setActiveIndex(selectedIndex);
          return !current;
        })}
        onKeyDown={handleKeyDown}
      >
        <span>{selected?.label || "请选择"}</span>
        <ChevronDown size={16} aria-hidden="true" />
      </button>
      {open && (
        <div className="select-popover" id={listId} role="listbox" aria-label={ariaLabel}>
          {options.map((option, index) => (
            <button
              type="button"
              id={`${listId}-option-${index}`}
              role="option"
              aria-selected={index === selectedIndex}
              className={cx(index === selectedIndex && "is-selected", index === activeIndex && "is-active")}
              key={option.value}
              onMouseEnter={() => setActiveIndex(index)}
              onMouseDown={(event) => event.preventDefault()}
              onClick={() => commit(index)}
            >
              <span>{option.label}</span>
              {index === selectedIndex && <Check size={15} aria-hidden="true" />}
            </button>
          ))}
        </div>
      )}
    </div>
  );
}

function BrandMark() {
  return (
    <span className="brand-mark" aria-hidden="true">
      <i />
      <i />
      <i />
      <i />
    </span>
  );
}

function StatusBadge({ status }: { status: string }) {
  return (
    <span className={cx("status-badge", `is-${status}`)}>
      <i />
      {STATUS_TEXT[status] || status || "未知"}
    </span>
  );
}

function EmptyState({ icon: Icon = FileChartColumn, title, detail }: { icon?: typeof FileChartColumn; title: string; detail: string }) {
  return (
    <div className="empty-state">
      <span><Icon size={24} /></span>
      <strong>{title}</strong>
      <p>{detail}</p>
    </div>
  );
}

function ReportPicker({ reports, selectedId, onChange }: { reports: ReportItem[]; selectedId: number | null; onChange: (id: number) => void }) {
  if (!reports.length) return null;
  return (
    <div className="report-picker">
      <span className="report-picker-icon"><Video size={17} /></span>
      <span className="report-picker-copy">
        <small>当前课堂</small>
        <SelectMenu
          className="report-select"
          ariaLabel="当前课堂"
          value={String(selectedId ?? reports[0].id)}
          onChange={(value) => onChange(Number(value))}
          options={reports.map((report) => ({
            value: String(report.id),
            label: `${report.course} · ${report.teacher || "未填教师"} · ${report.lessonDate || "未填日期"}`,
          }))}
        />
      </span>
    </div>
  );
}

function MetricCard({ icon: Icon, label, value, hint, accent = "blue" }: { icon: typeof Activity; label: string; value: string | number; hint: string; accent?: string }) {
  return (
    <article className={cx("metric-card", `accent-${accent}`)}>
      <div className="metric-top">
        <span className="metric-icon"><Icon size={19} /></span>
        <span className="metric-kicker">实时</span>
      </div>
      <strong>{value}</strong>
      <div>
        <b>{label}</b>
        <p>{hint}</p>
      </div>
    </article>
  );
}

function DistributionChart({ report }: { report: ReportItem }) {
  const maximum = Math.max(1, ...BEHAVIORS.map((item) => Number(report.distribution[item.key] || 0)));
  return (
    <div className="distribution-chart">
      {BEHAVIORS.map((item) => {
        const percentage = Number(report.distribution[item.key] || 0);
        return (
          <div className="distribution-row" key={item.key}>
            <div className="distribution-label"><i style={{ background: item.color }} />{item.label}</div>
            <div className="distribution-track">
              <i style={{ width: `${Math.max(2, (percentage / maximum) * 100)}%`, background: item.color }} />
            </div>
            <strong>{percentage.toFixed(1)}%</strong>
            <span>{report.counts[item.key] || 0} 条</span>
          </div>
        );
      })}
    </div>
  );
}

function TaskTable({ tasks, compact = false, onCancel, onDelete }: { tasks: TaskItem[]; compact?: boolean; onCancel?: (id: string) => void; onDelete?: (id: string) => Promise<boolean> }) {
  const [confirmDeleteId, setConfirmDeleteId] = useState<string | null>(null);
  const [deletingId, setDeletingId] = useState<string | null>(null);
  const visible = compact ? tasks.slice(0, 6) : tasks;
  if (!visible.length) return <EmptyState icon={SquareStack} title="尚无分析任务" detail="上传课堂视频并创建任务后，队列状态会出现在这里。" />;
  const remove = async (taskId: string) => {
    if (!onDelete) return;
    setDeletingId(taskId);
    try {
      const deleted = await onDelete(taskId);
      if (deleted) setConfirmDeleteId(null);
    } catch {
      // The page-level handler owns user-facing error reporting.
    } finally {
      setDeletingId(null);
    }
  };
  return (
    <div className="table-scroll">
      <table className="task-table">
        <thead>
          <tr><th>任务</th><th>课程 / 视频</th><th>状态</th><th>进度</th><th>模型</th>{!compact && <th>操作</th>}</tr>
        </thead>
        <tbody>
          {visible.map((task) => (
            <tr key={task.id}>
              <td><span className="task-number">#{task.displayId || task.id}</span><small>{task.createdAt || "—"}</small></td>
              <td><strong>{task.course}</strong><small>{task.video}</small>{task.errorMessage && <span className="task-error" title={task.errorMessage}>{task.errorMessage}</span>}</td>
              <td><StatusBadge status={task.status} /></td>
              <td>
                <div className="progress-cell"><span><i style={{ width: `${Math.max(0, Math.min(100, task.progress))}%` }} /></span><b>{Math.round(task.progress)}%</b></div>
              </td>
              <td><span className="model-pill">{task.mode || "PENDING"}</span><small className="task-runtime">尝试 {task.attemptCount || 0} 次{task.workerId ? ` · ${task.workerId}` : ""}</small></td>
              {!compact && (
                <td>
                  <div className="table-actions">
                    {onCancel && ["waiting", "running"].includes(task.status) && (
                      <button className="table-action danger" onClick={() => onCancel(task.id)}><X size={14} /> 取消</button>
                    )}
                    {onDelete && !["waiting", "running", "canceling"].includes(task.status) && (confirmDeleteId === task.id ? (
                      <>
                        <button className="table-action danger solid" disabled={deletingId === task.id} onClick={() => void remove(task.id)}>{deletingId === task.id ? <LoaderCircle className="spin" size={14} /> : <Trash2 size={14} />} 确认</button>
                        <button className="table-action" onClick={() => setConfirmDeleteId(null)}>返回</button>
                      </>
                    ) : <button className="table-action" onClick={() => setConfirmDeleteId(task.id)}><Trash2 size={14} /> 删除</button>)}
                    {!onCancel && !onDelete && <span className="muted-dash">—</span>}
                  </div>
                </td>
              )}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function OverviewPage({ dashboard, report, health, onNavigate, onCancel }: { dashboard: DashboardPayload; report: ReportItem | null; health: HealthPayload | null; onNavigate: (view: ViewName) => void; onCancel: (id: string) => void }) {
  const running = dashboard.tasks.filter((task) => ["waiting", "running"].includes(task.status)).length;
  return (
    <>
      <section className="metrics-grid">
        <MetricCard icon={Film} label="课堂视频" value={dashboard.summary.videoCount} hint="已进入本地资料库" accent="blue" />
        <MetricCard icon={Activity} label="分析任务" value={dashboard.summary.taskCount} hint={`${running} 个任务正在队列中`} accent="green" />
        <MetricCard icon={FileChartColumn} label="课堂报告" value={dashboard.summary.reportCount} hint={`${dashboard.summary.completedCount} 个任务已完成`} accent="violet" />
        <MetricCard icon={Eye} label="待复核片段" value={dashboard.summary.reviewSegmentCount} hint="建议结合课堂情境确认" accent="orange" />
      </section>

      <section className="overview-grid">
        <article className="panel evidence-panel">
          <div className="panel-heading">
            <div><span className="eyebrow">CURRENT EVIDENCE</span><h2>当前课堂证据</h2></div>
            {report && <button className="text-button" onClick={() => onNavigate("analysis")}>查看完整分析 <ArrowRight size={15} /></button>}
          </div>
          {report ? (
            <>
              <div className="course-banner">
                <div>
                  <span className="course-label">{report.lessonSection || "课堂记录"}</span>
                  <h3>{report.course}</h3>
                  <p>{[report.teacher, report.className, report.classroom, report.lessonDate].filter(Boolean).join(" · ")}</p>
                </div>
                <div className="score-ring" style={{ "--score": `${report.evidenceCompleteness * 3.6}deg` } as CSSProperties}>
                  <span><strong>{Math.round(report.evidenceCompleteness)}</strong><small>证据完整度</small></span>
                </div>
              </div>
              <div className="evidence-stats">
                <div><span><Database size={17} /> 行为证据</span><strong>{report.totalCount.toLocaleString()}</strong></div>
                <div><span><Gauge size={17} /> 主要行为</span><strong>{report.dominantBehavior}</strong></div>
                <div><span><Eye size={17} /> 复核线索</span><strong>{report.reviewCueCount}</strong></div>
                <div><span><Clock3 size={17} /> 视频时长</span><strong>{formatDuration(report.duration)}</strong></div>
              </div>
              <div className="mini-distribution">
                {BEHAVIORS.map((item) => (
                  <span key={item.key} title={`${item.label} ${Number(report.distribution[item.key] || 0).toFixed(1)}%`} style={{ width: `${Math.max(0, Number(report.distribution[item.key] || 0))}%`, background: item.color }} />
                ))}
              </div>
              <p className="panel-note"><ShieldCheck size={16} /> 系统只呈现可观察行为证据，不推断学生内在认知状态。</p>
            </>
          ) : <EmptyState title="等待第一份真实报告" detail="已完成的模型分析会在这里形成行为证据、时序片段和教师复核建议。" />}
        </article>

        <aside className="panel runtime-panel">
          <div className="panel-heading"><div><span className="eyebrow">LOCAL RUNTIME</span><h2>本地运行状态</h2></div></div>
          <div className={cx("runtime-hero", health?.worker.online ? "online" : "offline")}>
            <span><ServerCog size={25} /></span>
            <div><small>独立推理 Worker</small><strong>{health?.worker.online ? (health.worker.status === "busy" ? "正在执行任务" : "已就绪") : "未连接"}</strong></div>
            <i />
          </div>
          <div className="runtime-list">
            <div><span><MonitorCheck size={16} /> FastAPI 服务</span><b className="ok">在线</b></div>
            <div><span><Database size={16} /> SQLite / WAL</span><b className="ok">已启用</b></div>
            <div><span><BrainCircuit size={16} /> 当前模型</span><b>{dashboard.model.family}</b></div>
            <div><span><HardDrive size={16} /> 模型权重</span><b title={dashboard.model.weight}>{dashboard.model.weight || "未配置"}</b></div>
          </div>
          {!health?.worker.online && <div className="runtime-warning"><AlertTriangle size={16} /><span>启动独立 Worker 后，等待任务才会开始推理。</span></div>}
        </aside>
      </section>

      <section className="panel history-panel">
        <div className="panel-heading"><div><span className="eyebrow">CLASSROOM HISTORY</span><h2>历史课堂行为构成</h2></div><span className="section-total">最近 {Math.min(8, dashboard.reports.length)} 堂</span></div>
        {dashboard.reports.length ? <div className="history-list">{dashboard.reports.slice(0, 8).reverse().map((item) => (
          <article key={item.id}>
            <div><strong>{item.course}</strong><span>{item.lessonDate || "日期未填写"} · {item.teacher || "教师未填写"}</span></div>
            <div className="history-stack">{BEHAVIORS.map((behavior) => <i key={behavior.key} style={{ width: `${Math.max(0, Number(item.distribution[behavior.key] || 0))}%`, background: behavior.color }} title={`${behavior.label} ${Number(item.distribution[behavior.key] || 0).toFixed(1)}%`} />)}</div>
            <b>{item.dominantBehavior}</b>
          </article>
        ))}</div> : <EmptyState title="暂无历史课堂" detail="完成多个真实分析任务后，可在这里跨课堂比较六类行为构成。" />}
      </section>

      <section className="panel tasks-panel">
        <div className="panel-heading">
          <div><span className="eyebrow">RECENT QUEUE</span><h2>最近任务</h2></div>
          <button className="text-button" onClick={() => onNavigate("tasks")}>进入任务中心 <ArrowRight size={15} /></button>
        </div>
        <TaskTable tasks={dashboard.tasks} compact onCancel={onCancel} />
      </section>
    </>
  );
}

function AnalysisPage({ report, frames, framesLoading, diagnosis, diagnosing, teachingContext, setTeachingContext, onDiagnose, onReviewOcclusion, teacherReview, reviewSaving, onSaveReview }: {
  report: ReportItem | null;
  frames: FramePayload | null;
  framesLoading: boolean;
  diagnosis: DiagnosisPayload | null;
  diagnosing: boolean;
  teachingContext: string;
  setTeachingContext: (value: string) => void;
  onDiagnose: () => void;
  onReviewOcclusion: (frameId: number) => Promise<OcclusionReview>;
  teacherReview: TeacherReview | null;
  reviewSaving: boolean;
  onSaveReview: (value: Omit<TeacherReview, "task_id" | "updated_at">) => Promise<boolean>;
}) {
  const [tab, setTab] = useState<"distribution" | "timeline" | "frames">("distribution");
  const [selectedFrameId, setSelectedFrameId] = useState<number | null>(null);
  const [occlusionReviews, setOcclusionReviews] = useState<Record<number, OcclusionReview>>({});
  const [reviewingFrameId, setReviewingFrameId] = useState<number | null>(null);
  const [frameReviewError, setFrameReviewError] = useState("");
  const [cleanFrameFailed, setCleanFrameFailed] = useState(false);
  const [reviewDraft, setReviewDraft] = useState<TeacherReview | null>(teacherReview);
  const frameReviewRequestRef = useRef(0);

  useEffect(() => {
    frameReviewRequestRef.current += 1;
    setOcclusionReviews({});
    setFrameReviewError("");
    setSelectedFrameId(null);
    setReviewingFrameId(null);
  }, [report?.id]);

  useEffect(() => {
    if (!frames?.frames.length) return;
    setSelectedFrameId((current) => current && frames.frames.some((frame) => frame.frameId === current) ? current : frames.highlightFrameId ?? frames.frames[0].frameId);
  }, [frames]);

  useEffect(() => setReviewDraft(teacherReview), [teacherReview]);

  useEffect(() => setCleanFrameFailed(false), [selectedFrameId]);

  if (!report) return <section className="panel"><EmptyState title="暂无可分析课堂" detail="完成一个真实模型任务后即可查看行为分布、时间段和关键帧。" /></section>;

  const selectedFrame = frames?.frames.find((frame) => frame.frameId === selectedFrameId) || frames?.frames[0] || null;
  const selectedOcclusion = selectedFrame ? occlusionReviews[selectedFrame.frameId] : undefined;
  const occlusionById = new Map((selectedOcclusion?.items || []).map((item) => [item.id, item]));
  const diagnosisSections = splitDiagnosis(diagnosis, report);

  const performFrameReview = async () => {
    if (!selectedFrame) return;
    const reportId = report.id;
    const requestId = ++frameReviewRequestRef.current;
    setReviewingFrameId(selectedFrame.frameId);
    setFrameReviewError("");
    try {
      const result = await onReviewOcclusion(selectedFrame.frameId);
      if (requestId !== frameReviewRequestRef.current || report.id !== reportId) return;
      setOcclusionReviews((current) => ({ ...current, [selectedFrame.frameId]: result }));
      if (!result.used_llm) setFrameReviewError(result.reason || "本次视觉复核未完成，可稍后重试。");
    } catch (reason) {
      if (requestId === frameReviewRequestRef.current && !(reason instanceof DOMException && reason.name === "AbortError")) {
        setFrameReviewError(reason instanceof Error ? reason.message : "视觉复核未完成");
      }
    } finally {
      if (requestId === frameReviewRequestRef.current) setReviewingFrameId(null);
    }
  };

  const submitTeacherReview = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    if (!reviewDraft) return;
    const { task_id: _taskId, updated_at: _updatedAt, ...payload } = reviewDraft;
    try {
      await onSaveReview(payload);
    } catch {
      // The page-level handler owns user-facing error reporting.
    }
  };

  return (
    <>
      <section className="analysis-hero">
        <div>
          <span className="eyebrow">{report.lessonDate || "课堂证据"} · {report.lessonSection || "完整课堂"}</span>
          <h2>{report.course}</h2>
          <p>{[report.teacher, report.className, report.classroom, report.videoName].filter(Boolean).join(" · ")}</p>
        </div>
        <div className="analysis-hero-metrics">
          <div><small>行为证据</small><strong>{report.totalCount}</strong></div>
          <div><small>覆盖类别</small><strong>{report.classCount}<i>/ 6</i></strong></div>
          <div><small>待复核片段</small><strong>{report.reviewSegmentCount}</strong></div>
          <a className="primary-button compact" href={`/api/reports/${report.id}/download`}><FileDown size={16} /> 导出报告</a>
        </div>
      </section>

      {report.warnings.length > 0 && (
        <div className="notice-bar"><AlertTriangle size={18} /><div><strong>复核提示</strong><span>{report.warnings.join("；")}</span></div></div>
      )}

      <section className="panel analysis-panel">
        <div className="analysis-tabs" role="tablist" aria-label="分析视图">
          <button className={tab === "distribution" ? "active" : ""} onClick={() => setTab("distribution")}><BarChart3 size={16} /> 行为分布</button>
          <button className={tab === "timeline" ? "active" : ""} onClick={() => setTab("timeline")}><Activity size={16} /> 时序证据</button>
          <button className={tab === "frames" ? "active" : ""} onClick={() => setTab("frames")}><Film size={16} /> 关键帧</button>
        </div>

        {tab === "distribution" && (
          <>
            <div className="analysis-two-column">
              <div>
                <div className="section-heading"><div><span className="eyebrow">BEHAVIOR MIX</span><h3>六类外显行为</h3></div><span className="section-total">共 {report.totalCount} 条</span></div>
                <DistributionChart report={report} />
              </div>
              <div className="dimensions-card">
                <div className="section-heading"><div><span className="eyebrow">EVIDENCE QUALITY</span><h3>证据可用性</h3></div></div>
                {report.dimensions.map((dimension) => (
                  <div className="dimension-row" key={dimension.name}>
                    <div><strong>{dimension.name}</strong><span title={dimension.evidence}>{dimension.evidence}</span></div>
                    <b>{Math.round(dimension.score)}</b>
                    <i><span style={{ width: `${dimension.score}%` }} /></i>
                  </div>
                ))}
              </div>
            </div>
            <div className="context-guide">
              <div className="context-guide-head"><ShieldCheck size={18} /><div><strong>行为线索的情境解释边界</strong><span>检测结果不等同于认知状态或教学质量结论</span></div></div>
              <div>{BEHAVIOR_CONTEXT.map(([label, detail]) => <article key={label}><b>{label}</b><p>{detail}</p></article>)}</div>
            </div>
          </>
        )}

        {tab === "timeline" && (
          <div className="timeline-view">
            <div className="section-heading"><div><span className="eyebrow">SEGMENT REVIEW</span><h3>课堂时序证据</h3></div><span className="section-total">{report.segments.length} 个时间段</span></div>
            {report.segments.length ? report.segments.map((segment, index) => (
              <article className={cx("timeline-row", segment.requiresReview && "needs-review")} key={`${segment.label}-${index}`}>
                <div className="timeline-time"><strong>{segment.label || `片段 ${index + 1}`}</strong><span>{segment.total} 条证据</span></div>
                <div className="timeline-bars">
                  <div>
                    {BEHAVIORS.map((item) => (
                      <i key={item.key} style={{ width: `${Math.max(0, Number(segment.distribution[item.key] || 0))}%`, background: item.color }} title={`${item.label} ${Number(segment.distribution[item.key] || 0).toFixed(1)}%`} />
                    ))}
                  </div>
                  <span>主要行为：{segment.dominantBehavior} · {segment.dominantBehaviorRate.toFixed(1)}%</span>
                </div>
                <div className="timeline-review">
                  {segment.requiresReview ? <><AlertTriangle size={16} /><span><b>{segment.reviewPriority}优先级</b>{segment.reviewReason}</span></> : <><CheckCircle2 size={16} /><span><b>常规抽查</b>暂无规则触发项</span></>}
                </div>
              </article>
            )) : <EmptyState title="暂无时序数据" detail="当前报告没有可展示的分段统计。" />}
          </div>
        )}

        {tab === "frames" && (
          <div className="frames-view">
            <div className="section-heading"><div><span className="eyebrow">TRACEABLE FRAMES</span><h3>代表性关键帧</h3></div><span className="section-total">{frames?.availableFrameCount || 0} 帧可用</span></div>
            {framesLoading ? <div className="loading-inline"><LoaderCircle className="spin" /> 正在读取关键帧</div> : frames?.frames.length ? (
              <div className="frames-grid">
                {frames.frames.map((frame) => (
                  <button type="button" className={cx("frame-card", selectedFrame?.frameId === frame.frameId && "selected")} key={frame.frameId} onClick={() => setSelectedFrameId(frame.frameId)}>
                    <div className="frame-image"><img src={frame.imageUrl} alt={`${frame.timeLabel} 课堂行为检测关键帧`} loading="lazy" /><span>{frame.timeLabel}</span></div>
                    <div className="frame-copy">
                      <div><strong>{frame.dominantBehavior || "行为证据"}</strong><span>{frame.targetCount} 个目标 · {frame.classCount} 类</span></div>
                      <span className={frame.reviewCueCount ? "review-chip" : "plain-chip"}>{frame.reviewCueCount ? `${frame.reviewCueCount} 条复核线索` : "常规抽查"}</span>
                    </div>
                  </button>
                ))}
              </div>
            ) : <EmptyState icon={Film} title="暂无关键帧" detail="任务完成且检测到行为目标后，这里会展示可追溯画面。" />}
            {selectedFrame && (
              <div className="frame-review-detail">
                <div className="frame-review-main">
                  <div className="frame-review-toolbar">
                    <div><span className="eyebrow">VISUAL VERIFICATION</span><h3>{selectedFrame.timeLabel} 视觉复核</h3><p>遮挡只按人体上半身局部缺失判断，不以框重叠或低置信度替代。</p></div>
                    <button className="primary-button" onClick={() => void performFrameReview()} disabled={reviewingFrameId === selectedFrame.frameId}>{reviewingFrameId === selectedFrame.frameId ? <LoaderCircle className="spin" size={16} /> : <Eye size={16} />} {reviewingFrameId === selectedFrame.frameId ? "Qwen 正在复核" : "Qwen 视觉复核"}</button>
                  </div>
                  {frameReviewError && <div className="inline-error"><AlertTriangle size={16} />{frameReviewError}</div>}
                  {cleanFrameFailed && <div className="inline-error"><AlertTriangle size={16} />原始视频无法提取干净帧，当前展示历史标注帧；右侧目标明细仍可用于复核。</div>}
                  <div className="frame-review-stage" style={{ aspectRatio: `${frames?.resolution.width || 16} / ${frames?.resolution.height || 9}` }}>
                    <img src={cleanFrameFailed ? selectedFrame.imageUrl : selectedFrame.cleanImageUrl} onError={() => setCleanFrameFailed(true)} alt={`${selectedFrame.timeLabel} 课堂关键帧`} />
                    {!cleanFrameFailed && selectedFrame.detections.map((detection) => {
                      const width = frames?.resolution.width || 1;
                      const height = frames?.resolution.height || 1;
                      const reviewed = occlusionById.get(detection.id);
                      return <span className={cx("detection-box", reviewed?.occlusion_type && "occluded")} key={detection.id} title={`#${detection.id + 1} ${detection.labelText}`} style={{ left: `${detection.box.x1 / width * 100}%`, top: `${detection.box.y1 / height * 100}%`, width: `${(detection.box.x2 - detection.box.x1) / width * 100}%`, height: `${(detection.box.y2 - detection.box.y1) / height * 100}%`, borderColor: behaviorColor(detection.label) }}><b style={{ background: behaviorColor(detection.label) }}>#{detection.id + 1}{reviewed?.occlusion_type ? ` · ${reviewed.occlusion_type}` : ""}</b></span>;
                    })}
                  </div>
                  {selectedOcclusion?.used_llm && <p className="review-summary"><CheckCircle2 size={15} /> {selectedOcclusion.model || "Qwen"} 已复核 {selectedOcclusion.reviewed_count || selectedFrame.detections.length} 个目标：S-S {selectedOcclusion.summary.ss_count}，S-O {selectedOcclusion.summary.so_count}。</p>}
                </div>
                <aside className="target-review-list">
                  <div className="section-heading"><div><span className="eyebrow">TARGETS</span><h3>目标明细</h3></div><span className="section-total">{selectedFrame.detections.length} 个</span></div>
                  {selectedFrame.detections.map((detection) => {
                    const reviewed = occlusionById.get(detection.id);
                    return <article key={detection.id}><i style={{ background: behaviorColor(detection.label) }}>{detection.id + 1}</i><div><strong>{detection.labelText}<span>{detection.confidence.toFixed(1)}%</span></strong><p>{selectedOcclusion?.used_llm ? (reviewed?.reason || "上半身完整可见") : "等待视觉复核"}</p></div><b className={reviewed?.occlusion_type ? "has-occlusion" : ""}>{selectedOcclusion?.used_llm ? reviewed?.occlusion_type || "无遮挡" : "待复核"}</b></article>;
                  })}
                </aside>
              </div>
            )}
            {frames?.selectionRule && <p className="panel-note"><ShieldCheck size={16} /> {frames.selectionRule}</p>}
          </div>
        )}
      </section>

      <section className="diagnosis-grid">
        <article className="panel agent-panel">
          <div className="panel-heading"><div><span className="eyebrow">ASSISTED REVIEW</span><h2>辅助诊断</h2></div><Sparkles size={20} /></div>
          <p className="panel-intro">补充本节课的讲授、练习或讨论安排，大模型只会在可追溯证据范围内解释。</p>
          <textarea value={teachingContext} onChange={(event) => setTeachingContext(event.target.value)} maxLength={2000} placeholder="例如：前 20 分钟讲授，随后分组阅读并使用手机查询资料……" />
          <div className="agent-actions">
            <span>{teachingContext.length} / 2000</span>
            <button className="primary-button" onClick={onDiagnose} disabled={diagnosing}>{diagnosing ? <LoaderCircle className="spin" size={16} /> : <Sparkles size={16} />} {diagnosing ? "正在生成" : "生成情境化诊断"}</button>
          </div>
        </article>
        <article className="panel diagnosis-result">
          <div className="panel-heading"><div><span className="eyebrow">TEACHER REVIEW</span><h2>分析结论</h2></div></div>
          {diagnosis?.llm_diagnosis && !diagnosis.llm_diagnosis.enabled && <div className="inline-error"><AlertTriangle size={16} />{diagnosis.llm_diagnosis.reason || "Qwen 辅助分析未完成，已保留本地规则说明。"}</div>}
          <div className="diagnosis-sections">{diagnosisSections.map((section, index) => <article key={section.title}><span>分析 {String(index + 1).padStart(2, "0")}</span><strong>{section.title}</strong><p>{section.body}</p></article>)}</div>
          <span className="diagnosis-source"><CircleDot size={14} /> {diagnosis?.llm_diagnosis?.enabled ? `${diagnosis.llm_diagnosis.model || "大模型"} 辅助生成` : "本地规则生成"}</span>
        </article>
      </section>

      <section className="panel review-plan-panel">
        <div className="panel-heading"><div><span className="eyebrow">FOLLOW-UP REVIEW</span><h2>反馈改进与复评</h2></div>{teacherReview?.updated_at ? <span className="section-total">保存于 {formatDateTime(teacherReview.updated_at)}</span> : <ClipboardCheck size={20} />}</div>
        {reviewDraft ? <form onSubmit={(event) => void submitTeacherReview(event)}>
          <div className="review-form-grid top">
            <label><span>责任人</span><input required maxLength={80} value={reviewDraft.owner} onChange={(event) => setReviewDraft({ ...reviewDraft, owner: event.target.value })} /></label>
            <label><span>完成时间</span><input required maxLength={80} value={reviewDraft.due} onChange={(event) => setReviewDraft({ ...reviewDraft, due: event.target.value })} /></label>
            <div className="form-field"><span>改进状态</span><SelectMenu ariaLabel="改进状态" value={reviewDraft.status} onChange={(value) => setReviewDraft({ ...reviewDraft, status: value as TeacherReview["status"] })} options={["待提交", "已提交", "复评中", "已完成"].map((label) => ({ value: label, label }))} /></div>
            <div className="form-field"><span>教师复核结论</span><SelectMenu ariaLabel="教师复核结论" value={reviewDraft.review_conclusion} onChange={(value) => setReviewDraft({ ...reviewDraft, review_conclusion: value as TeacherReview["review_conclusion"] })} options={["尚未复核", "与课堂任务一致", "需要持续关注", "证据不足，无法判断"].map((label) => ({ value: label, label }))} /></div>
          </div>
          <div className="review-form-grid notes">
            <label><span>改进措施</span><textarea maxLength={4000} value={reviewDraft.actions} onChange={(event) => setReviewDraft({ ...reviewDraft, actions: event.target.value })} placeholder="记录具体行动、适用时段与复查依据" /></label>
            <label><span>情境记录</span><textarea maxLength={4000} value={reviewDraft.context_notes} onChange={(event) => setReviewDraft({ ...reviewDraft, context_notes: event.target.value })} placeholder="补充教师观察与课堂任务信息" /></label>
          </div>
          <div className="review-form-actions"><span>记录持久保存在本机 SQLite，不再随页面会话丢失。</span><button className="primary-button" type="submit" disabled={reviewSaving}>{reviewSaving ? <LoaderCircle className="spin" size={16} /> : <Save size={16} />}{reviewSaving ? "正在保存" : "保存改进记录"}</button></div>
        </form> : <div className="loading-inline"><LoaderCircle className="spin" size={18} /> 正在读取改进记录</div>}
      </section>
    </>
  );
}

function TasksPage({ dashboard, health, onCancel, onDelete }: { dashboard: DashboardPayload; health: HealthPayload | null; onCancel: (id: string) => void; onDelete: (id: string) => Promise<boolean> }) {
  const counts = useMemo(() => ({
    running: dashboard.tasks.filter((task) => task.status === "running").length,
    waiting: dashboard.tasks.filter((task) => task.status === "waiting").length,
    failed: dashboard.tasks.filter((task) => task.status === "failed").length,
  }), [dashboard.tasks]);
  return (
    <>
      <section className="queue-summary">
        <article className="queue-hero">
          <div className={cx("worker-orbit", health?.worker.online && "online")}><span><Zap size={25} /></span><i /><i /></div>
          <div><span className="eyebrow">DEDICATED WORKER</span><h2>{health?.worker.online ? "推理 Worker 已连接" : "等待 Worker 启动"}</h2><p>{health?.worker.online ? `当前状态：${health.worker.status === "busy" ? "正在处理任务" : "空闲待命"}` : "任务会安全保留在 SQLite 队列中，不会丢失。"}</p></div>
          <div className="queue-stat"><span>执行中</span><strong>{counts.running}</strong></div>
          <div className="queue-stat"><span>等待中</span><strong>{counts.waiting}</strong></div>
          <div className="queue-stat"><span>失败</span><strong>{counts.failed}</strong></div>
        </article>
      </section>
      <section className="panel queue-table-panel">
        <div className="panel-heading"><div><span className="eyebrow">PERSISTENT QUEUE</span><h2>全部分析任务</h2></div><span className="section-total">{dashboard.tasks.length} 项</span></div>
        <TaskTable tasks={dashboard.tasks} onCancel={onCancel} onDelete={onDelete} />
      </section>
      <section className="worker-explainer">
        <article><span><Database size={19} /></span><div><strong>持久队列</strong><p>任务状态与租约写入 SQLite，API 重启后等待任务仍然存在。</p></div></article>
        <article><span><BrainCircuit size={19} /></span><div><strong>模型常驻</strong><p>Worker 复用当前模型，减少重复加载和 GPU 显存波动。</p></div></article>
        <article><span><ShieldCheck size={19} /></span><div><strong>自动恢复</strong><p>Worker 超时后任务会回到等待队列，避免永久卡在运行中。</p></div></article>
      </section>
    </>
  );
}

function ManagePage({ videos, dashboard, modelInfo, uploading, uploadProgress, modelUploading, modelUploadProgress, creatingTask, onUpload, onCreateTask, onUploadModel }: {
  videos: VideoItem[];
  dashboard: DashboardPayload;
  modelInfo: ModelInfo | null;
  uploading: boolean;
  uploadProgress: number;
  modelUploading: "weight" | "config" | null;
  modelUploadProgress: number;
  creatingTask: boolean;
  onUpload: (form: HTMLFormElement) => void;
  onCreateTask: (videoId: number, confidence: number, sample: number, segment: number) => void;
  onUploadModel: (form: HTMLFormElement, kind: "weight" | "config") => void;
}) {
  const [selectedVideo, setSelectedVideo] = useState<number>(videos[0]?.id || 0);
  const [confidence, setConfidence] = useState(0.5);
  const [sample, setSample] = useState(1);
  const [segment, setSegment] = useState(60);

  useEffect(() => {
    if (!selectedVideo && videos[0]) setSelectedVideo(videos[0].id);
  }, [selectedVideo, videos]);

  const submitUpload = (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    onUpload(event.currentTarget);
  };

  return (
    <section className="manage-grid">
      <article className="panel upload-panel">
        <div className="panel-heading"><div><span className="eyebrow">NEW CLASSROOM</span><h2>上传课堂视频</h2></div><UploadCloud size={20} /></div>
        <form onSubmit={submitUpload}>
          <div className="form-grid three">
            <label><span>课程名称 *</span><input name="course_name" required placeholder="例如：数据结构" /></label>
            <label><span>教师姓名</span><input name="teacher_name" placeholder="教师姓名" /></label>
            <label><span>班级名称</span><input name="class_name" placeholder="例如：计科 2301" /></label>
            <label><span>教室</span><input name="classroom" placeholder="例如：A301" /></label>
            <label><span>上课日期</span><input name="lesson_date" type="date" /></label>
            <label><span>上课节次</span><input name="lesson_section" defaultValue="第1-2节" /></label>
          </div>
          <label className="file-drop">
            <input name="video_file" type="file" accept=".mp4,.avi,.mov,.mkv" required />
            <span><UploadCloud size={25} /></span>
            <strong>选择课堂视频</strong>
            <p>MP4、AVI、MOV 或 MKV，最大 2GB；文件只写入本机。</p>
          </label>
          {uploading && <div className="upload-progress"><div><span>正在写入本地资料库</span><b>{uploadProgress}%</b></div><i><span style={{ width: `${uploadProgress}%` }} /></i></div>}
          <button className="primary-button full" type="submit" disabled={uploading}>{uploading ? <LoaderCircle className="spin" size={17} /> : <UploadCloud size={17} />} {uploading ? "正在上传" : "上传视频"}</button>
        </form>
      </article>

      <article className="panel create-task-panel">
        <div className="panel-heading"><div><span className="eyebrow">INFERENCE SETTINGS</span><h2>创建分析任务</h2></div><Play size={20} /></div>
        {videos.length ? (
          <div className="task-form">
            <div className="form-field"><span>选择视频</span><SelectMenu ariaLabel="选择视频" value={String(selectedVideo)} onChange={(value) => setSelectedVideo(Number(value))} options={videos.map((video) => ({ value: String(video.id), label: `${video.course_name || "未命名课程"} · ${video.video_name}` }))} /></div>
            <div className="range-control"><div><span>置信度阈值</span><b>{confidence.toFixed(2)}</b></div><input type="range" min="0.1" max="1" step="0.05" value={confidence} onChange={(event) => setConfidence(Number(event.target.value))} /><p>数值越高，保留的检测结果越严格。</p></div>
            <div className="form-grid two">
              <div className="form-field"><span>抽帧间隔</span><SelectMenu ariaLabel="抽帧间隔" value={String(sample)} onChange={(value) => setSample(Number(value))} options={[{ value: "0.5", label: "0.5 秒" }, { value: "1", label: "1 秒" }, { value: "2", label: "2 秒" }, { value: "3", label: "3 秒" }]} /></div>
              <div className="form-field"><span>统计时间段</span><SelectMenu ariaLabel="统计时间段" value={String(segment)} onChange={(value) => setSegment(Number(value))} options={[{ value: "30", label: "30 秒" }, { value: "60", label: "60 秒" }, { value: "120", label: "2 分钟" }, { value: "300", label: "5 分钟" }]} /></div>
            </div>
            <button className="primary-button full" onClick={() => onCreateTask(selectedVideo, confidence, sample, segment)} disabled={!selectedVideo || creatingTask}>{creatingTask ? <LoaderCircle className="spin" size={17} /> : <Play size={17} />} {creatingTask ? "正在创建任务" : "加入持久分析队列"}</button>
          </div>
        ) : <EmptyState icon={Video} title="请先上传课堂视频" detail="上传完成后可以在这里设置检测参数并创建任务。" />}
      </article>

      <article className="panel library-panel">
        <div className="panel-heading"><div><span className="eyebrow">LOCAL LIBRARY</span><h2>视频资料库</h2></div><span className="section-total">{videos.length} 个视频</span></div>
        {videos.length ? <div className="video-list">{videos.slice(0, 8).map((video) => (
          <div key={video.id}><span className="video-icon"><Film size={18} /></span><div><strong>{video.course_name || "未命名课程"}</strong><span>{video.video_name}</span></div><StatusBadge status={video.analysis_status || "uploaded"} /></div>
        ))}</div> : <EmptyState icon={Film} title="资料库为空" detail="上传的视频会保存在受管本地目录中。" />}
      </article>

      <article className="panel model-panel">
        <div className="panel-heading"><div><span className="eyebrow">MODEL RUNTIME</span><h2>模型管理</h2></div><BrainCircuit size={20} /></div>
        <div className={cx("model-state", modelInfo?.runtime_available && "ready")}><span><BrainCircuit size={27} /></span><div><small>{modelInfo?.model_family || dashboard.model.family}</small><strong>{modelInfo?.model_name || dashboard.model.name || "课堂行为模型"}</strong><p>{modelInfo?.runtime_message || "正在读取模型运行状态"}</p></div>{modelInfo?.runtime_available ? <CheckCircle2 size={20} /> : <AlertTriangle size={20} />}</div>
        <div className="model-detail"><span>模型权重</span><b title={modelInfo?.model_path}>{modelInfo?.available_models.find((item) => item.is_default)?.name || dashboard.model.weight || "未配置"}</b></div>
        <div className="model-detail"><span>检测配置</span><b title={modelInfo?.config_path}>{modelInfo?.available_configs.find((item) => item.is_default)?.name || dashboard.model.config || "未配置"}</b></div>
        <div className="model-detail"><span>运行设备</span><b>{modelInfo?.device || "—"}</b></div>
        {modelInfo?.deim_missing_dependencies?.length ? <div className="inline-error"><AlertTriangle size={15} />缺少依赖：{modelInfo.deim_missing_dependencies.join("、")}</div> : null}

        <div className="model-assets">
          <div><strong>已登记权重</strong><span>{modelInfo?.available_models.length || 0}</span></div>
          {(modelInfo?.available_models || []).map((item) => <article key={item.path}><Cpu size={15} /><div><b>{item.name}</b><span>{item.family} · {Number(item.size_mb || 0).toFixed(2)} MB</span></div>{item.is_default && <em>默认</em>}</article>)}
          <div><strong>检测配置</strong><span>{modelInfo?.available_configs.length || 0}</span></div>
          {(modelInfo?.available_configs || []).map((item) => <article key={item.path}><FileCog size={15} /><div><b>{item.name}</b><span>{Number(item.size_kb || 0).toFixed(2)} KB</span></div>{item.is_default && <em>默认</em>}</article>)}
        </div>

        <div className="model-upload-grid">
          <form onSubmit={(event) => { event.preventDefault(); onUploadModel(event.currentTarget, "weight"); }}>
            <label className="compact-file"><input name="model_file" type="file" accept=".pt,.pth" required /><UploadCloud size={16} /><span>选择 .pt / .pth 权重</span></label>
            <label className="check-row"><input name="make_default" type="checkbox" defaultChecked /><span>设为默认模型</span></label>
            <button className="secondary-button" type="submit" disabled={modelUploading !== null}>{modelUploading === "weight" ? <LoaderCircle className="spin" size={15} /> : <UploadCloud size={15} />} 上传权重</button>
          </form>
          <form onSubmit={(event) => { event.preventDefault(); onUploadModel(event.currentTarget, "config"); }}>
            <label className="compact-file"><input name="config_file" type="file" accept=".yml,.yaml" required /><FileCog size={16} /><span>选择 .yml / .yaml 配置</span></label>
            <label className="check-row"><input name="make_default" type="checkbox" defaultChecked /><span>设为默认配置</span></label>
            <button className="secondary-button" type="submit" disabled={modelUploading !== null}>{modelUploading === "config" ? <LoaderCircle className="spin" size={15} /> : <UploadCloud size={15} />} 上传配置</button>
          </form>
        </div>
        {modelUploading && <div className="upload-progress"><div><span>正在上传{modelUploading === "weight" ? "模型权重" : "检测配置"}</span><b>{modelUploadProgress}%</b></div><i><span style={{ width: `${modelUploadProgress}%` }} /></i></div>}
        <p className="panel-note"><ShieldCheck size={16} /> 新建任务会记录当时的模型与配置，保证结果可追溯。</p>
      </article>
    </section>
  );
}

export default function App() {
  const [view, setView] = useState<ViewName>("overview");
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [dashboard, setDashboard] = useState<DashboardPayload | null>(null);
  const [health, setHealth] = useState<HealthPayload | null>(null);
  const [modelInfo, setModelInfo] = useState<ModelInfo | null>(null);
  const [videos, setVideos] = useState<VideoItem[]>([]);
  const [selectedReportId, setSelectedReportId] = useState<number | null>(null);
  const [frames, setFrames] = useState<FramePayload | null>(null);
  const [framesLoading, setFramesLoading] = useState(false);
  const [diagnosis, setDiagnosis] = useState<DiagnosisPayload | null>(null);
  const [diagnosing, setDiagnosing] = useState(false);
  const [teachingContext, setTeachingContext] = useState("");
  const [teacherReview, setTeacherReview] = useState<TeacherReview | null>(null);
  const [reviewSaving, setReviewSaving] = useState(false);
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [error, setError] = useState("");
  const [toast, setToast] = useState<{ type: "success" | "error"; message: string } | null>(null);
  const [uploading, setUploading] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [modelUploading, setModelUploading] = useState<"weight" | "config" | null>(null);
  const [modelUploadProgress, setModelUploadProgress] = useState(0);
  const [creatingTask, setCreatingTask] = useState(false);
  const selectedReportIdRef = useRef<number | null>(null);
  const diagnosisRequestRef = useRef<{ reportId: number; controller: AbortController } | null>(null);
  const reviewRequestIdRef = useRef(0);
  const createTaskPendingRef = useRef(false);
  const loadRequestIdRef = useRef(0);

  const loadData = useCallback(async (quiet = false) => {
    const requestId = ++loadRequestIdRef.current;
    if (!quiet) setRefreshing(true);
    try {
      const [nextDashboard, nextHealth, nextVideos, nextModelInfo] = await Promise.all([getDashboard(), getHealth(), getVideos(), getModelInfo()]);
      if (requestId !== loadRequestIdRef.current) return;
      setDashboard(nextDashboard);
      setHealth(nextHealth);
      setVideos(nextVideos);
      setModelInfo(nextModelInfo);
      setSelectedReportId((current) => current && nextDashboard.reports.some((item) => item.id === current) ? current : nextDashboard.reports[0]?.id ?? null);
      setError("");
    } catch (reason) {
      if (requestId === loadRequestIdRef.current) setError(reason instanceof Error ? reason.message : "无法连接本地分析服务");
    } finally {
      if (requestId === loadRequestIdRef.current) {
        setLoading(false);
        setRefreshing(false);
      }
    }
  }, []);

  useEffect(() => {
    let events: EventSource | null = null;
    let interval = 0;
    let disposed = false;
    bootstrapSession()
      .then(() => {
        if (disposed) return;
        void loadData();
        events = new EventSource("/api/tasks/events");
        events.addEventListener("tasks", () => void loadData(true));
        interval = window.setInterval(() => void loadData(true), 15_000);
      })
      .catch((reason) => {
        setError(reason instanceof Error ? reason.message : "无法建立本地安全会话");
        setLoading(false);
      });
    return () => {
      disposed = true;
      events?.close();
      window.clearInterval(interval);
    };
  }, [loadData]);

  useEffect(() => {
    if (!toast) return;
    const timer = window.setTimeout(() => setToast(null), 4200);
    return () => window.clearTimeout(timer);
  }, [toast]);

  const selectedReport = useMemo(() => dashboard?.reports.find((item) => item.id === selectedReportId) || dashboard?.reports[0] || null, [dashboard, selectedReportId]);

  useEffect(() => {
    const reportId = selectedReport?.id ?? null;
    if (selectedReportIdRef.current === reportId) return;
    selectedReportIdRef.current = reportId;
    diagnosisRequestRef.current?.controller.abort();
    diagnosisRequestRef.current = null;
    reviewRequestIdRef.current += 1;
    setDiagnosis(null);
    setDiagnosing(false);
    setTeachingContext("");
    setReviewSaving(false);
  }, [selectedReport?.id]);

  useEffect(() => () => diagnosisRequestRef.current?.controller.abort(), []);

  useEffect(() => {
    let disposed = false;
    setTeacherReview(null);
    if (!selectedReport || view !== "analysis") return () => { disposed = true; };
    getTeacherReview(selectedReport.id)
      .then((value) => { if (!disposed) setTeacherReview(value); })
      .catch((reason) => { if (!disposed) showToast("error", reason instanceof Error ? reason.message : "无法读取改进记录"); });
    return () => { disposed = true; };
  }, [selectedReport?.id, view]);

  useEffect(() => {
    const taskId = selectedReport?.id;
    if (!taskId || view !== "analysis") return;
    let disposed = false;
    setFramesLoading(true);
    setFrames(null);
    getFrames(taskId)
      .then((payload) => { if (!disposed) setFrames(payload); })
      .catch(() => { if (!disposed) setFrames(null); })
      .finally(() => { if (!disposed) setFramesLoading(false); });
    return () => { disposed = true; };
  }, [selectedReport?.id, view]);

  const showToast = (type: "success" | "error", message: string) => setToast({ type, message });

  const handleCancel = async (taskId: string) => {
    try {
      await cancelTask(taskId);
      showToast("success", "取消请求已提交");
      await loadData(true);
    } catch (reason) {
      showToast("error", reason instanceof Error ? reason.message : "无法取消任务");
    }
  };

  const handleDelete = async (taskId: string): Promise<boolean> => {
    try {
      await deleteTask(taskId);
      showToast("success", `任务 #${taskId} 及其检测结果、报告和关键帧已删除`);
      await loadData(true);
      return true;
    } catch (reason) {
      showToast("error", reason instanceof Error ? reason.message : "无法删除任务");
      return false;
    }
  };

  const handleUpload = async (form: HTMLFormElement) => {
    setUploading(true);
    setUploadProgress(0);
    try {
      const result = await uploadVideo(new FormData(form), setUploadProgress);
      form.reset();
      showToast("success", `视频“${result.video_name}”已保存到本地资料库`);
      await loadData(true);
    } catch (reason) {
      showToast("error", reason instanceof Error ? reason.message : "视频上传失败");
    } finally {
      setUploading(false);
      setUploadProgress(0);
    }
  };

  const handleCreateTask = async (videoId: number, confidence: number, sample: number, segment: number) => {
    if (createTaskPendingRef.current) return;
    createTaskPendingRef.current = true;
    setCreatingTask(true);
    try {
      const result = await createAnalysis({ video_id: videoId, confidence_threshold: confidence, frame_sample_seconds: sample, segment_seconds: segment });
      showToast("success", result.created === false ? `相同任务 #${result.task_id} 已在队列中` : `任务 #${result.task_id} 已进入持久队列`);
      setView("tasks");
      await loadData(true);
    } catch (reason) {
      showToast("error", reason instanceof Error ? reason.message : "任务创建失败");
    } finally {
      createTaskPendingRef.current = false;
      setCreatingTask(false);
    }
  };

  const handleDiagnose = async () => {
    if (!selectedReport) return;
    const reportId = selectedReport.id;
    const controller = new AbortController();
    diagnosisRequestRef.current?.controller.abort();
    diagnosisRequestRef.current = { reportId, controller };
    setDiagnosing(true);
    try {
      const result = await generateDiagnosis(reportId, teachingContext, controller.signal);
      if (diagnosisRequestRef.current?.controller !== controller || selectedReportIdRef.current !== reportId) return;
      setDiagnosis(result);
      showToast("success", "情境化辅助诊断已生成");
    } catch (reason) {
      if (!(reason instanceof DOMException && reason.name === "AbortError") && diagnosisRequestRef.current?.controller === controller) {
        showToast("error", reason instanceof Error ? reason.message : "辅助诊断生成失败");
      }
    } finally {
      if (diagnosisRequestRef.current?.controller === controller) {
        diagnosisRequestRef.current = null;
        setDiagnosing(false);
      }
    }
  };

  const handleReviewOcclusion = async (frameId: number) => {
    if (!selectedReport) throw new Error("未选择课堂报告");
    const reportId = selectedReport.id;
    const result = await reviewFrameOcclusion(reportId, frameId);
    if (selectedReportIdRef.current !== reportId) throw new DOMException("课堂已切换", "AbortError");
    if (result.used_llm) showToast("success", `关键帧 ${frameId} 的视觉复核已完成`);
    return result;
  };

  const handleSaveReview = async (payload: Omit<TeacherReview, "task_id" | "updated_at">): Promise<boolean> => {
    if (!selectedReport) return false;
    const reportId = selectedReport.id;
    const requestId = ++reviewRequestIdRef.current;
    setReviewSaving(true);
    try {
      const saved = await saveTeacherReview(reportId, payload);
      if (requestId !== reviewRequestIdRef.current || selectedReportIdRef.current !== reportId) return false;
      setTeacherReview(saved);
      showToast("success", "改进记录已保存到 SQLite");
      return true;
    } catch (reason) {
      if (requestId === reviewRequestIdRef.current) showToast("error", reason instanceof Error ? reason.message : "改进记录保存失败");
      return false;
    } finally {
      if (requestId === reviewRequestIdRef.current) setReviewSaving(false);
    }
  };

  const handleUploadModel = async (form: HTMLFormElement, kind: "weight" | "config") => {
    setModelUploading(kind);
    setModelUploadProgress(0);
    try {
      const data = new FormData(form);
      const checked = (form.elements.namedItem("make_default") as HTMLInputElement | null)?.checked ?? true;
      data.set("make_default", checked ? "true" : "false");
      if (kind === "weight") await uploadModel(data, setModelUploadProgress);
      else await uploadModelConfig(data, setModelUploadProgress);
      form.reset();
      showToast("success", kind === "weight" ? "模型权重上传成功" : "检测配置上传成功");
      await loadData(true);
    } catch (reason) {
      showToast("error", reason instanceof Error ? reason.message : "模型文件上传失败");
    } finally {
      setModelUploading(null);
      setModelUploadProgress(0);
    }
  };

  const navigate = (nextView: ViewName) => {
    setView(nextView);
    setSidebarOpen(false);
    window.scrollTo({ top: 0, behavior: "smooth" });
  };

  const copy = VIEW_COPY[view];

  return (
    <div className="app-shell">
      <aside className={cx("sidebar", sidebarOpen && "open")}>
        <div className="brand"><BrandMark /><div><strong>ClassFocus</strong><span>课堂行为分析</span></div><button className="mobile-close" onClick={() => setSidebarOpen(false)} aria-label="关闭导航"><X size={20} /></button></div>
        <nav aria-label="主导航">
          <span className="nav-label">工作台</span>
          {NAV_ITEMS.map((item) => {
            const Icon = item.icon;
            return <button key={item.id} className={view === item.id ? "active" : ""} onClick={() => navigate(item.id)}><span><Icon size={19} /></span><div><strong>{item.label}</strong><small>{item.detail}</small></div></button>;
          })}
        </nav>
        <div className="sidebar-runtime">
          <div className="sidebar-runtime-head"><span><ServerCog size={17} /></span><div><strong>本地分析服务</strong><small>隐私数据不离开本机</small></div></div>
          <div className="sidebar-runtime-row"><span>API</span><b className={health ? "online" : "offline"}>{health ? "在线" : "离线"}</b></div>
          <div className="sidebar-runtime-row"><span>Worker</span><b className={health?.worker.online ? "online" : "offline"}>{health?.worker.online ? "在线" : "离线"}</b></div>
        </div>
        <footer><ShieldCheck size={15} /><span>仅用于教学证据回看</span></footer>
      </aside>
      {sidebarOpen && <button className="sidebar-backdrop" onClick={() => setSidebarOpen(false)} aria-label="关闭导航遮罩" />}

      <main>
        <header className="topbar">
          <button className="menu-button" onClick={() => setSidebarOpen(true)} aria-label="打开导航"><Menu size={20} /></button>
          <div className="page-title"><span>{copy.eyebrow}</span><h1>{copy.title}</h1><p>{copy.description}</p></div>
          <div className="topbar-actions">
            <ReportPicker reports={dashboard?.reports || []} selectedId={selectedReport?.id || null} onChange={(id) => {
              selectedReportIdRef.current = id;
              diagnosisRequestRef.current?.controller.abort();
              diagnosisRequestRef.current = null;
              reviewRequestIdRef.current += 1;
              setSelectedReportId(id);
              setDiagnosis(null);
              setDiagnosing(false);
              setTeachingContext("");
              setFrames(null);
              setFramesLoading(true);
              setTeacherReview(null);
              setReviewSaving(false);
            }} />
            <button className="icon-button" onClick={() => void loadData()} disabled={refreshing} aria-label="刷新数据" title="刷新数据"><RefreshCw size={18} className={refreshing ? "spin" : ""} /></button>
          </div>
        </header>

        <div className="content">
          {error && <div className="error-banner"><XCircle size={19} /><div><strong>本地分析服务暂不可用</strong><span>{error}</span></div><button onClick={() => void loadData()}><RefreshCw size={15} /> 重试</button></div>}
          {loading ? (
            <div className="page-loader"><BrandMark /><strong>正在连接 ClassFocus</strong><span>读取本地课堂数据与推理状态</span></div>
          ) : dashboard ? (
            <div className="view-stage" key={`${view}-${selectedReport?.id || 0}`}>
              {view === "overview" && <OverviewPage dashboard={dashboard} report={selectedReport} health={health} onNavigate={navigate} onCancel={handleCancel} />}
              {view === "analysis" && <AnalysisPage report={selectedReport} frames={frames?.taskId === selectedReport?.id ? frames : null} framesLoading={framesLoading} diagnosis={diagnosis} diagnosing={diagnosing} teachingContext={teachingContext} setTeachingContext={setTeachingContext} onDiagnose={handleDiagnose} onReviewOcclusion={handleReviewOcclusion} teacherReview={teacherReview?.task_id === selectedReport?.id ? teacherReview : null} reviewSaving={reviewSaving} onSaveReview={handleSaveReview} />}
              {view === "tasks" && <TasksPage dashboard={dashboard} health={health} onCancel={handleCancel} onDelete={handleDelete} />}
              {view === "manage" && <ManagePage videos={videos} dashboard={dashboard} modelInfo={modelInfo} uploading={uploading} uploadProgress={uploadProgress} modelUploading={modelUploading} modelUploadProgress={modelUploadProgress} creatingTask={creatingTask} onUpload={handleUpload} onCreateTask={handleCreateTask} onUploadModel={handleUploadModel} />}
            </div>
          ) : null}
        </div>
      </main>

      {toast && <div className={cx("toast", toast.type)}>{toast.type === "success" ? <CheckCircle2 size={19} /> : <XCircle size={19} />}<span>{toast.message}</span><button onClick={() => setToast(null)} aria-label="关闭提示"><X size={16} /></button></div>}
    </div>
  );
}
