import type {
  DashboardPayload,
  DiagnosisPayload,
  FramePayload,
  HealthPayload,
  ModelInfo,
  OcclusionReview,
  TeacherReview,
  VideoItem,
} from "./types";

export class ApiError extends Error {
  status: number;

  constructor(message: string, status = 0) {
    super(message);
    this.name = "ApiError";
    this.status = status;
  }
}

let sessionRequest: Promise<void> | null = null;

async function errorFromResponse(response: Response, fallbackPrefix = "请求失败"): Promise<ApiError> {
  let detail = `${fallbackPrefix}（${response.status}）`;
  try {
    const payload = (await response.json()) as { detail?: string };
    detail = payload.detail || detail;
  } catch {
    // Keep the status-based fallback.
  }
  return new ApiError(detail, response.status);
}

function authenticatedFetch(path: string, init: RequestInit): Promise<Response> {
  return fetch(path, {
    ...init,
    credentials: "include",
    headers: {
      ...(init.body instanceof FormData ? {} : { "Content-Type": "application/json" }),
      ...(init.headers || {}),
    },
  });
}

async function apiFetch<T>(path: string, init: RequestInit = {}, retrySession = true): Promise<T> {
  let response = await authenticatedFetch(path, init);
  if (response.status === 401 && retrySession && path !== "/api/session") {
    await bootstrapSession();
    response = await authenticatedFetch(path, init);
  }
  if (!response.ok) {
    throw await errorFromResponse(response);
  }
  return (await response.json()) as T;
}

export async function bootstrapSession(): Promise<void> {
  if (!sessionRequest) {
    sessionRequest = authenticatedFetch("/api/session", { method: "POST" })
      .then(async (response) => {
        if (!response.ok) throw await errorFromResponse(response, "无法建立本地安全会话");
      })
      .finally(() => {
        sessionRequest = null;
      });
  }
  await sessionRequest;
}

export function getDashboard(): Promise<DashboardPayload> {
  return apiFetch<DashboardPayload>("/api/tasks/dashboard");
}

export function getHealth(): Promise<HealthPayload> {
  return apiFetch<HealthPayload>("/api/health");
}

export async function getVideos(): Promise<VideoItem[]> {
  const payload = await apiFetch<{ items: VideoItem[] }>("/api/videos?limit=100");
  return payload.items;
}

export function createAnalysis(payload: {
  video_id: number;
  confidence_threshold: number;
  frame_sample_seconds: number;
  segment_seconds: number;
}): Promise<{ task_id: number; message: string; created?: boolean }> {
  return apiFetch("/api/tasks/analyze", {
    method: "POST",
    body: JSON.stringify(payload),
  });
}

export function cancelTask(taskId: string): Promise<{ status: string }> {
  return apiFetch(`/api/tasks/${taskId}/cancel`, { method: "POST" });
}

export function deleteTask(taskId: string): Promise<{ task_id: number; message: string }> {
  return apiFetch(`/api/tasks/${taskId}`, { method: "DELETE" });
}

export function getFrames(taskId: number): Promise<FramePayload> {
  return apiFetch<FramePayload>(`/api/tasks/${taskId}/frames?limit=8`);
}

export function generateDiagnosis(taskId: number, teachingContext: string, signal?: AbortSignal): Promise<DiagnosisPayload> {
  return apiFetch<DiagnosisPayload>(`/api/agent/task/${taskId}`, {
    method: "POST",
    body: JSON.stringify({ teaching_context: teachingContext }),
    signal,
  });
}

export function reviewFrameOcclusion(taskId: number, frameId: number): Promise<OcclusionReview> {
  return apiFetch<OcclusionReview>(`/api/tasks/${taskId}/frames/${frameId}/occlusion`, { method: "POST" });
}

export function getTeacherReview(taskId: number): Promise<TeacherReview> {
  return apiFetch<TeacherReview>(`/api/tasks/${taskId}/review`);
}

export function saveTeacherReview(taskId: number, payload: Omit<TeacherReview, "task_id" | "updated_at">): Promise<TeacherReview> {
  return apiFetch<TeacherReview>(`/api/tasks/${taskId}/review`, {
    method: "POST",
    body: JSON.stringify(payload),
  });
}

export function getModelInfo(): Promise<ModelInfo> {
  return apiFetch<ModelInfo>("/api/models/current");
}

function uploadFile<T>(path: string, formData: FormData, onProgress: (value: number) => void, retrySession = true): Promise<T> {
  return new Promise((resolve, reject) => {
    const request = new XMLHttpRequest();
    request.open("POST", path);
    request.withCredentials = true;
    request.upload.addEventListener("progress", (event) => {
      if (event.lengthComputable) onProgress(Math.round((event.loaded / event.total) * 100));
    });
    request.addEventListener("load", () => {
      let payload: { detail?: string } = {};
      try {
        payload = JSON.parse(request.responseText) as typeof payload;
      } catch {
        // Handled by the generic status fallback.
      }
      if (request.status >= 200 && request.status < 300) {
        resolve(payload as T);
      } else if (request.status === 401 && retrySession) {
        bootstrapSession()
          .then(() => uploadFile<T>(path, formData, onProgress, false))
          .then(resolve, reject);
      } else {
        reject(new ApiError(payload.detail || `上传失败（${request.status}）`, request.status));
      }
    });
    request.addEventListener("error", () => reject(new ApiError("无法连接本地分析服务")));
    request.send(formData);
  });
}

export function uploadModel(formData: FormData, onProgress: (value: number) => void): Promise<{ message: string; is_default: boolean }> {
  return uploadFile("/api/models/upload", formData, onProgress);
}

export function uploadModelConfig(formData: FormData, onProgress: (value: number) => void): Promise<{ message: string; is_default: boolean }> {
  return uploadFile("/api/models/config", formData, onProgress);
}

export function uploadVideo(formData: FormData, onProgress: (value: number) => void): Promise<{ video_id: number; video_name: string }> {
  return uploadFile<{ video_id?: number; video_name?: string }>("/api/videos/upload", formData, onProgress)
    .then((payload) => {
      if (!payload.video_id) throw new ApiError("视频上传响应不完整");
      return { video_id: payload.video_id, video_name: payload.video_name || "课堂视频" };
    });
}
