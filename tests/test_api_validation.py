from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import cv2
import numpy as np
from fastapi.testclient import TestClient
from PIL import Image, ImageDraw

from app_api.core.config import settings
from app_api.core.exceptions import TaskOwnershipLost
from app_api.core.security import API_TOKEN_HEADER, get_local_api_token
from app_api.db import crud
from app_api.db.database import get_connection, init_db, reset_initialization_state
from app_api.main import app
from app_api.schemas.task_schema import AgentReportRequest
from app_api.services.llm_service import _request_chat_completion, assess_frame_occlusion
from app_api.services import dashboard_service
from app_api.services.analysis_service import run_analysis
from app_api.services.yolo_service import ClassroomYOLOService
from app_api.services.statistics_service import (
    aggregate_by_segment,
    calculate_overall_statistics,
    detect_warnings,
)


class ApiValidationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.temp_directory = tempfile.TemporaryDirectory()
        root = Path(cls.temp_directory.name)
        cls.original_settings = {
            name: getattr(settings, name)
            for name in (
                "database_path",
                "upload_dir",
                "frame_dir",
                "result_dir",
                "report_dir",
                "model_path",
                "model_config_path",
            )
        }
        replacements = {
            "database_path": root / "classfocus-test.db",
            "upload_dir": root / "uploads" / "videos",
            "frame_dir": root / "uploads" / "frames",
            "result_dir": root / "uploads" / "results",
            "report_dir": root / "uploads" / "reports",
            "model_path": root / "models" / "test.pt",
            "model_config_path": root / "configs" / "test.yml",
        }
        for name, value in replacements.items():
            object.__setattr__(settings, name, value)
        reset_initialization_state()
        init_db()
        cls.task_id = cls._seed_completed_task()
        dashboard_service._cache_key = None
        dashboard_service._cache_payload = None
        cls.client_context = TestClient(app)
        cls.client = cls.client_context.__enter__()
        cls.client.headers.update({API_TOKEN_HEADER: get_local_api_token()})

    @classmethod
    def _seed_completed_task(cls) -> int:
        settings.upload_dir.mkdir(parents=True, exist_ok=True)
        video_path = settings.upload_dir / "fixture.avi"
        writer = cv2.VideoWriter(
            str(video_path),
            cv2.VideoWriter_fourcc(*"MJPG"),
            25,
            (640, 360),
        )
        if not writer.isOpened():
            raise RuntimeError("无法创建测试视频")
        for index in range(80):
            frame = np.full((360, 640, 3), (235 - index % 20, 240, 248), dtype=np.uint8)
            writer.write(frame)
        writer.release()
        course_id, video_id = crud.create_course_with_video(
            {
                "course_name": "测试课程",
                "teacher_name": "测试教师",
                "class_name": "测试班级",
                "classroom": "A101",
                "lesson_date": "2026-08-07",
                "lesson_section": "第1节",
            },
            {
                "video_name": "fixture.avi",
                "video_path": str(video_path),
                "analysis_status": "completed",
                "duration": 4,
                "fps": 25,
                "resolution": "640x360",
            },
        )
        task_id = crud.create_task(
            {
                "video_id": video_id,
                "status": "completed",
                "progress": 100,
                "analysis_mode": "yolo",
            }
        )
        frame_dir = settings.frame_dir / str(task_id)
        frame_dir.mkdir(parents=True, exist_ok=True)
        detections = []
        labels = ["listening", "writing", "using phone", "sleeping"]
        for index, frame_id in enumerate((0, 25, 50, 75)):
            image_path = frame_dir / f"frame_{frame_id}.jpg"
            image = Image.new("RGB", (640, 360), (235 - index * 8, 240, 248))
            draw = ImageDraw.Draw(image)
            draw.rectangle((60, 50, 240, 310), outline=(10, 132, 255), width=5)
            draw.rectangle((330, 70, 520, 320), outline=(48, 209, 88), width=5)
            image.save(image_path, format="JPEG", quality=88)
            for offset in range(2):
                detections.append(
                    {
                        "frame_id": frame_id,
                        "timestamp": float(index),
                        "label": labels[(index + offset) % len(labels)],
                        "confidence": 0.82 - offset * 0.05,
                        "x1": 60 + offset * 270,
                        "y1": 50 + offset * 20,
                        "x2": 240 + offset * 280,
                        "y2": 310 + offset * 10,
                        "image_path": str(image_path),
                    }
                )
        overall = calculate_overall_statistics(detections)
        segments = aggregate_by_segment(detections, segment_seconds=60, duration=4)
        payload = {
            "video_info": {"duration": 4, "fps": 25, "resolution": "640x360", "mode": "yolo"},
            "overall": overall,
            "segments": segments,
            "warnings": detect_warnings(overall, segments),
            "detections": detections,
        }
        crud.replace_detections(task_id, detections)
        crud.replace_segments(task_id, segments)
        result_path = crud.save_result_json(task_id, payload)
        report_path = settings.report_dir / f"classfocus_report_{task_id}.txt"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text("测试报告", encoding="utf-8")
        crud.upsert_report(
            {
                "task_id": task_id,
                "course_id": course_id,
                "main_problem": "待复核片段 1 个",
                "ai_summary": "测试摘要",
                "ai_suggestion": "测试建议",
                "risk_level": "行为证据已生成",
                "report_path": str(report_path),
            }
        )
        crud.update_task(task_id, result_path=str(result_path))
        return task_id

    @classmethod
    def tearDownClass(cls) -> None:
        cls.client_context.__exit__(None, None, None)
        for name, value in cls.original_settings.items():
            object.__setattr__(settings, name, value)
        reset_initialization_state()
        dashboard_service._cache_key = None
        dashboard_service._cache_payload = None
        cls.temp_directory.cleanup()

    def test_health_check(self) -> None:
        response = self.client.get("/api/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "ok")

    def test_missing_video_record_returns_not_found(self) -> None:
        response = self.client.post("/api/tasks/analyze", json={"video_id": 999999})
        self.assertEqual(response.status_code, 404)
        self.assertEqual(response.json()["detail"], "视频不存在")

    def test_model_and_config_uploads_are_registered(self) -> None:
        model_response = self.client.post(
            "/api/models/upload",
            data={"make_default": "false"},
            files={"model_file": ("browser-test.pt", b"fixture-model", "application/octet-stream")},
        )
        self.assertEqual(model_response.status_code, 200)
        self.assertFalse(model_response.json()["is_default"])
        config_response = self.client.post(
            "/api/models/config",
            data={"make_default": "false"},
            files={"config_file": ("browser-test.yaml", b"model: fixture\nclasses: 6\n", "application/x-yaml")},
        )
        self.assertEqual(config_response.status_code, 200)
        current = self.client.get("/api/models/current")
        self.assertEqual(current.status_code, 200)
        self.assertIn("browser-test.pt", [item["name"] for item in current.json()["available_models"]])
        self.assertIn("browser-test.yaml", [item["name"] for item in current.json()["available_configs"]])

    def test_invalid_model_config_is_removed(self) -> None:
        response = self.client.post(
            "/api/models/config",
            data={"make_default": "false"},
            files={"config_file": ("invalid.yaml", b"- not\n- a\n- mapping\n", "application/x-yaml")},
        )
        self.assertEqual(response.status_code, 400)
        self.assertIn("检测配置无效", response.json()["detail"])

    def test_direct_video_path_is_rejected(self) -> None:
        response = self.client.post(
            "/api/tasks/analyze",
            json={"video_path": "uploads/videos/not-found.mp4"},
        )
        self.assertEqual(response.status_code, 422)

    def test_upload_rejects_blank_course_name(self) -> None:
        response = self.client.post(
            "/api/videos/upload",
            data={"course_name": "   "},
            files={"video_file": ("lesson.mp4", b"video", "video/mp4")},
        )
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json()["detail"], "课程名称不能为空")

    def test_upload_rejects_unsupported_extension(self) -> None:
        response = self.client.post(
            "/api/videos/upload",
            data={"course_name": "数据结构"},
            files={"video_file": ("notes.txt", b"not-video", "text/plain")},
        )
        self.assertEqual(response.status_code, 400)
        self.assertIn("不支持的视频格式", response.json()["detail"])

    def test_agent_request_defaults_are_isolated(self) -> None:
        first = AgentReportRequest(overall={})
        second = AgentReportRequest(overall={})
        first.segments.append({"listening_count": 8})
        self.assertEqual(second.segments, [])

    def test_list_limit_is_bounded(self) -> None:
        self.assertEqual(self.client.get("/api/tasks?limit=0").status_code, 422)
        self.assertEqual(self.client.get("/api/videos?limit=201").status_code, 422)

    def test_dashboard_returns_compact_real_data(self) -> None:
        response = self.client.get("/api/tasks/dashboard")
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["schemaVersion"], 3)
        self.assertIn("generatedAt", payload)
        self.assertGreaterEqual(len(payload["reports"]), 1)
        self.assertNotIn("detections", payload["reports"][0])
        self.assertIn("segments", payload["reports"][0])
        self.assertNotIn("attention", payload["reports"][0])
        self.assertNotIn("abnormal", payload["reports"][0])
        self.assertNotIn("risk", payload["reports"][0])
        self.assertIn("reviewSegmentCount", payload["reports"][0])
        task = payload["tasks"][0]
        for field in ("errorMessage", "attemptCount", "workerId", "heartbeatAt", "startedAt", "endedAt"):
            self.assertIn(field, task)

    def test_teacher_review_is_persisted_in_sqlite(self) -> None:
        payload = {
            "owner": "测试教师",
            "due": "下节课前",
            "actions": "在阅读环节增加一次可观察的任务检查。",
            "status": "已提交",
            "review_conclusion": "与课堂任务一致",
            "context_notes": "手机使用发生在扫码答题时段。",
        }
        saved = self.client.post(f"/api/tasks/{self.task_id}/review", json=payload)
        self.assertEqual(saved.status_code, 200)
        self.assertEqual(saved.json()["actions"], payload["actions"])
        self.assertTrue(saved.json()["updated_at"])
        loaded = self.client.get(f"/api/tasks/{self.task_id}/review")
        self.assertEqual(loaded.status_code, 200)
        self.assertEqual(loaded.json()["review_conclusion"], "与课堂任务一致")

    def test_teacher_review_rejects_unknown_status(self) -> None:
        response = self.client.post(
            f"/api/tasks/{self.task_id}/review",
            json={
                "owner": "测试教师",
                "due": "下节课前",
                "status": "未知状态",
                "review_conclusion": "尚未复核",
            },
        )
        self.assertEqual(response.status_code, 422)

    def test_result_file_does_not_duplicate_detection_rows(self) -> None:
        task = crud.get_task(self.task_id)
        with Path(task["result_path"]).open("r", encoding="utf-8") as file:
            payload = json.load(file)
        self.assertNotIn("detections", payload)
        self.assertEqual(payload["detection_count"], 8)

    def test_dashboard_limit_is_bounded(self) -> None:
        self.assertEqual(self.client.get("/api/tasks/dashboard?limit=0").status_code, 422)
        self.assertEqual(self.client.get("/api/tasks/dashboard?limit=101").status_code, 422)

    def test_frame_analysis_returns_safe_visual_data(self) -> None:
        dashboard = self.client.get("/api/tasks/dashboard").json()
        task_id = dashboard["reports"][0]["id"]
        response = self.client.get(f"/api/tasks/{task_id}/frames?limit=4")
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertGreaterEqual(payload["availableFrameCount"], 4)
        self.assertEqual(len(payload["frames"]), 4)
        self.assertIn(payload["highlightFrameId"], [frame["frameId"] for frame in payload["frames"]])
        frame = payload["frames"][0]
        self.assertTrue(frame["imageUrl"].startswith(f"/api/tasks/{task_id}/frames/"))
        self.assertNotIn("image_path", response.text.lower())
        self.assertNotIn("video_path", response.text.lower())
        self.assertIn("box", frame["detections"][0])
        self.assertIn("reviewCueCount", frame)
        self.assertNotIn("nonFocusRate", frame)

    def test_frame_image_is_served_only_for_task_frame(self) -> None:
        dashboard = self.client.get("/api/tasks/dashboard").json()
        task_id = dashboard["reports"][0]["id"]
        frames = self.client.get(f"/api/tasks/{task_id}/frames?limit=4").json()["frames"]
        response = self.client.get(frames[0]["imageUrl"])
        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.headers["content-type"].startswith("image/"))
        self.assertGreater(len(response.content), 1000)
        self.assertEqual(self.client.get(f"/api/tasks/{task_id}/frames/999999999/image").status_code, 404)

    def test_frame_limit_is_bounded(self) -> None:
        self.assertEqual(self.client.get(f"/api/tasks/{self.task_id}/frames?limit=3").status_code, 422)
        self.assertEqual(self.client.get(f"/api/tasks/{self.task_id}/frames?limit=25").status_code, 422)

    def test_frame_occlusion_review_uses_server_side_image(self) -> None:
        dashboard = self.client.get("/api/tasks/dashboard").json()
        task_id = dashboard["reports"][0]["id"]
        frame_id = self.client.get(f"/api/tasks/{task_id}/frames?limit=4").json()["highlightFrameId"]
        fake_result = {"enabled": True, "used_llm": True, "model": "vision-test", "items": [], "summary": {"occluded_count": 0, "ss_count": 0, "so_count": 0, "hg_count": 0}}
        with patch("app_api.routers.task.assess_frame_occlusion", return_value=fake_result) as mocked:
            response = self.client.post(f"/api/tasks/{task_id}/frames/{frame_id}/occlusion")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["model"], "vision-test")
        request_payload = mocked.call_args.args[0]
        self.assertTrue(request_payload["image_base64"])
        self.assertTrue(request_payload["detail_image_base64"])
        self.assertTrue(request_payload["detail_image_batches"])
        self.assertGreater(len(request_payload["detections"]), 0)

    def test_occlusion_review_rejects_incomplete_model_output(self) -> None:
        payload = {
            "image_base64": "dGVzdA==",
            "image_mime": "image/jpeg",
            "detections": [{"id": 0}, {"id": 1}],
        }
        partial = '{"items":[{"id":0,"occlusion_type":"","confidence":0,"reason":"上半身完整可见"}]}'
        with (
            patch.dict(
                os.environ,
                {"VISION_API_KEY": "test-secret", "VISION_MODEL": "test-vision", "VISION_FALLBACK_MODEL": ""},
                clear=False,
            ),
            patch("app_api.services.llm_service._request_chat_completion", return_value=partial),
        ):
            result = assess_frame_occlusion(payload)
        self.assertFalse(result["used_llm"])
        self.assertEqual(result["error_code"], "vision_review_failed")

    def test_occlusion_review_accepts_complete_model_output(self) -> None:
        payload = {
            "image_base64": "dGVzdA==",
            "image_mime": "image/jpeg",
            "detections": [{"id": 0}, {"id": 1}],
        }
        complete = (
            '{"items":['
            '{"id":0,"upper_body_visibility":"partial","blocker":"person","occlusion_type":"S-S","confidence":0.82,"reason":"右肩被相邻学生遮挡"},'
            '{"id":1,"upper_body_visibility":"complete","blocker":"none","occlusion_type":"","confidence":0,"reason":"头肩与躯干连续可见"}'
            ']}'
        )
        with (
            patch.dict(
                os.environ,
                {"VISION_API_KEY": "test-secret", "VISION_MODEL": "test-vision", "VISION_FALLBACK_MODEL": ""},
                clear=False,
            ),
            patch("app_api.services.llm_service._request_chat_completion", return_value=complete),
        ):
            result = assess_frame_occlusion(payload)
        self.assertTrue(result["used_llm"])
        self.assertEqual(result["reviewed_count"], 2)
        self.assertEqual(result["summary"]["ss_count"], 1)

    def test_occlusion_review_retains_moderate_confidence_real_occlusion(self) -> None:
        payload = {
            "image_base64": "dGVzdA==",
            "image_mime": "image/jpeg",
            "detections": [{"id": 0}, {"id": 1}],
        }
        complete = json.dumps(
            {
                "items": [
                    {
                        "id": 0,
                        "upper_body_visibility": "partial",
                        "blocker": "person",
                        "occlusion_type": "S-S",
                        "confidence": 0.56,
                        "reason": "前排学生覆盖目标左肩轮廓",
                    },
                    {
                        "id": 1,
                        "upper_body_visibility": "partial",
                        "blocker": "object",
                        "occlusion_type": "S-O",
                        "confidence": 0.54,
                        "reason": "桌面遮住目标胸部下缘",
                    },
                ]
            },
            ensure_ascii=False,
        )
        with (
            patch.dict(
                os.environ,
                {
                    "VISION_API_KEY": "test-secret",
                    "VISION_MODEL": "test-vision",
                    "VISION_FALLBACK_MODEL": "",
                    "VISION_OCCLUSION_MIN_CONFIDENCE": "0.50",
                },
                clear=False,
            ),
            patch("app_api.services.llm_service._request_chat_completion", return_value=complete),
        ):
            result = assess_frame_occlusion(payload)

        self.assertTrue(result["used_llm"])
        self.assertEqual(result["summary"]["ss_count"], 1)
        self.assertEqual(result["summary"]["so_count"], 1)

    def test_occlusion_review_confidence_threshold_remains_configurable(self) -> None:
        payload = {
            "image_base64": "dGVzdA==",
            "image_mime": "image/jpeg",
            "detections": [{"id": 0}],
        }
        complete = json.dumps(
            {
                "items": [
                    {
                        "id": 0,
                        "upper_body_visibility": "partial",
                        "blocker": "person",
                        "occlusion_type": "S-S",
                        "confidence": 0.56,
                        "reason": "前排学生覆盖目标左肩轮廓",
                    }
                ]
            },
            ensure_ascii=False,
        )
        with (
            patch.dict(
                os.environ,
                {
                    "VISION_API_KEY": "test-secret",
                    "VISION_MODEL": "test-vision",
                    "VISION_FALLBACK_MODEL": "",
                    "VISION_OCCLUSION_MIN_CONFIDENCE": "0.60",
                },
                clear=False,
            ),
            patch("app_api.services.llm_service._request_chat_completion", return_value=complete),
        ):
            result = assess_frame_occlusion(payload)

        self.assertTrue(result["used_llm"])
        self.assertEqual(result["summary"]["occluded_count"], 0)

    def test_occlusion_review_infers_type_from_partial_visibility_and_blocker(self) -> None:
        payload = {
            "image_base64": "dGVzdA==",
            "image_mime": "image/jpeg",
            "detections": [{"id": 0}, {"id": 1}],
        }
        complete = json.dumps(
            {
                "items": [
                    {
                        "id": 0,
                        "upper_body_visibility": "partial",
                        "blocker": "person",
                        "occlusion_type": "",
                        "confidence": 0.41,
                        "reason": "前排学生轻微覆盖目标右侧肩部",
                    },
                    {
                        "id": 1,
                        "upper_body_visibility": "partial",
                        "blocker": "object",
                        "occlusion_type": "",
                        "confidence": 0.38,
                        "reason": "椅背覆盖目标胸腹部边缘",
                    },
                ]
            },
            ensure_ascii=False,
        )
        with (
            patch.dict(
                os.environ,
                {
                    "VISION_API_KEY": "test-secret",
                    "VISION_MODEL": "test-vision",
                    "VISION_FALLBACK_MODEL": "",
                    "VISION_OCCLUSION_MIN_CONFIDENCE": "0.35",
                },
                clear=False,
            ),
            patch("app_api.services.llm_service._request_chat_completion", return_value=complete),
        ):
            result = assess_frame_occlusion(payload)

        self.assertTrue(result["used_llm"])
        self.assertEqual(result["summary"]["ss_count"], 1)
        self.assertEqual(result["summary"]["so_count"], 1)

    def test_occlusion_review_uses_only_the_configured_primary_model(self) -> None:
        payload = {
            "image_base64": "dGVzdA==",
            "image_mime": "image/jpeg",
            "detections": [{"id": 0}],
        }
        complete = (
            '{"items":['
            '{"id":0,"upper_body_visibility":"complete","blocker":"none",'
            '"occlusion_type":"","confidence":0,"reason":"头颈肩胸连续可见"}'
            "]}"
        )
        with (
            patch.dict(
                os.environ,
                {
                    "VISION_API_KEY": "test-secret",
                    "VISION_MODEL": "Qwen/Qwen3.5-9B",
                    "VISION_FALLBACK_MODEL": "",
                    "LLM_OCCLUSION_FALLBACK_MODEL": "",
                },
                clear=False,
            ),
            patch(
                "app_api.services.llm_service._request_chat_completion",
                return_value=complete,
            ) as request_completion,
        ):
            result = assess_frame_occlusion(payload)
        self.assertTrue(result["used_llm"])
        self.assertEqual(result["model"], "Qwen/Qwen3.5-9B")
        self.assertEqual(result["primary_model"], "Qwen/Qwen3.5-9B")
        self.assertFalse(result["used_fallback"])
        self.assertEqual(request_completion.call_count, 1)
        self.assertEqual(request_completion.call_args.args[2], "Qwen/Qwen3.5-9B")

    def test_occlusion_review_batches_large_frames(self) -> None:
        payload = {
            "image_base64": "dGVzdA==",
            "image_mime": "image/jpeg",
            "detections": [{"id": item_id} for item_id in range(25)],
        }

        def complete_batch(start: int, end: int) -> str:
            return json.dumps(
                {
                    "items": [
                        {
                            "id": item_id,
                            "upper_body_visibility": "complete",
                            "blocker": "none",
                            "occlusion_type": "",
                            "confidence": 0,
                            "reason": f"目标 {item_id} 头颈肩胸连续可见",
                        }
                        for item_id in range(start, end)
                    ]
                },
                ensure_ascii=False,
            )

        responses = [complete_batch(0, 12), complete_batch(12, 24), complete_batch(24, 25)]
        with (
            patch.dict(
                os.environ,
                {
                    "VISION_API_KEY": "test-secret",
                    "VISION_MODEL": "test-vision",
                    "VISION_FALLBACK_MODEL": "",
                    "VISION_BATCH_SIZE": "12",
                },
                clear=False,
            ),
            patch(
                "app_api.services.llm_service._request_chat_completion",
                side_effect=responses,
            ) as request_completion,
        ):
            result = assess_frame_occlusion(payload)

        self.assertTrue(result["used_llm"])
        self.assertEqual(result["reviewed_count"], 25)
        self.assertEqual(result["batch_count"], 3)
        self.assertEqual(request_completion.call_count, 3)

    def test_occlusion_review_repairs_incomplete_batch(self) -> None:
        payload = {
            "image_base64": "dGVzdA==",
            "image_mime": "image/jpeg",
            "detections": [{"id": 0}, {"id": 1}],
        }
        incomplete = json.dumps(
            {
                "items": [
                    {
                        "id": 0,
                        "upper_body_visibility": "complete",
                        "blocker": "none",
                        "occlusion_type": "",
                        "confidence": 0,
                        "reason": "目标 0 头颈肩胸连续可见",
                    }
                ]
            },
            ensure_ascii=False,
        )
        complete = json.dumps(
            {
                "items": [
                    {
                        "id": item_id,
                        "upper_body_visibility": "complete",
                        "blocker": "none",
                        "occlusion_type": "",
                        "confidence": 0,
                        "reason": f"目标 {item_id} 头颈肩胸连续可见",
                    }
                    for item_id in range(2)
                ]
            },
            ensure_ascii=False,
        )
        with (
            patch.dict(
                os.environ,
                {
                    "VISION_API_KEY": "test-secret",
                    "VISION_MODEL": "test-vision",
                    "VISION_FALLBACK_MODEL": "",
                    "VISION_FORMAT_RETRY_ATTEMPTS": "2",
                },
                clear=False,
            ),
            patch(
                "app_api.services.llm_service._request_chat_completion",
                side_effect=[incomplete, complete],
            ) as request_completion,
        ):
            result = assess_frame_occlusion(payload)

        self.assertTrue(result["used_llm"])
        self.assertEqual(result["reviewed_count"], 2)
        self.assertEqual(request_completion.call_count, 2)

    def test_chat_completion_retries_rate_limit(self) -> None:
        limited = Mock(status_code=429, text="rate limited", headers={"Retry-After": "0"})
        completed = Mock(status_code=200, text="ok", headers={})
        completed.json.return_value = {
            "choices": [{"message": {"content": '{"ok":true}'}, "finish_reason": "stop"}]
        }
        with (
            patch.dict(os.environ, {"LLM_RETRY_ATTEMPTS": "2"}, clear=False),
            patch("app_api.services.llm_service.requests.post", side_effect=[limited, completed]) as post,
            patch("app_api.services.llm_service.time.sleep"),
        ):
            content = _request_chat_completion(
                "https://example.invalid/v1",
                "secret",
                "test-model",
                [{"role": "user", "content": "return json"}],
                30,
                response_format={"type": "json_object"},
            )

        self.assertEqual(content, '{"ok":true}')
        self.assertEqual(post.call_count, 2)

    def test_chat_completion_retries_incomplete_success_response(self) -> None:
        incomplete = Mock(status_code=200, text="ok", headers={})
        incomplete.json.return_value = {"choices": []}
        completed = Mock(status_code=200, text="ok", headers={})
        completed.json.return_value = {
            "choices": [{"message": {"content": '{"ok":true}'}, "finish_reason": "stop"}]
        }
        with (
            patch.dict(os.environ, {"LLM_RETRY_ATTEMPTS": "2"}, clear=False),
            patch("app_api.services.llm_service.requests.post", side_effect=[incomplete, completed]) as post,
            patch("app_api.services.llm_service.time.sleep"),
        ):
            content = _request_chat_completion(
                "https://example.invalid/v1",
                "secret",
                "test-model",
                [{"role": "user", "content": "return json"}],
                30,
            )

        self.assertEqual(content, '{"ok":true}')
        self.assertEqual(post.call_count, 2)

    def test_llm_status_never_exposes_credentials(self) -> None:
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-secret", "OPENAI_MODEL": "test-model"}, clear=False):
            response = self.client.get("/api/agent/status")
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertTrue(payload["enabled"])
        self.assertEqual(payload["model"], "test-model")
        self.assertIn("text", payload)
        self.assertIn("vision", payload)
        self.assertNotIn("api_key", payload)
        self.assertNotIn("base_url", payload)

    def test_agent_generate_returns_llm_diagnosis(self) -> None:
        with (
            patch.dict(os.environ, {"OPENAI_API_KEY": "test-secret", "OPENAI_MODEL": "test-model"}, clear=False),
            patch(
                "app_api.services.llm_service._request_chat_completion",
                return_value="行为证据汇总：已形成六类行为证据。\n\n情境一致性分析：需教师复核。\n\n教学反思建议：增加一次随堂提问。",
            ),
        ):
            response = self.client.post(
                "/api/agent/generate",
                json={
                    "overall": {
                        "total_count": 100,
                        "raw_counts": {
                            "listening": 60,
                            "writing": 12,
                            "reading": 10,
                            "using phone": 5,
                            "bowing the head": 12,
                            "sleeping": 1,
                        },
                        "behavior_distribution": {
                            "listening": 60,
                            "writing": 12,
                            "reading": 10,
                            "using phone": 5,
                            "bowing the head": 12,
                            "sleeping": 1,
                        },
                    },
                    "segments": [
                        {
                            "time_range": "00:00-01:00",
                            "listening_count": 60,
                            "writing_count": 12,
                            "reading_count": 10,
                            "using_phone_count": 5,
                            "bowing_head_count": 12,
                            "sleeping_count": 1,
                        }
                    ],
                    "course": {"course_name": "数据结构", "class_name": "教技1班", "teaching_context": "本时段为扫码答题"},
                },
            )
        self.assertEqual(response.status_code, 200)
        diagnosis = response.json()["llm_diagnosis"]
        self.assertTrue(diagnosis["enabled"])
        self.assertEqual(diagnosis["model"], "test-model")
        self.assertIn("教学反思建议", diagnosis["content"])

    def test_agent_repairs_incomplete_structured_diagnosis(self) -> None:
        incomplete = '{"behavior_evidence":"已汇总 100 次行为观测。"}'
        complete = json.dumps(
            {
                "behavior_evidence": "共记录 100 次行为观测，其中听课 60 次。",
                "context_consistency": "00:00-01:00 时段需结合课堂任务由教师复核。",
                "teaching_reflection": "下一节课在同类时段增加一次可观察任务检查。",
            },
            ensure_ascii=False,
        )
        with (
            patch.dict(
                os.environ,
                {
                    "OPENAI_API_KEY": "test-secret",
                    "OPENAI_MODEL": "test-model",
                    "LLM_FORMAT_RETRY_ATTEMPTS": "2",
                },
                clear=False,
            ),
            patch(
                "app_api.services.llm_service._request_chat_completion",
                side_effect=[incomplete, complete],
            ) as request_completion,
        ):
            response = self.client.post(
                "/api/agent/generate",
                json={"overall": {"total_count": 100}, "segments": [], "course": {}},
            )

        self.assertEqual(response.status_code, 200)
        diagnosis = response.json()["llm_diagnosis"]
        self.assertTrue(diagnosis["enabled"])
        self.assertIn("行为证据汇总：", diagnosis["content"])
        self.assertIn("情境一致性分析：", diagnosis["content"])
        self.assertIn("教学反思建议：", diagnosis["content"])
        self.assertEqual(request_completion.call_count, 2)

    def test_behavior_statistics_do_not_create_cognitive_scores(self) -> None:
        overall = calculate_overall_statistics(
            [
                {"timestamp": 1, "label": "listening"},
                {"timestamp": 2, "label": "bowing the head"},
                {"timestamp": 3, "label": "using phone"},
            ]
        )
        self.assertNotIn("attention_rate", overall)
        self.assertNotIn("abnormal_rate", overall)
        self.assertNotIn("risk_level", overall)
        self.assertEqual(overall["total_count"], 3)
        self.assertEqual(overall["class_count"], 3)

    def test_low_head_alone_is_not_marked_for_priority_review(self) -> None:
        detections = [{"timestamp": second, "label": "bowing the head"} for second in range(12)]
        overall = calculate_overall_statistics(detections)
        segments = aggregate_by_segment(detections, segment_seconds=60, duration=60)
        warnings = detect_warnings(overall, segments)
        self.assertFalse(segments[0]["requires_review"])
        self.assertEqual(segments[0]["review_priority"], "常规")
        self.assertNotIn("低头", " ".join(item["detail"] for item in warnings))

    def test_phone_and_sleeping_are_context_review_cues_only(self) -> None:
        detections = [
            {"timestamp": 2, "label": "using phone"},
            {"timestamp": 65, "label": "sleeping"},
        ]
        overall = calculate_overall_statistics(detections)
        segments = aggregate_by_segment(detections, segment_seconds=60, duration=120)
        warnings = detect_warnings(overall, segments)
        text = " ".join(item["detail"] for item in warnings)
        self.assertTrue(all(segment["requires_review"] for segment in segments))
        self.assertIn("结合课堂任务", text)
        self.assertIn("不直接推断认知状态", text)

    def test_api_requires_local_token(self) -> None:
        response = self.client.get("/api/tasks", headers={API_TOKEN_HEADER: ""})
        self.assertEqual(response.status_code, 401)

    def test_react_workbench_can_create_httponly_session(self) -> None:
        with TestClient(app, headers={"Origin": "http://127.0.0.1:5173"}) as browser:
            response = browser.post("/api/session")
            self.assertEqual(response.status_code, 200)
            cookie = response.headers.get("set-cookie", "")
            self.assertIn("HttpOnly", cookie)
            self.assertIn("SameSite=strict", cookie)
            tasks = browser.get("/api/tasks")
            self.assertEqual(tasks.status_code, 200)

    def test_sqlite_worker_claims_and_cancels_persistent_tasks(self) -> None:
        source = crud.get_task(self.task_id)
        self.assertIsNotNone(source)
        video_id = int(source["video_id"])
        task_id = crud.create_task({"video_id": video_id, "status": "waiting"})
        try:
            claimed = crud.claim_next_task("test-worker", lease_seconds=45)
            self.assertIsNotNone(claimed)
            self.assertEqual(int(claimed["id"]), task_id)
            self.assertEqual(claimed["worker_id"], "test-worker")
            self.assertEqual(crud.request_task_cancellation(task_id), "canceling")
            self.assertTrue(crud.is_task_cancel_requested(task_id))
        finally:
            crud.update_task(task_id, status="failed", progress=100)
            crud.delete_task(task_id)
            crud.update_video_status(video_id, "completed")

    def test_running_analysis_cancellation_stays_canceled_and_releases_video(self) -> None:
        source = crud.get_task(self.task_id)
        self.assertIsNotNone(source)
        video_id = int(source["video_id"])
        task_id = crud.create_task({"video_id": video_id, "status": "waiting"})
        worker_id = "cancel-during-inference-worker"
        claimed = crud.claim_next_task(worker_id, lease_seconds=45)
        self.assertIsNotNone(claimed)
        self.assertEqual(int(claimed["id"]), task_id)

        class FakeCapture:
            def __init__(self) -> None:
                self.read_count = 0
                self.released = False

            def isOpened(self) -> bool:
                return True

            def get(self, prop: int) -> float:
                values = {
                    cv2.CAP_PROP_FPS: 25,
                    cv2.CAP_PROP_FRAME_COUNT: 1,
                    cv2.CAP_PROP_FRAME_WIDTH: 640,
                    cv2.CAP_PROP_FRAME_HEIGHT: 360,
                }
                return float(values.get(prop, 0))

            def read(self):
                self.read_count += 1
                if self.read_count == 1:
                    return True, np.zeros((360, 640, 3), dtype=np.uint8)
                return False, None

            def release(self) -> None:
                self.released = True

        capture = FakeCapture()
        service = ClassroomYOLOService.__new__(ClassroomYOLOService)
        service.mode = "yolo"
        service.message = "test model"
        service.device = "cpu"
        service.model = Mock()
        service.model.predict.side_effect = lambda **_: (
            crud.request_task_cancellation(task_id),
            [],
        )[1]
        service._parse_result = Mock(return_value=[])
        model_cache = Mock()
        model_cache.get.return_value = service

        try:
            with patch("cv2.VideoCapture", return_value=capture):
                run_analysis(task_id, worker_id=worker_id, model_cache=model_cache)
            task = crud.get_task(task_id)
            self.assertEqual(task["status"], "canceled")
            self.assertNotEqual(task["status"], "failed")
            self.assertTrue(capture.released)
        finally:
            crud.update_task(task_id, status="failed", progress=100)
            crud.delete_task(task_id)
            crud.update_video_status(video_id, "completed")

    def test_duplicate_active_analysis_request_reuses_task(self) -> None:
        source = crud.get_task(self.task_id)
        self.assertIsNotNone(source)
        video_id = int(source["video_id"])
        payload = {
            "video_id": video_id,
            "confidence_threshold": 0.61,
            "frame_sample_seconds": 2,
            "segment_seconds": 120,
        }
        first_id: int | None = None
        try:
            first = self.client.post("/api/tasks/analyze", json=payload)
            second = self.client.post("/api/tasks/analyze", json=payload)
            self.assertEqual(first.status_code, 202)
            self.assertEqual(second.status_code, 202)
            self.assertTrue(first.json()["created"])
            self.assertFalse(second.json()["created"])
            first_id = int(first.json()["task_id"])
            self.assertEqual(first_id, int(second.json()["task_id"]))
        finally:
            if first_id is not None:
                crud.update_task(first_id, status="failed", progress=100)
                crud.delete_task(first_id)
            crud.update_video_status(video_id, "completed")

    def test_dashboard_cache_changes_after_video_upload(self) -> None:
        dashboard_service._cache_key = None
        dashboard_service._cache_payload = None
        before = dashboard_service.build_dashboard_payload(use_cache=True)["summary"]["videoCount"]
        course_id, video_id = crud.create_course_with_video(
            {"course_name": "缓存测试课程"},
            {"video_name": "cache-test.mp4", "video_path": str(settings.upload_dir / "cache-test.mp4")},
        )
        try:
            after = dashboard_service.build_dashboard_payload(use_cache=True)["summary"]["videoCount"]
            self.assertEqual(after, before + 1)
        finally:
            with get_connection() as connection:
                connection.execute("DELETE FROM video WHERE id = ?", (video_id,))
                connection.execute("DELETE FROM course WHERE id = ?", (course_id,))
                connection.commit()
            dashboard_service._cache_key = None
            dashboard_service._cache_payload = None

    def test_stale_worker_cannot_write_after_task_is_reclaimed(self) -> None:
        source = crud.get_task(self.task_id)
        self.assertIsNotNone(source)
        video_id = int(source["video_id"])
        task_id = crud.create_task({"video_id": video_id, "status": "waiting"})
        try:
            first = crud.claim_next_task("worker-a", lease_seconds=15)
            self.assertEqual(int(first["id"]), task_id)
            crud.update_task(task_id, heartbeat_at="2000-01-01T00:00:00")
            crud.recover_expired_tasks(lease_seconds=15)
            second = crud.claim_next_task("worker-b", lease_seconds=15)
            self.assertEqual(int(second["id"]), task_id)
            self.assertFalse(crud.update_task_owned(task_id, "worker-a", progress=88))
            with self.assertRaises(TaskOwnershipLost):
                crud.replace_segments(task_id, [], worker_id="worker-a")
            self.assertTrue(crud.update_task_owned(task_id, "worker-b", progress=42))
            task = crud.get_task(task_id)
            self.assertEqual(task["worker_id"], "worker-b")
            self.assertEqual(float(task["progress"]), 42)
        finally:
            crud.update_task(task_id, status="failed", progress=100)
            crud.delete_task(task_id)
            crud.update_video_status(video_id, "completed")

    def test_delete_file_staging_failure_preserves_database_record(self) -> None:
        source = crud.get_task(self.task_id)
        self.assertIsNotNone(source)
        video_id = int(source["video_id"])
        task_id = crud.create_task({"video_id": video_id, "status": "failed", "progress": 100})
        result_path = crud.save_result_json(task_id, {"overall": {}})
        crud.update_task(task_id, result_path=str(result_path))
        try:
            with patch("app_api.db.crud._stage_generated_paths", side_effect=OSError("locked")):
                response = self.client.delete(f"/api/tasks/{task_id}")
            self.assertEqual(response.status_code, 409)
            self.assertIsNotNone(crud.get_task(task_id))
            self.assertTrue(result_path.exists())
        finally:
            crud.delete_task(task_id)
            crud.update_video_status(video_id, "completed")

    def test_sqlite_uses_wal_journal_mode(self) -> None:
        with get_connection() as connection:
            mode = connection.execute("PRAGMA journal_mode").fetchone()[0]
        self.assertEqual(str(mode).lower(), "wal")

    def test_sqlite_worker_recovers_expired_lease(self) -> None:
        source = crud.get_task(self.task_id)
        self.assertIsNotNone(source)
        video_id = int(source["video_id"])
        task_id = crud.create_task({"video_id": video_id, "status": "waiting"})
        try:
            claimed = crud.claim_next_task("expired-worker", lease_seconds=15)
            self.assertIsNotNone(claimed)
            crud.update_task(task_id, heartbeat_at="2000-01-01T00:00:00")
            recovered = crud.recover_expired_tasks(lease_seconds=15)
            self.assertGreaterEqual(recovered, 1)
            task = crud.get_task(task_id)
            self.assertEqual(task["status"], "waiting")
            self.assertIsNone(task["worker_id"])
        finally:
            crud.update_task(task_id, status="failed", progress=100)
            crud.delete_task(task_id)
            crud.update_video_status(video_id, "completed")

    def test_canceling_task_cannot_be_deleted(self) -> None:
        source = crud.get_task(self.task_id)
        self.assertIsNotNone(source)
        video_id = int(source["video_id"])
        task_id = crud.create_task({"video_id": video_id, "status": "waiting"})
        try:
            claimed = crud.claim_next_task("delete-guard-worker", lease_seconds=45)
            self.assertIsNotNone(claimed)
            self.assertEqual(int(claimed["id"]), task_id)
            self.assertEqual(crud.request_task_cancellation(task_id), "canceling")
            response = self.client.delete(f"/api/tasks/{task_id}")
            self.assertEqual(response.status_code, 409)
        finally:
            crud.update_task(task_id, status="failed", progress=100)
            crud.delete_task(task_id)
            crud.update_video_status(video_id, "completed")

    def test_cors_allows_only_local_frontend(self) -> None:
        response = self.client.get(
            "/api/health",
            headers={"Origin": "http://127.0.0.1:5173"},
        )
        self.assertEqual(response.headers.get("access-control-allow-origin"), "http://127.0.0.1:5173")
        self.assertEqual(response.headers.get("access-control-allow-credentials"), "true")
        blocked = self.client.get("/api/health", headers={"Origin": "https://example.com"})
        self.assertIsNone(blocked.headers.get("access-control-allow-origin"))


if __name__ == "__main__":
    unittest.main()
