"""Episode video recorder: cameras on the left, model I/O on the right.

Frame layout (1280x720):

    +-----------------+--------------------------------+
    | agentview (top) | header: episode / turn / task  |
    |                 | 模型输出 (assistant text)       |
    +-----------------+ 工具调用 (parsed call)          |
    | wrist (bottom)  | 工具结果 (measured feedback)    |
    |                 | footer: status / retries        |
    +-----------------+--------------------------------+

Robustness rule (the reason this is frame-based, not a live encoder): EVERY
composed frame is written to disk as a PNG the moment it is produced, so a hard
crash / OOM / srun timeout still leaves a usable frame sequence, which
`finalize()` (or `scripts/encode_frames.py` afterwards) turns into an mp4.
On a clean finalize the frames are deleted after the encode succeeds.

Fonts: a CJK font is required (the whole transcript is Chinese). It is NOT
installed on this cluster, so `assets/fonts/wqy-zenhei.ttc` was fetched once
from the aliyun Ubuntu mirror and lives in-repo; if it is missing we fall back
to DejaVu (Chinese renders as boxes) rather than crashing the episode.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent  # arm_agent/
FONT_CANDIDATES = [
    REPO_ROOT / "assets" / "fonts" / "wqy-zenhei.ttc",
    Path("/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc"),
    Path("/usr/share/fonts/dejavu/DejaVuSans.ttf"),
]

FRAME_W, FRAME_H = 1280, 720
CAM_PANEL_W = 640
CAM_IMG = 320  # each camera is drawn at 320x320 (source frames are 256x256)

BG = (14, 16, 20)
PANEL = (26, 29, 35)
BAR = (38, 43, 52)
TEXT = (226, 229, 236)
DIM = (150, 156, 166)
GREEN = (92, 202, 124)
RED = (236, 96, 96)
YELLOW = (236, 200, 92)
ACCENT = (104, 170, 255)


def _load_font(size: int) -> ImageFont.FreeTypeFont:
    for path in FONT_CANDIDATES:
        if Path(path).exists():
            try:
                return ImageFont.truetype(str(path), size)
            except OSError:
                continue
    return ImageFont.load_default(size=size)  # type: ignore[call-arg]


def _wrap(text: str, font: ImageFont.FreeTypeFont, max_w: int) -> list[str]:
    """Character-wise wrap that also handles embedded newlines (CJK-safe)."""
    lines: list[str] = []
    for para in str(text).split("\n"):
        cur = ""
        for ch in para:
            if font.getlength(cur + ch) > max_w:
                # Prefer breaking Latin runs at a space for readability.
                if ch.isascii() and len(cur) > 20 and " " in cur:
                    cut = cur.rfind(" ")
                    lines.append(cur[:cut])
                    cur = cur[cut + 1 :] + ch
                else:
                    lines.append(cur)
                    cur = ch
            else:
                cur += ch
        lines.append(cur)
    return lines


def _draw_lines(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int],
    lines: list[str],
    font: ImageFont.FreeTypeFont,
    fill: tuple[int, int, int],
    line_h: int,
    max_lines: int = 99,
) -> int:
    x, y = xy
    for i, line in enumerate(lines):
        if i >= max_lines:
            draw.text((x, y), "…", font=font, fill=fill)
            y += line_h
            break
        draw.text((x, y), line, font=font, fill=fill)
        y += line_h
    return y


@dataclass
class _FrameInfo:
    turn: int
    reply: str = ""
    tool_call: str = ""
    result: str = ""
    status: str = ""
    status_color: tuple[int, int, int] = DIM
    images: dict[str, Any] = field(default_factory=dict)


class EpisodeRecorder:
    """Compose one frame per turn; PNG now, mp4 at finalize."""

    def __init__(
        self,
        video_dir: Path | str,
        episode_id: int,
        task: str = "",
        fps: int = 2,
        keep_frames: bool = False,
        enabled: bool = True,
    ) -> None:
        self.enabled = enabled
        self.episode_id = episode_id
        self.task = task
        self.fps = fps
        self.keep_frames = keep_frames
        self.frames_written = 0
        self._last_images: dict[str, Any] = {}
        self._t0 = time.time()
        stem = f"ep{episode_id:03d}_{int(time.time())}"
        self.run_dir = Path(video_dir) / stem
        self.frames_dir = self.run_dir / "frames"
        self.video_path = self.run_dir / "video.mp4"
        if self.enabled:
            self.frames_dir.mkdir(parents=True, exist_ok=True)

        self._font_head = _load_font(26)
        self._font_body = _load_font(21)
        self._font_small = _load_font(18)

    # ------------------------------------------------------------------ API
    def record(
        self,
        turn: int,
        *,
        reply: str = "",
        tool_call: str = "",
        result: str = "",
        status: str = "",
        status_color: tuple[int, int, int] = DIM,
        images: dict[str, Any] | None = None,
        extra_line: str = "",
    ) -> None:
        if not self.enabled:
            return
        if images:
            self._last_images = images
        info = _FrameInfo(
            turn=turn,
            reply=reply,
            tool_call=tool_call,
            result=result,
            status=status,
            status_color=status_color,
            images=dict(self._last_images),
        )
        frame = self._compose(info, extra_line=extra_line)
        frame.save(self.frames_dir / f"frame_{self.frames_written:05d}.png")
        self.frames_written += 1

    def finalize(self, *, success: bool, done_declared: bool, turns: int, elapsed_s: float, error: str = "") -> Path | None:
        """Write the summary frame, encode mp4, clean up frames on success."""
        if not self.enabled:
            return None
        status = "SUCCESS" if success else ("FAILED" if not error else f"CRASHED: {error[:80]}")
        color = GREEN if success else RED
        self.record(
            turn=turns,
            reply="",
            tool_call="(episode end)",
            result=(
                f"任务{'成功' if success else '未成功'}；declare_done={done_declared}；"
                f"共 {turns} 轮，用时 {elapsed_s / 60:.1f} 分钟。"
            ),
            status=status,
            status_color=color,
            extra_line=error,
        )
        if self.frames_written == 0:
            return None
        try:
            self._encode()
        except Exception as exc:  # noqa: BLE001 - keep frames for later repair
            print(f"[recorder] encode failed ({exc}); frames kept at {self.frames_dir}", flush=True)
            return None
        if not self.keep_frames:
            for frame in sorted(self.frames_dir.glob("*.png")):
                frame.unlink(missing_ok=True)
            self.frames_dir.rmdir()
        print(f"[recorder] video: {self.video_path} ({self.frames_written} frames)", flush=True)
        return self.video_path

    # ------------------------------------------------------------ internals
    def _encode(self) -> None:
        import imageio.v2 as imageio

        frames = []
        for frame_path in sorted(self.frames_dir.glob("*.png")):
            frames.append(imageio.imread(frame_path))
        if not frames:
            raise RuntimeError("no frames to encode")
        imageio.mimsave(
            str(self.video_path),
            frames,
            fps=self.fps,
            codec="libx264",
            quality=8,
            macro_block_size=None,
            ffmpeg_log_level="error",
        )

    def _compose(self, info: _FrameInfo, extra_line: str = "") -> Image.Image:
        img = Image.new("RGB", (FRAME_W, FRAME_H), BG)
        draw = ImageDraw.Draw(img)

        # ---------------- left: cameras ----------------
        draw.rectangle([0, 0, CAM_PANEL_W - 1, FRAME_H], fill=PANEL)
        self._draw_camera(img, draw, info.images.get("agent"), "agentview（全局视角）", top=True)
        self._draw_camera(img, draw, info.images.get("wrist"), "wrist（腕部视角）", top=False)
        draw.line([(CAM_PANEL_W, 0), (CAM_PANEL_W, FRAME_H)], fill=BAR, width=2)

        # ---------------- right: text ----------------
        x0 = CAM_PANEL_W + 18
        width = FRAME_W - x0 - 18

        # header
        draw.rectangle([x0 - 8, 12, FRAME_W - 10, 62], fill=BAR)
        head = f"episode {self.episode_id}   turn {info.turn}"
        draw.text((x0, 18), head, font=self._font_head, fill=ACCENT)
        if info.status:
            draw.text((x0, 70), info.status, font=self._font_body, fill=info.status_color)

        y = 104
        # task line
        task_lines = _wrap(self.task or "(任务未设置)", self._font_small, width)
        draw.text((x0, y), "任务: " + task_lines[0][:60], font=self._font_small, fill=DIM)
        y += 30

        # model output
        draw.text((x0, y), "模型输出", font=self._font_body, fill=ACCENT)
        y += 30
        reply = info.reply.strip() or "（本轮无前置说明）"
        # Drop the XML call body; it is shown separately below.
        cut = reply.find("<tool_call>")
        if cut != -1:
            reply = (reply[:cut].strip() or "（直接给出工具调用）")
        body_lines = _wrap(reply, self._font_body, width)
        y = _draw_lines(draw, (x0, y), body_lines, self._font_body, TEXT, 27, max_lines=8) + 10

        # tool call
        draw.text((x0, y), "工具调用", font=self._font_body, fill=ACCENT)
        y += 30
        call_text = info.tool_call or "（无）"
        y = _draw_lines(draw, (x0, y), _wrap(call_text, self._font_body, width), self._font_body, YELLOW, 27, max_lines=2) + 10

        # tool result
        draw.text((x0, y), "工具结果", font=self._font_body, fill=ACCENT)
        y += 30
        y = _draw_lines(draw, (x0, y), _wrap(info.result, self._font_small, width), self._font_small, TEXT, 24, max_lines=8) + 8

        if extra_line:
            _draw_lines(draw, (x0, y), _wrap(extra_line, self._font_small, width), self._font_small, RED, 24, max_lines=4)

        # footer
        elapsed = time.time() - self._t0
        draw.text(
            (x0, FRAME_H - 34),
            f"recording {elapsed / 60:.1f} min | frame {self.frames_written}",
            font=self._font_small,
            fill=DIM,
        )
        return img

    def _draw_camera(
        self, img: Image.Image, draw: ImageDraw.ImageDraw, cam_img: Any, label: str, *, top: bool
    ) -> None:
        y0 = 20 if top else 380
        area_h = 320
        if cam_img is None:
            draw.rectangle(
                [CAM_PANEL_W // 2 - CAM_IMG // 2, y0, CAM_PANEL_W // 2 + CAM_IMG // 2, y0 + area_h],
                outline=BAR,
                width=2,
            )
            draw.text((CAM_PANEL_W // 2 - 60, y0 + area_h // 2), "（暂无画面）", font=self._font_body, fill=DIM)
        else:
            im = cam_img.convert("RGB") if hasattr(cam_img, "convert") else Image.fromarray(cam_img)
            im = im.resize((CAM_IMG, CAM_IMG), Image.LANCZOS)
            x = (CAM_PANEL_W - CAM_IMG) // 2
            draw.rectangle([x - 2, y0 - 2, x + CAM_IMG + 2, y0 + area_h + 2], fill=BAR)
            img.paste(im, (x, y0))
        draw.rectangle([12, y0 + 6, 12 + int(self._font_small.getlength(label)) + 12, y0 + 32], fill=BAR)
        draw.text((18, y0 + 8), label, font=self._font_small, fill=TEXT)
