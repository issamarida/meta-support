import { useCallback, useEffect, useRef, useState } from "react";
import type { ManualRegion, Override, Rect, VideoInfo } from "../api/types";
import {
  frameAtTime,
  normaliseRect,
  regionsAtFrame,
  type IndexedTrack,
  type Policy,
} from "../review/model";

export type ViewMode = "preview" | "boxes";

interface Props {
  src: string;
  video: VideoInfo;
  tracks: IndexedTrack[];
  policy: Policy;
  overrides: Record<string, Override>;
  manual: ManualRegion[];
  mode: ViewMode;
  showBoxes: boolean;
  drawing: boolean;
  selectedTrack: number | null;
  selectedRegion: string | null;
  onFrame: (frame: number) => void;
  onDrawn: (rect: Rect) => void;
  onSelectTrack: (id: number | null) => void;
  videoRef: React.RefObject<HTMLVideoElement | null>;
}

// Same padding the renderer adds around every region (anonymize.DEFAULT_PADDING).
const PADDING = 0.08;

const COLORS = {
  minor: "#e5484d",
  uncertain: "#f5a524",
  adult: "#30a46c",
  manual: "#8e4ec6",
  kept: "#30a46c",
};

export function VideoStage(props: Props) {
  const { src, video, drawing, onFrame, onDrawn, videoRef } = props;
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const frameRef = useRef(0);
  const dragRef = useRef<{ a: [number, number]; b: [number, number] } | null>(null);
  const latest = useRef(props);
  latest.current = props;
  const [filterSupported] = useState(() => {
    const ctx = document.createElement("canvas").getContext("2d");
    return !!ctx && "filter" in ctx;
  });

  const draw = useCallback(() => {
    const canvas = canvasRef.current;
    const v = videoRef.current;
    if (!canvas || !v || !v.videoWidth) return;
    const p = latest.current;
    if (canvas.width !== v.videoWidth || canvas.height !== v.videoHeight) {
      canvas.width = v.videoWidth;
      canvas.height = v.videoHeight;
    }
    const ctx = canvas.getContext("2d")!;
    const s = canvas.width / p.video.width; // source px -> preview px
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    const regions = regionsAtFrame(frameRef.current, p.tracks, p.policy, p.overrides, p.manual);
    for (const r of regions) {
      const [x1, y1, x2, y2] = r.rect;
      const padX = (x2 - x1) * PADDING;
      const padY = (y2 - y1) * PADDING;
      const rx = Math.max(0, (x1 - padX) * s);
      const ry = Math.max(0, (y1 - padY) * s);
      const rw = Math.min(canvas.width, (x2 + padX) * s) - rx;
      const rh = Math.min(canvas.height, (y2 + padY) * s) - ry;
      if (rw <= 0 || rh <= 0) continue;

      if (p.mode === "preview" && r.blurred) {
        ctx.save();
        ctx.beginPath();
        ctx.rect(rx, ry, rw, rh);
        ctx.clip();
        if (filterSupported) {
          const radius = Math.max(4, Math.max(rw, rh) / 14);
          const m = radius * 3;
          ctx.filter = `blur(${radius}px) grayscale(1)`;
          ctx.drawImage(v, rx - m, ry - m, rw + 2 * m, rh + 2 * m, rx - m, ry - m, rw + 2 * m, rh + 2 * m);
        } else {
          ctx.fillStyle = "#808080";
          ctx.fillRect(rx, ry, rw, rh);
        }
        ctx.restore();
      }

      if (p.showBoxes || p.mode === "boxes") {
        const selected =
          (r.trackId !== undefined && r.trackId === p.selectedTrack) ||
          (r.kind === "manual" && r.key === `m${p.selectedRegion}`);
        const color =
          r.kind === "manual" ? COLORS.manual : r.blurred ? COLORS[r.status ?? "minor"] : COLORS.kept;
        ctx.lineWidth = selected ? 4 : 2;
        ctx.strokeStyle = color;
        ctx.setLineDash(r.interpolated ? [6, 4] : r.blurred ? [] : [2, 3]);
        ctx.strokeRect(rx, ry, rw, rh);
        ctx.setLineDash([]);
        const label =
          r.kind === "manual" ? "manual" : `#${r.trackId} ${r.blurred ? "blur" : "keep"}`;
        ctx.font = "600 12px system-ui, sans-serif";
        const tw = ctx.measureText(label).width + 8;
        ctx.fillStyle = color;
        ctx.fillRect(rx, Math.max(0, ry - 16), tw, 16);
        ctx.fillStyle = "#fff";
        ctx.fillText(label, rx + 4, Math.max(12, ry - 4));
      }
    }

    const drag = dragRef.current;
    if (drag) {
      const [ax, ay] = drag.a;
      const [bx, by] = drag.b;
      ctx.strokeStyle = COLORS.manual;
      ctx.lineWidth = 2;
      ctx.setLineDash([5, 3]);
      ctx.strokeRect(Math.min(ax, bx) * s, Math.min(ay, by) * s, Math.abs(bx - ax) * s, Math.abs(by - ay) * s);
      ctx.setLineDash([]);
    }
  }, [filterSupported, videoRef]);

  // Redraw whenever decisions change.
  useEffect(draw, [draw, props.policy, props.overrides, props.manual, props.mode, props.showBoxes,
    props.selectedTrack, props.selectedRegion, props.tracks]);

  // Keep the overlay locked to the displayed frame.
  useEffect(() => {
    const v = videoRef.current;
    if (!v) return;
    const fps = video.fps;
    let handle = 0;
    const update = (t: number) => {
      const f = Math.min(frameAtTime(t, fps), video.frame_count - 1);
      frameRef.current = f;
      onFrame(f);
      draw();
    };
    const hasRvfc = "requestVideoFrameCallback" in v;
    const onVideoFrame = (_: number, meta: { mediaTime: number }) => {
      update(meta.mediaTime);
      handle = v.requestVideoFrameCallback(onVideoFrame);
    };
    let raf = 0;
    const onRaf = () => {
      update(v.currentTime);
      if (!v.paused) raf = requestAnimationFrame(onRaf);
    };
    const onPlay = () => {
      if (!hasRvfc) raf = requestAnimationFrame(onRaf);
    };
    const onStill = () => update(v.currentTime);
    if (hasRvfc) handle = v.requestVideoFrameCallback(onVideoFrame);
    v.addEventListener("play", onPlay);
    v.addEventListener("seeked", onStill);
    v.addEventListener("loadeddata", onStill);
    v.addEventListener("pause", onStill);
    return () => {
      if (hasRvfc) v.cancelVideoFrameCallback(handle);
      cancelAnimationFrame(raf);
      v.removeEventListener("play", onPlay);
      v.removeEventListener("seeked", onStill);
      v.removeEventListener("loadeddata", onStill);
      v.removeEventListener("pause", onStill);
    };
  }, [draw, onFrame, video.fps, video.frame_count, videoRef]);

  const toSource = (e: React.PointerEvent<HTMLCanvasElement>): [number, number] => {
    const box = e.currentTarget.getBoundingClientRect();
    return [
      ((e.clientX - box.left) / box.width) * video.width,
      ((e.clientY - box.top) / box.height) * video.height,
    ];
  };

  const onPointerDown = (e: React.PointerEvent<HTMLCanvasElement>) => {
    const pt = toSource(e);
    if (drawing) {
      e.currentTarget.setPointerCapture(e.pointerId);
      dragRef.current = { a: pt, b: pt };
      return;
    }
    // Click a box to select its track.
    const p = latest.current;
    const hit = regionsAtFrame(frameRef.current, p.tracks, p.policy, p.overrides, [])
      .filter((r) => r.rect[0] <= pt[0] && pt[0] <= r.rect[2] && r.rect[1] <= pt[1] && pt[1] <= r.rect[3])
      .sort((a, b) => area(a.rect) - area(b.rect))[0];
    p.onSelectTrack(hit?.trackId ?? null);
  };
  const onPointerMove = (e: React.PointerEvent<HTMLCanvasElement>) => {
    if (!dragRef.current) return;
    dragRef.current.b = toSource(e);
    draw();
  };
  const onPointerUp = () => {
    const drag = dragRef.current;
    dragRef.current = null;
    if (!drag) return;
    const rect = normaliseRect(drag.a, drag.b, video.width, video.height);
    if (rect) onDrawn(rect);
    draw();
  };

  return (
    <div className="stage" style={{ aspectRatio: `${video.width} / ${video.height}` }}>
      <video ref={videoRef} src={src} preload="auto" playsInline muted crossOrigin="anonymous" />
      <canvas
        ref={canvasRef}
        className={drawing ? "drawing" : ""}
        onPointerDown={onPointerDown}
        onPointerMove={onPointerMove}
        onPointerUp={onPointerUp}
        aria-label={drawing ? "Drag to draw a region to blur" : "Video preview with detected people"}
      />
    </div>
  );
}

const area = (r: Rect) => (r[2] - r[0]) * (r[3] - r[1]);
