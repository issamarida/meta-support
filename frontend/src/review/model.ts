/**
 * Review-screen logic, kept free of React so it can be unit tested.
 *
 * `classify` and `shouldBlur` mirror backend/src/safeframe/pipeline/policy.py
 * exactly: the preview must show what the export will contain.
 */
import type { ManualRegion, Override, Rect, TrackData } from "../api/types";

export type TrackStatus = "minor" | "adult" | "uncertain";

export interface Policy {
  threshold: number;
  minFaceSamples: number;
}

export function classify(track: TrackData, policy: Policy): TrackStatus {
  if (track.face_samples < policy.minFaceSamples || track.p_minor === null) return "uncertain";
  return track.p_minor >= policy.threshold ? "minor" : "adult";
}

export function shouldBlur(track: TrackData, policy: Policy, override?: Override): boolean {
  if (override === "blur") return true;
  if (override === "keep") return false;
  return classify(track, policy) !== "adult";
}

/** Sensitivity is the user-facing inverse of the P(under 18) threshold. */
export const sensitivityToThreshold = (s: number) => round2(1 - s);
export const thresholdToSensitivity = (t: number) => round2(1 - t);
const round2 = (n: number) => Math.round(n * 100) / 100;

/**
 * Frame index shown at media time `t`. Frame i has presentation time i/fps;
 * the epsilon absorbs floating-point error in reported times.
 */
export function frameAtTime(t: number, fps: number): number {
  return Math.max(0, Math.floor(t * fps + 1e-3));
}

/** Seek target for frame i: mid-frame so the browser lands on exactly that frame. */
export function timeOfFrame(frame: number, fps: number): number {
  return (frame + 0.5) / fps;
}

export interface IndexedTrack extends TrackData {
  byFrame: Map<number, { rect: Rect; interpolated: boolean }>;
}

export function indexTracks(tracks: TrackData[]): IndexedTrack[] {
  return tracks.map((t) => ({
    ...t,
    byFrame: new Map(
      t.boxes.map(([f, x1, y1, x2, y2, interp]) => [
        f,
        { rect: [x1, y1, x2, y2] as Rect, interpolated: interp === 1 },
      ]),
    ),
  }));
}

export interface FrameRegion {
  key: string;
  rect: Rect;
  blurred: boolean;
  kind: "track" | "manual";
  status?: TrackStatus;
  trackId?: number;
  interpolated?: boolean;
}

/** Every region visible on `frame`, with whether it will be blurred on export. */
export function regionsAtFrame(
  frame: number,
  tracks: IndexedTrack[],
  policy: Policy,
  overrides: Record<string, Override>,
  manual: ManualRegion[],
): FrameRegion[] {
  const out: FrameRegion[] = [];
  for (const t of tracks) {
    if (frame < t.first_frame || frame > t.last_frame) continue;
    const hit = t.byFrame.get(frame);
    if (!hit) continue;
    out.push({
      key: `t${t.track_id}`,
      rect: hit.rect,
      blurred: shouldBlur(t, policy, overrides[t.track_id]),
      kind: "track",
      status: classify(t, policy),
      trackId: t.track_id,
      interpolated: hit.interpolated,
    });
  }
  for (const m of manual) {
    if (frame >= m.start_frame && frame <= m.end_frame) {
      out.push({ key: `m${m.region_id}`, rect: m.box, blurred: true, kind: "manual" });
    }
  }
  return out;
}

export interface Summary {
  minor: number;
  adult: number;
  uncertain: number;
  blurred: number;
  kept: number;
}

export function summarise(
  tracks: TrackData[],
  policy: Policy,
  overrides: Record<string, Override>,
): Summary {
  const s: Summary = { minor: 0, adult: 0, uncertain: 0, blurred: 0, kept: 0 };
  for (const t of tracks) {
    s[classify(t, policy)] += 1;
    if (shouldBlur(t, policy, overrides[t.track_id])) s.blurred += 1;
    else s.kept += 1;
  }
  return s;
}

/** Normalise a dragged rectangle (any corner order) and clamp it to the frame. */
export function normaliseRect(
  a: [number, number],
  b: [number, number],
  width: number,
  height: number,
): Rect | null {
  const clamp = (v: number, max: number) => Math.min(Math.max(v, 0), max);
  const x1 = clamp(Math.min(a[0], b[0]), width);
  const y1 = clamp(Math.min(a[1], b[1]), height);
  const x2 = clamp(Math.max(a[0], b[0]), width);
  const y2 = clamp(Math.max(a[1], b[1]), height);
  if (x2 - x1 < 4 || y2 - y1 < 4) return null;
  return [Math.round(x1), Math.round(y1), Math.round(x2), Math.round(y2)];
}

export function formatTime(seconds: number): string {
  const m = Math.floor(seconds / 60);
  const s = seconds - m * 60;
  return `${m}:${s.toFixed(2).padStart(5, "0")}`;
}
