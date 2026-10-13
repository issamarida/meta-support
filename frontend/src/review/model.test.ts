import { describe, expect, it } from "vitest";
import type { TrackData } from "../api/types";
import {
  classify,
  frameAtTime,
  indexTracks,
  normaliseRect,
  regionsAtFrame,
  sensitivityToThreshold,
  shouldBlur,
  summarise,
  timeOfFrame,
} from "./model";

const policy = { threshold: 0.35, minFaceSamples: 3 };

function track(id: number, p: number | null, samples: number, frames = [0, 1, 2]): TrackData {
  return {
    track_id: id,
    first_frame: frames[0]!,
    last_frame: frames[frames.length - 1]!,
    p_minor: p,
    expected_age: null,
    face_samples: samples,
    peak_confidence: 0.9,
    boxes: frames.map((f) => [f, 10, 10, 50, 90, 0]),
  };
}

describe("policy (mirrors backend)", () => {
  it("treats too little face evidence as uncertain, and blurs it", () => {
    const t = track(1, 0.01, 2);
    expect(classify(t, policy)).toBe("uncertain");
    expect(shouldBlur(t, policy)).toBe(true);
  });

  it("splits minors and adults at the threshold", () => {
    expect(classify(track(1, 0.35, 5), policy)).toBe("minor");
    expect(classify(track(1, 0.34, 5), policy)).toBe("adult");
    expect(shouldBlur(track(1, 0.1, 5), policy)).toBe(false);
  });

  it("lets reviewer overrides win", () => {
    expect(shouldBlur(track(1, 0.0, 9), policy, "blur")).toBe(true);
    expect(shouldBlur(track(1, 1.0, 9), policy, "keep")).toBe(false);
  });

  it("maps sensitivity to threshold", () => {
    expect(sensitivityToThreshold(0.65)).toBe(0.35);
    expect(sensitivityToThreshold(1)).toBe(0);
  });
});

describe("frame timing", () => {
  it("round-trips frame -> seek time -> frame", () => {
    for (const fps of [24, 25, 30000 / 1001, 60]) {
      for (const f of [0, 1, 29, 1799]) {
        expect(frameAtTime(timeOfFrame(f, fps), fps)).toBe(f);
        expect(frameAtTime(f / fps, fps)).toBe(f); // presentation timestamps
      }
    }
  });
});

describe("regionsAtFrame", () => {
  const tracks = indexTracks([track(1, 0.9, 5, [0, 1]), track(2, 0.05, 5, [1, 2])]);
  const manual = [{ region_id: "a", start_frame: 2, end_frame: 4, box: [0, 0, 5, 5] as const }];

  it("reports tracks and manual regions present on a frame", () => {
    const at1 = regionsAtFrame(1, tracks, policy, {}, []);
    expect(at1.map((r) => [r.trackId, r.blurred])).toEqual([
      [1, true],
      [2, false],
    ]);
    const at2 = regionsAtFrame(2, tracks, policy, { 2: "blur" }, [
      { ...manual[0]!, box: [0, 0, 5, 5] },
    ]);
    expect(at2.map((r) => [r.kind, r.blurred])).toEqual([
      ["track", true],
      ["manual", true],
    ]);
  });

  it("summarises decisions", () => {
    expect(summarise(tracks, policy, {})).toEqual({
      minor: 1,
      adult: 1,
      uncertain: 0,
      blurred: 1,
      kept: 1,
    });
  });
});

describe("normaliseRect", () => {
  it("orders corners and clamps to the frame", () => {
    expect(normaliseRect([120, 90], [-5, 10], 100, 80)).toEqual([0, 10, 100, 80]);
  });
  it("ignores tiny accidental drags", () => {
    expect(normaliseRect([10, 10], [12, 11], 100, 100)).toBeNull();
  });
});
