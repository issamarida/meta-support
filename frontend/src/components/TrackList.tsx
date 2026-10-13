import { memo } from "react";
import type { Override } from "../api/types";
import { classify, formatTime, shouldBlur, type IndexedTrack, type Policy } from "../review/model";

interface Props {
  thumbnailUrl: (id: number) => string;
  fps: number;
  tracks: IndexedTrack[];
  policy: Policy;
  overrides: Record<string, Override>;
  selected: number | null;
  reveal: boolean;
  onSelect: (id: number) => void;
  onJump: (frame: number) => void;
  onOverride: (id: number, value: Override | null) => void;
}

const STATUS_TEXT = {
  minor: "Likely under 18",
  uncertain: "Uncertain",
  adult: "Likely adult",
} as const;

export const TrackList = memo(function TrackList(props: Props) {
  const { tracks, policy, overrides, fps } = props;
  if (tracks.length === 0) {
    return (
      <p className="muted small">
        No people were detected. If someone was missed, add a manual region.
      </p>
    );
  }
  return (
    <ul className="tracks">
      {tracks.map((t) => {
        const status = classify(t, policy);
        const override = overrides[t.track_id];
        const blurred = shouldBlur(t, policy, override);
        return (
          <li
            key={t.track_id}
            className={`track${props.selected === t.track_id ? " selected" : ""}${blurred ? "" : " is-kept"}`}
            onClick={() => props.onSelect(t.track_id)}
          >
            <img
              className={`thumb${props.reveal ? "" : " veiled"}`}
              src={props.thumbnailUrl(t.track_id)}
              alt={`Person #${t.track_id}`}
              loading="lazy"
              onError={(e) => (e.currentTarget.style.visibility = "hidden")}
            />
            <div className="track-body">
              <div className="track-head">
                <strong>Person #{t.track_id}</strong>
                <span className={`badge ${status}`}>{STATUS_TEXT[status]}</span>
              </div>
              <div className="meter" title="Estimated probability of being under 18">
                <div
                  className="meter-fill"
                  style={{ width: `${(t.p_minor ?? 0) * 100}%` }}
                  data-empty={t.p_minor === null}
                />
                <div className="meter-threshold" style={{ left: `${policy.threshold * 100}%` }} />
              </div>
              <div className="track-meta muted small">
                {t.p_minor === null ? "no usable face" : `P(<18) ${(t.p_minor * 100).toFixed(0)}%`}
                {t.expected_age !== null && <> · ~{Math.round(t.expected_age)} y</>} ·{" "}
                {t.face_samples} face{t.face_samples === 1 ? "" : "s"} ·{" "}
                <button
                  className="link"
                  onClick={(e) => {
                    e.stopPropagation();
                    props.onJump(t.first_frame);
                  }}
                >
                  {formatTime(t.first_frame / fps)}–{formatTime(t.last_frame / fps)}
                </button>
              </div>
              <div className="segmented" role="radiogroup" aria-label={`Decision for person #${t.track_id}`}>
                {(
                  [
                    [null, `Auto · ${status === "adult" ? "keep" : "blur"}`],
                    ["blur", "Blur"],
                    ["keep", "Keep"],
                  ] as const
                ).map(([value, label]) => (
                  <button
                    key={label}
                    role="radio"
                    aria-checked={(override ?? null) === value}
                    className={(override ?? null) === value ? "on" : ""}
                    onClick={(e) => {
                      e.stopPropagation();
                      props.onOverride(t.track_id, value);
                    }}
                  >
                    {label}
                  </button>
                ))}
              </div>
            </div>
          </li>
        );
      })}
    </ul>
  );
});
