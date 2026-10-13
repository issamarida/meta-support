import { useEffect, useState } from "react";
import { api, watchJob } from "../api/client";
import type { Job } from "../api/types";
import { ProgressBar } from "./ProgressBar";

interface Props {
  job: Job;
  onUpdate: (job: Job) => void;
  onBackToReview: () => void;
  onDelete: () => void;
  onStartOver: () => void;
}

export function ExportScreen({ job, onUpdate, onBackToReview, onDelete, onStartOver }: Props) {
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (job.state !== "rendering") return;
    return watchJob(job.id, onUpdate, (e) => setError(e.message));
  }, [job.id, job.state, onUpdate]);

  if (job.state === "rendering") {
    return (
      <section className="card processing">
        <h2>Exporting “{job.filename}”</h2>
        <p className="muted">
          Blurring and greyscaling flagged regions, then encoding at the original resolution and
          frame rate with the original audio.
        </p>
        <ProgressBar value={job.progress} label={job.stage} />
        {error && <p className="error">{error}</p>}
        <button className="btn" onClick={onDelete}>
          Cancel and delete
        </button>
      </section>
    );
  }

  const expires = new Date(job.expires_at * 1000);
  return (
    <section className="card done">
      <div className="done-icon" aria-hidden>
        ✓
      </div>
      <h2>Your blurred video is ready</h2>
      <a className="btn primary big" href={api.downloadUrl(job.id)} download>
        Download MP4
      </a>
      <ul className="facts small">
        <li>Original resolution, frame rate and audio preserved.</li>
        <li>Location and device metadata removed.</li>
        <li>
          {job.upload_deleted
            ? "Your original upload, the preview and thumbnails have been deleted from the server."
            : "Your original upload is being kept, as you asked."}
        </li>
        <li>
          Everything left for this job is deleted automatically after{" "}
          {expires.toLocaleString(undefined, { dateStyle: "medium", timeStyle: "short" })}.
        </li>
      </ul>
      <p className="warn small">
        Please watch the result before sharing it: age estimation can be wrong.
      </p>
      <div className="row">
        {!job.upload_deleted && (
          <button className="btn" onClick={onBackToReview}>
            Back to review
          </button>
        )}
        <button className="btn danger" onClick={onDelete}>
          Delete everything now
        </button>
        <button className="btn" onClick={onStartOver}>
          Process another video
        </button>
      </div>
    </section>
  );
}
