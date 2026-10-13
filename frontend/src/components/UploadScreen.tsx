import { useRef, useState } from "react";
import { uploadVideo } from "../api/client";
import type { AppConfig, Job } from "../api/types";
import { ProgressBar } from "./ProgressBar";

interface Props {
  config: AppConfig;
  onUploaded: (job: Job) => void;
}

export function UploadScreen({ config, onUploaded }: Props) {
  const [file, setFile] = useState<File | null>(null);
  const [keep, setKeep] = useState(false);
  const [dragging, setDragging] = useState(false);
  const [progress, setProgress] = useState<number | null>(null);
  const [error, setError] = useState<string | null>(null);
  const abortRef = useRef<(() => void) | null>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  const choose = (f: File | undefined) => {
    setError(null);
    if (!f) return;
    if (f.size > config.max_upload_mb * 1024 * 1024) {
      setError(`That file is larger than the ${config.max_upload_mb} MB limit.`);
      return;
    }
    setFile(f);
  };

  const start = async () => {
    if (!file) return;
    setError(null);
    setProgress(0);
    const { promise, abort } = uploadVideo(file, keep, setProgress);
    abortRef.current = abort;
    try {
      onUploaded(await promise);
    } catch (e) {
      setError((e as Error).message);
      setProgress(null);
    } finally {
      abortRef.current = null;
    }
  };

  const uploading = progress !== null;

  return (
    <section className="card upload">
      <h1>Blur children in a video before you share it</h1>
      <p className="lead">
        SafeFrame finds every person, estimates who is likely under 18 and blurs them in every
        frame they appear. Adults and the background stay untouched. You review every decision
        before anything is exported.
      </p>

      <div
        className={`dropzone${dragging ? " dragging" : ""}${file ? " has-file" : ""}`}
        onDragOver={(e) => {
          e.preventDefault();
          setDragging(true);
        }}
        onDragLeave={() => setDragging(false)}
        onDrop={(e) => {
          e.preventDefault();
          setDragging(false);
          choose(e.dataTransfer.files[0]);
        }}
        onClick={() => !uploading && inputRef.current?.click()}
        role="button"
        tabIndex={0}
        onKeyDown={(e) => (e.key === "Enter" || e.key === " ") && inputRef.current?.click()}
        aria-label="Choose a video file"
      >
        <input
          ref={inputRef}
          type="file"
          accept="video/*"
          hidden
          onChange={(e) => choose(e.target.files?.[0])}
        />
        {file ? (
          <>
            <strong>{file.name}</strong>
            <span className="muted">{(file.size / 1024 / 1024).toFixed(1)} MB · click to change</span>
          </>
        ) : (
          <>
            <strong>Drop a video here, or click to choose</strong>
            <span className="muted">MP4, MOV, MKV, WebM… up to {config.max_upload_mb} MB</span>
          </>
        )}
      </div>

      <label className="check">
        <input type="checkbox" checked={keep} onChange={(e) => setKeep(e.target.checked)} />
        <span>
          Keep my original upload after export
          <small className="muted">
            {keep
              ? ` It stays on the server so you can re-export, until you delete it or after ${config.job_ttl_hours} h of inactivity.`
              : " By default the original, the preview and all thumbnails are deleted as soon as the blurred video is ready."}
          </small>
        </span>
      </label>

      {error && (
        <p className="error" role="alert">
          {error}
        </p>
      )}

      {uploading ? (
        <div className="stack">
          <ProgressBar value={progress} label="Uploading" />
          <button className="btn" onClick={() => abortRef.current?.()}>
            Cancel upload
          </button>
        </div>
      ) : (
        <button className="btn primary big" disabled={!file} onClick={start}>
          Upload and analyse
        </button>
      )}

      <details className="how">
        <summary>How it works</summary>
        <ol>
          <li>People are detected in each frame (YOLO) and followed with a tracker, so each person keeps one identity through head turns and brief occlusions.</li>
          <li>Whenever a face is visible, its apparent age is estimated; evidence accumulates per person.</li>
          <li>Likely minors — and anyone without enough evidence — are marked for blurring.</li>
          <li>You review the result, adjust sensitivity, override decisions or draw extra regions.</li>
          <li>Each marked region gets a strong Gaussian blur and is converted to greyscale. Resolution, frame rate and audio are preserved; location metadata is removed.</li>
        </ol>
      </details>
    </section>
  );
}
