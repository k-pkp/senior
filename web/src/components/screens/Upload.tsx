"use client";

import { useState } from "react";
import { createJob } from "@/lib/api";
import { Button, Caveat, Label, Panel } from "@/components/ui/primitives";

const MIN_FILES = 6;
const MAX_FILES = 12;

export function Upload({
  onStart,
  onBack,
  backendUp,
}: {
  /** Called with the job id once the service has the photos and stage 0 is
   *  queued. Everything after this point is driven by polling that job. */
  onStart: (jobId: string, frames: number) => void;
  onBack: () => void;
  backendUp: boolean;
}) {
  const [files, setFiles] = useState<File[]>([]);
  const [sending, setSending] = useState(false);
  const [progress, setProgress] = useState(0);
  const [error, setError] = useState<string | null>(null);
  const ok = files.length >= MIN_FILES && files.length <= MAX_FILES;

  async function send() {
    setSending(true);
    setError(null);
    setProgress(0);
    try {
      const { job_id, frames } = await createJob(files, setProgress);
      onStart(job_id, frames);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
      setSending(false);
    }
  }

  return (
    <div className="fadein" style={{ display: "grid", gap: 16, maxWidth: 680 }}>
      <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
        <Button variant="ghost" onClick={onBack} disabled={sending}>← Back</Button>
        <div style={{ font: "500 15px/1 var(--sans)" }}>Upload a photo set</div>
      </div>

      <Panel>
        <Label>Requirements</Label>
        <ul
          style={{
            font: "400 13px/1.7 var(--sans)",
            color: "var(--muted)",
            margin: "10px 0 0",
            paddingLeft: 18,
          }}
        >
          <li>{MIN_FILES}–{MAX_FILES} photos, orbiting around the object</li>
          <li>The reference cube visible in every shot</li>
          <li>A coloured band on the limb where the measurement should stop</li>
          <li>Even lighting; avoid strong shadows under the object</li>
        </ul>
      </Panel>

      <Panel pad={0}>
        <label
          style={{
            display: "block",
            padding: 34,
            textAlign: "center",
            border: "1px dashed var(--line)",
            borderRadius: "var(--radius)",
            cursor: sending ? "default" : "pointer",
            opacity: sending ? 0.5 : 1,
          }}
        >
          <input
            type="file"
            multiple
            disabled={sending}
            accept="image/*,.heic,.HEIC"
            style={{ display: "none" }}
            onChange={(e) => {
              setFiles(Array.from(e.target.files ?? []));
              setError(null);
            }}
          />
          <div style={{ font: "500 14px/1.4 var(--sans)" }}>
            {files.length ? `${files.length} photos selected` : "Choose photos"}
          </div>
          <div
            style={{
              font: "400 12px/1.5 var(--sans)",
              color: "var(--muted)",
              marginTop: 6,
            }}
          >
            {files.length && !ok
              ? `${MIN_FILES}–${MAX_FILES} photos expected`
              : "JPG, PNG or HEIC"}
          </div>
        </label>
      </Panel>

      {sending && (
        <Panel>
          <Label>Uploading</Label>
          <div
            style={{
              height: 6,
              borderRadius: 999,
              background: "var(--soft)",
              overflow: "hidden",
              marginTop: 10,
            }}
          >
            <div
              style={{
                height: "100%",
                width: `${Math.round(progress * 100)}%`,
                background: "var(--accent)",
                transition: "width .2s linear",
              }}
            />
          </div>
          <div
            style={{
              font: "400 11.5px/1.5 var(--mono)",
              color: "var(--muted)",
              marginTop: 8,
            }}
          >
            {Math.round(progress * 100)}% — full-resolution photos, so this takes
            a moment over wifi.
          </div>
        </Panel>
      )}

      {error && (
        <Panel>
          <Label>Not accepted</Label>
          <div style={{ font: "400 13px/1.6 var(--sans)", marginTop: 8 }}>
            {error}
          </div>
        </Panel>
      )}

      {!backendUp && (
        <Caveat>
          The compute service is not reachable right now. Processing runs on a
          single local GPU — browse the precomputed sample results instead, they
          need no server.
        </Caveat>
      )}

      <Button
        variant="primary"
        onClick={send}
        disabled={!ok || !backendUp || sending}
        style={{ padding: 13 }}
      >
        {sending ? "Uploading…" : "Start processing"}
      </Button>
    </div>
  );
}
