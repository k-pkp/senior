"use client";

import { useEffect, useRef, useState } from "react";
import { Button, Label, Panel } from "@/components/ui/primitives";
import { getJob, type JobStatus } from "@/lib/api";
import { CUT_STAGES, MEASURE_STAGES } from "@/lib/data";

const POLL_MS = 1000;

/** Stage-by-stage progress, driven by the service.
 *
 * The service launches each stage as its own subprocess, so `stage` in the
 * status is observed rather than estimated — a tick here means that stage
 * really finished. The seconds beside each row stay as expectations, which is
 * what they always were; they no longer decide anything.
 *
 * This screen is also the only place a pipeline crash can surface. Left to a
 * timer it would look like a successful run that produced nothing, so a failed
 * job shows the stage that died and the tail of its output. */
export function Processing({
  jobId,
  phase,
  onDone,
  onBack,
}: {
  jobId: string | null;
  /** Which pass is running. They share stage numbers but not meaning: the
   *  first measures the reference and stops before the cut, the second applies
   *  the confirmed cut and measures the object. */
  phase: "measure" | "cut";
  onDone: () => void;
  onBack: () => void;
}) {
  const stages = phase === "cut" ? CUT_STAGES : MEASURE_STAGES;
  const first = stages[0].n;
  const last = stages[stages.length - 1].n;
  const [job, setJob] = useState<JobStatus | null>(null);
  const [unreachable, setUnreachable] = useState(false);
  const done = useRef(false);

  useEffect(() => {
    if (!jobId) return;
    let live = true;

    // Polls the job while stages run, advancing the progress display until it finishes or fails.
    const tick = async () => {
      try {
        const j = await getJob(jobId);
        if (!live) return;
        setJob(j);
        setUnreachable(false);
        // awaiting-cut means the measuring pass finished — it just did not
        // apply the cut, which is the review's job. Both are "the stages are
        // over", so both hand off.
        if ((j.state === "done" || j.state === "awaiting-cut") && !done.current) {
          done.current = true;
          setTimeout(onDone, 400);
        }
      } catch {
        if (live) setUnreachable(true);
      }
    };

    tick();
    const id = setInterval(tick, POLL_MS);
    return () => {
      live = false;
      clearInterval(id);
    };
  }, [jobId, onDone]);

  const failed = job?.state === "failed";
  // `stage` is the one in flight; everything before it has finished. A job
  // still queued behind another has not started stage 1 at all.
  const current =
    job == null || job.state === "queued"
      ? first - 1
      : job.state === "done" || job.state === "awaiting-cut"
        ? last + 1
        : job.stage;

  return (
    <div className="fadein" style={{ display: "grid", gap: 16, maxWidth: 640 }}>
      <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
        <div style={{ font: "500 15px/1 var(--sans)" }}>
          {phase === "cut" ? "Applying your cut" : "Reconstructing"}
        </div>
        {job?.state === "queued" && job.queue > 0 && (
          <span style={{ font: "400 12px/1 var(--mono)", color: "var(--muted)" }}>
            queued behind {job.queue}
          </span>
        )}
      </div>

      <Panel>
        {stages.map((s, i) => {
          const finished =
            s.n < current || job?.state === "done" || job?.state === "awaiting-cut";
          const running = s.n === current && !failed;
          const broke = failed && s.n === job?.stage;
          return (
            <div
              key={s.n}
              style={{
                display: "flex",
                alignItems: "center",
                gap: 12,
                padding: "11px 0",
                borderBottom:
                  i < stages.length - 1 ? "1px solid var(--line)" : "none",
                opacity: finished || running || broke ? 1 : 0.4,
              }}
            >
              <div
                style={{
                  width: 20,
                  height: 20,
                  borderRadius: 5,
                  flexShrink: 0,
                  display: "grid",
                  placeItems: "center",
                  background: broke
                    ? "#8a1f14"
                    : finished
                      ? "var(--accent)"
                      : "var(--soft)",
                  color: broke
                    ? "#fff"
                    : finished
                      ? "var(--accent-ink)"
                      : "var(--muted)",
                  font: "500 11px/1 var(--mono)",
                  animation: running ? "blip 1.1s ease-in-out infinite" : undefined,
                }}
              >
                {broke ? "!" : finished ? "✓" : s.n}
              </div>
              <div style={{ flex: 1, minWidth: 0 }}>
                <div style={{ font: "500 13px/1.3 var(--sans)" }}>{s.label}</div>
                <div
                  style={{
                    font: "400 11.5px/1.4 var(--mono)",
                    color: "var(--muted)",
                    marginTop: 2,
                  }}
                >
                  {s.out}
                </div>
              </div>
              <div style={{ font: "400 11.5px/1 var(--mono)", color: "var(--muted)" }}>
                ~{s.seconds}s
              </div>
            </div>
          );
        })}
      </Panel>

      {failed && (
        <Panel>
          <Label>Stage {job?.stage} failed</Label>
          <div style={{ font: "400 13px/1.6 var(--sans)", margin: "8px 0 10px" }}>
            {job?.error}
          </div>
          {job?.log?.length ? (
            <pre
              style={{
                font: "400 11px/1.5 var(--mono)",
                color: "var(--muted)",
                background: "var(--soft)",
                borderRadius: 6,
                padding: 10,
                margin: 0,
                maxHeight: 240,
                overflow: "auto",
                whiteSpace: "pre-wrap",
              }}
            >
              {job.log.join("\n")}
            </pre>
          ) : null}
          <div style={{ marginTop: 12 }}>
            <Button onClick={onBack}>← Back</Button>
          </div>
        </Panel>
      )}

      {unreachable && !failed && (
        <Panel>
          <Label>Lost contact</Label>
          <div style={{ font: "400 13px/1.6 var(--sans)", marginTop: 8 }}>
            The compute service stopped answering. The run may still be going —
            this page will pick it up again if it comes back.
          </div>
        </Panel>
      )}

      {!failed && (
        <div style={{ font: "400 11.5px/1.6 var(--sans)", color: "var(--muted)" }}>
          {phase === "cut"
            ? "Stages 1 and 2 are not repeated — the reconstruction does not depend on where the cut goes."
            : "Runs on a single local GPU, so uploads are handled one at a time. Stage 1 dominates; the rest is geometry. The object itself is not measured yet: its extent depends on the cut you are about to confirm."}
        </div>
      )}
    </div>
  );
}
