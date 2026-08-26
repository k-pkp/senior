"use client";

import { useEffect, useState } from "react";
import { Button, Caveat, Label, Panel } from "@/components/ui/primitives";
import { getJob } from "@/lib/api";
import type { FramingReport } from "@/lib/types";

const POLL_MS = 1000;

/** Uploads are stored with a numeric prefix so the order the user picked
 *  survives — "re-take photo 3" has to mean their third one. That prefix is
 *  bookkeeping, not part of the name they recognise, so hide it here. */
function displayName(source: string) {
  return source.replace(/^\d+_/, "");
}

/** Colour for a severity tier. Rejection is not one thing: a frame missing its
 *  marker band still reconstructs fine and only costs the cut, while a frame
 *  whose cube is clipped corrupts the scale of every number the run reports. */
const SEVERITY: Record<string, { fg: string; bg: string }> = {
  "not crucial": { fg: "#7a5a00", bg: "#fdf0d0" },
  crucial: { fg: "#8a3f14", bg: "#fadfd0" },
  "very crucial": { fg: "#8a1f14", bg: "#fadbd6" },
};

/** Plain English for stage 0's `mode`.
 *
 * Accepted is not one thing. A frame stage 0 could frame itself keeps its whole
 * subject; a frame that only survived VGGT's centre crop lost whatever that
 * crop removed. Both count as usable, and the difference is worth seeing. */
function describeMode(mode?: string): { text: string; ideal: boolean } | null {
  if (!mode) return null;
  if (mode.includes("uncropped") || mode === "original")
    return { text: "passed through — the model crops it", ideal: false };
  if (mode === "crop-clipped")
    return { text: "cropped, centred on the cube", ideal: true };
  if (mode === "crop") return { text: "cropped to fit", ideal: true };
  if (mode === "unbounded") return { text: "cube bounds not recoverable", ideal: false };
  return { text: mode, ideal: false };
}

/** Stage 0's verdict on a photo set, shown per frame.
 *
 * The pipeline stops before VGGT if any submitted frame cannot be cropped to
 * hold the whole reference cube and the marker band. That is deliberately
 * strict: a clipped cube silently corrupts the scale for every number the run
 * goes on to report. Being strict is only useful if the person can see WHY, so
 * each frame is shown with the boxes stage 0 actually measured — the crop
 * window, the cube, the band — rather than a bare filename.
 */
export function Framing({
  jobId,
  reportUrl,
  baseUrl,
  onBack,
  onContinue,
}: {
  /** A live run, whose report does not exist yet — stage 0 has to detect the
   *  cube, the limb and the band in every photo first. Null for a shipped
   *  sample, whose report is a static file. */
  jobId: string | null;
  reportUrl: string;
  baseUrl: string;
  onBack: () => void;
  /** `strict` false means the user has seen the rejected photos and chosen to
   *  measure anyway. The pipeline still runs; VGGT just does its own centre
   *  crop on the frames stage 0 could not frame. */
  onContinue?: (strict: boolean) => void | Promise<void>;
}) {
  const [report, setReport] = useState<FramingReport | null>(null);
  const [failed, setFailed] = useState(false);
  const [crashed, setCrashed] = useState<string | null>(null);
  const [zoom, setZoom] = useState<string | null>(null);
  // Stage 0 writes framing.json before its process exits — it still has models
  // to release and a summary to write. So the report can be readable while the
  // job is still `prep`, and /run would refuse. Gate the actions on the job's
  // own state rather than on the report existing.
  const [state, setState] = useState<string | null>(null);
  const [refused, setRefused] = useState<string | null>(null);
  const [sending, setSending] = useState(false);

  // A shipped sample: the report is a file that either exists or does not.
  useEffect(() => {
    if (jobId) return;
    let live = true;
    fetch(reportUrl)
      .then((r) => (r.ok ? r.json() : Promise.reject(new Error(String(r.status)))))
      .then((d: FramingReport) => live && setReport(d))
      .catch(() => live && setFailed(true));
    return () => {
      live = false;
    };
  }, [jobId, reportUrl]);

  // A live run: stage 0 takes about half a minute — it runs an open-vocabulary
  // detector and a segmenter over every photo — so the report has to be waited
  // for. Treating its absence as failure, which is right for a sample, would
  // reject every upload the instant it arrived.
  useEffect(() => {
    if (!jobId) return;
    let live = true;

    const tick = async () => {
      try {
        const j = await getJob(jobId);
        if (!live) return;
        if (j.framing) setReport(j.framing);
        setState(j.state);
        if (j.state === "failed") setCrashed(j.error ?? "stage 0 failed");
      } catch {
        /* transient; the next tick retries */
      }
    };

    tick();
    const id = setInterval(tick, POLL_MS);
    return () => {
      live = false;
      clearInterval(id);
    };
  }, [jobId]);

  if (crashed) {
    return (
      <div className="fadein" style={{ display: "grid", gap: 16, maxWidth: 720 }}>
        <Button variant="ghost" onClick={onBack}>← Back</Button>
        <Panel>
          <Label>Framing check failed</Label>
          <div style={{ font: "400 13px/1.6 var(--sans)", marginTop: 8 }}>
            {crashed}
          </div>
        </Panel>
      </div>
    );
  }

  if (jobId && !report) {
    return (
      <div className="fadein" style={{ display: "grid", gap: 16, maxWidth: 720 }}>
        <Panel>
          <Label>Checking framing</Label>
          <div style={{ font: "400 13px/1.7 var(--sans)", marginTop: 8 }}>
            Locating the reference cube, the limb and the marker band in each
            photo, then working out whether one crop can hold them all.
          </div>
          <div
            style={{
              font: "400 11.5px/1.5 var(--mono)",
              color: "var(--muted)",
              marginTop: 8,
            }}
          >
            about 30 seconds
          </div>
        </Panel>
      </div>
    );
  }

  if (failed) {
    return (
      <div className="fadein" style={{ display: "grid", gap: 16, maxWidth: 720 }}>
        <Button variant="ghost" onClick={onBack}>← Back</Button>
        <Panel>
          <Label>Framing check</Label>
          <Caveat>No framing report for this run. It was produced before
            stage 0 existed, or stage 0 was skipped with --no-prep.</Caveat>
        </Panel>
      </div>
    );
  }
  if (!report) return <div className="fadein" style={{ color: "var(--muted)" }}>Loading…</div>;

  // A warning is a frame the pipeline USES. Only rejects need re-taking, so
  // only rejects go in the "Re-take" line — listing warnings there would send
  // someone back out to re-shoot photos that are already fine.
  const rejected = report.frames.filter(
    (f) => (f.verdict ? f.verdict === "reject" : !f.accepted),
  );
  const warned = report.frames.filter((f) => f.verdict === "warning");
  const shortfall = report.accepted < report.required;
  // A sample has no job to be ready; a live one must have finished stage 0.
  const ready = !jobId || state === "awaiting-framing";

  async function go(strict: boolean) {
    if (!onContinue) return;
    setSending(true);
    setRefused(null);
    try {
      await onContinue(strict);
    } catch (e) {
      // The service refuses for reasons a person can act on — say them. A
      // click that silently does nothing is the worst possible answer.
      setRefused(e instanceof Error ? e.message : String(e));
      setSending(false);
    }
  }

  return (
    <div className="fadein" style={{ display: "grid", gap: 16, maxWidth: 980 }}>
      <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
        <Button variant="ghost" onClick={onBack}>← Back</Button>
        <div style={{ font: "500 15px/1 var(--sans)" }}>Framing check</div>
      </div>

      <Panel>
        <Label>
          {rejected.length > 0
            ? "Not accepted"
            : warned.length > 0
              ? "Usable, with warnings"
              : "Accepted"}
        </Label>
        <div style={{ font: "400 13px/1.7 var(--sans)", marginTop: 8 }}>
          {report.accepted} of {report.submitted} photos usable;{" "}
          {report.required} required. Warnings are used; only rejects are not.
        </div>
        {warned.length > 0 && rejected.length === 0 && (
          <div style={{ font: "400 13px/1.7 var(--sans)", color: "var(--warn, #b8860b)" }}>
            {warned.length} photo(s) carry a warning and are still measured —
            most often no marker band, which means you place the cut yourself in
            the next step.
          </div>
        )}
        {rejected.length > 0 && (
          <div style={{ font: "500 13px/1.7 var(--sans)", color: "var(--bad, #c0392b)" }}>
            Re-take:{" "}
            {rejected.map((f) => `img${f.index} (${displayName(f.source)})`).join(", ")}
          </div>
        )}
        {shortfall && rejected.length === 0 && (
          <div style={{ font: "500 13px/1.7 var(--sans)", color: "var(--bad, #c0392b)" }}>
            All photos passed, but only {report.accepted} were supplied — add{" "}
            {report.required - report.accepted} more view(s).
          </div>
        )}
        <Caveat>
          Four boxes: yellow is the window stage 0 chose, magenta the
          reference cube, orange the limb, green the marker band. Every photo
          gets one of three verdicts. It <b>passes</b> when the window holds
          everything. It carries a <b>warning</b> when something is missing or
          clipped but the photo is still measurable — no marker band means you
          place the cut yourself in the next step, and a clipped cube falls back
          to the model&apos;s own centre crop. It is <b>rejected</b> only when the
          reference cube was not found at all, or the file could not be read,
          because the cube sets the scale of every number and nothing downstream
          can recover it. Only rejects need re-taking.
        </Caveat>
      </Panel>

      <div
        style={{
          display: "grid",
          gap: 12,
          gridTemplateColumns: "repeat(auto-fill, minmax(240px, 1fr))",
        }}
      >
        {report.frames.map((f) => (
          <Panel key={f.index} pad={0}>
            {f.overlay ? (
              <button
                onClick={() => setZoom(`${baseUrl}/${f.overlay}`)}
                style={{
                  display: "block", width: "100%", border: 0, padding: 0,
                  background: "transparent", cursor: "zoom-in",
                }}
              >
                {/* eslint-disable-next-line @next/next/no-img-element */}
                <img
                  src={`${baseUrl}/${f.overlay}`}
                  alt={`${f.source} framing`}
                  style={{
                    width: "100%", display: "block",
                    borderTopLeftRadius: "var(--radius)",
                    borderTopRightRadius: "var(--radius)",
                  }}
                />
              </button>
            ) : (
              // No overlay means the file could not be decoded, so there is no
              // frame to draw boxes on. Say that, rather than rendering a broken
              // image element.
              <div
                style={{
                  aspectRatio: "3 / 4",
                  display: "grid",
                  placeItems: "center",
                  background: "var(--soft)",
                  borderTopLeftRadius: "var(--radius)",
                  borderTopRightRadius: "var(--radius)",
                  font: "400 12px/1.5 var(--sans)",
                  color: "var(--muted)",
                  textAlign: "center",
                  padding: 16,
                }}
              >
                this file could not be opened
              </div>
            )}
            <div style={{ padding: "10px 12px 12px" }}>
              <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
                <span
                  style={{
                    font: "600 11px/1 var(--sans)",
                    padding: "3px 7px",
                    borderRadius: 999,
                    color: f.accepted ? "#0b6b3a" : "#8a1f14",
                    background: f.accepted ? "#d8f2e3" : "#fadbd6",
                  }}
                >
                  {f.accepted ? "ACCEPTED" : "REJECTED"}
                </span>
                <span style={{ font: "500 12px/1 var(--sans)" }}>
                  img{f.index}
                </span>
              </div>
              <div style={{ font: "400 11px/1.5 var(--sans)", color: "var(--muted)", marginTop: 6 }}>
                {displayName(f.source)}
              </div>
              {f.accepted && describeMode(f.mode) && (
                <div
                  style={{
                    font: "400 11.5px/1.5 var(--sans)",
                    marginTop: 6,
                    color: describeMode(f.mode)!.ideal
                      ? "var(--muted)"
                      : "var(--warn, #8a5a00)",
                  }}
                >
                  {describeMode(f.mode)!.text}
                </div>
              )}
              {f.reasons.length > 0 && (
                <div style={{ marginTop: 6 }}>
                  {f.severity && (
                    <span
                      style={{
                        font: "600 10.5px/1 var(--sans)",
                        padding: "3px 7px",
                        borderRadius: 999,
                        marginRight: 6,
                        color: SEVERITY[f.severity]?.fg ?? "var(--muted)",
                        background: SEVERITY[f.severity]?.bg ?? "var(--soft)",
                        textTransform: "uppercase",
                        letterSpacing: ".03em",
                      }}
                    >
                      {f.severity}
                    </span>
                  )}
                  <span style={{ font: "400 12px/1.5 var(--sans)" }}>
                    {f.reasons.join("; ")}
                  </span>
                </div>
              )}
            </div>
          </Panel>
        ))}
      </div>

      {refused && (
        <Panel>
          <Label>Not started</Label>
          <div style={{ font: "400 13px/1.6 var(--sans)", marginTop: 8 }}>
            {refused}
          </div>
        </Panel>
      )}

      {onContinue && (
        <div style={{ display: "flex", gap: 10, flexWrap: "wrap", alignItems: "center" }}>
          {report.all_passed ? (
            <Button variant="primary" disabled={!ready || sending}
                    onClick={() => go(true)}>
              {ready ? "Continue →" : "Finishing the framing check…"}
            </Button>
          ) : (
            <>
              <Button variant="primary" onClick={onBack} disabled={sending}>
                Re-take and upload again
              </Button>
              <Button disabled={!ready || sending} onClick={() => go(false)}>
                {ready
                  ? `Measure anyway with ${report.accepted} of ${report.submitted}`
                  : "Finishing the framing check…"}
              </Button>
              <span
                style={{
                  font: "400 11.5px/1.5 var(--sans)",
                  color: "var(--muted)",
                  flexBasis: "100%",
                }}
              >
                Measuring anyway is not the same as passing. The rejected photos
                still go to the model, cropped by the model rather than by us,
                and whatever the crop cut off is lost to the reconstruction.
              </span>
            </>
          )}
        </div>
      )}

      {zoom && (
        <div
          onClick={() => setZoom(null)}
          style={{
            position: "fixed", inset: 0, background: "rgba(0,0,0,.82)",
            display: "grid", placeItems: "center", zIndex: 50, cursor: "zoom-out",
            padding: 24,
          }}
        >
          {/* eslint-disable-next-line @next/next/no-img-element */}
          <img src={zoom} alt="framing detail"
               style={{ maxWidth: "100%", maxHeight: "100%" }} />
        </div>
      )}
    </div>
  );
}
