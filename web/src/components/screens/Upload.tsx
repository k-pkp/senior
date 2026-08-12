"use client";

import { useState } from "react";
import { Button, Caveat, Label, Panel } from "@/components/ui/primitives";

export function Upload({
  onStart,
  onBack,
  backendUp,
}: {
  onStart: () => void;
  onBack: () => void;
  backendUp: boolean;
}) {
  const [files, setFiles] = useState<File[]>([]);
  const ok = files.length >= 6 && files.length <= 12;

  return (
    <div className="fadein" style={{ display: "grid", gap: 16, maxWidth: 680 }}>
      <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
        <Button variant="ghost" onClick={onBack}>← Back</Button>
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
          <li>6–9 photos, orbiting around the object</li>
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
            cursor: "pointer",
          }}
        >
          <input
            type="file"
            multiple
            accept="image/*,.heic,.HEIC"
            style={{ display: "none" }}
            onChange={(e) => setFiles(Array.from(e.target.files ?? []))}
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
            {files.length && !ok ? "6–12 photos expected" : "JPG, PNG or HEIC"}
          </div>
        </label>
      </Panel>

      {!backendUp && (
        <Caveat>
          The compute service is not reachable right now. Processing runs on a
          single local GPU — browse the precomputed sample results instead, they
          need no server.
        </Caveat>
      )}

      <Button
        variant="primary"
        onClick={onStart}
        disabled={!ok || !backendUp}
        style={{ padding: 13 }}
      >
        Start processing
      </Button>
    </div>
  );
}
