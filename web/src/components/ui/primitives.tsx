"use client";

import type { CSSProperties, ReactNode } from "react";

export const panel: CSSProperties = {
  background: "var(--panel)",
  border: "1px solid var(--line)",
  borderRadius: "var(--radius)",
};

export function Panel({
  children,
  style,
  pad = 18,
}: {
  children: ReactNode;
  style?: CSSProperties;
  pad?: number;
}) {
  return <div style={{ ...panel, padding: pad, ...style }}>{children}</div>;
}

export function Button({
  children,
  onClick,
  variant = "ghost",
  disabled,
  style,
  title,
}: {
  children: ReactNode;
  onClick?: () => void;
  variant?: "primary" | "ghost" | "quiet";
  disabled?: boolean;
  style?: CSSProperties;
  title?: string;
}) {
  const base: CSSProperties = {
    borderRadius: 7,
    padding: "9px 15px",
    font: "500 13px/1 var(--sans)",
    transition: "opacity .15s ease, background .15s ease",
    opacity: disabled ? 0.45 : 1,
    pointerEvents: disabled ? "none" : "auto",
  };
  const variants: Record<string, CSSProperties> = {
    primary: {
      border: 0,
      background: "var(--accent)",
      color: "var(--accent-ink)",
    },
    ghost: {
      border: "1px solid var(--line)",
      background: "transparent",
      color: "var(--ink)",
    },
    quiet: {
      border: 0,
      background: "transparent",
      color: "var(--muted)",
    },
  };
  return (
    <button
      onClick={onClick}
      disabled={disabled}
      title={title}
      style={{ ...base, ...variants[variant], ...style }}
    >
      {children}
    </button>
  );
}

export function Label({ children }: { children: ReactNode }) {
  return (
    <div
      style={{
        font: "500 11px/1.3 var(--mono)",
        letterSpacing: ".04em",
        textTransform: "uppercase",
        color: "var(--muted)",
      }}
    >
      {children}
    </div>
  );
}

export function Stat({
  label,
  value,
  unit,
  hint,
  big = false,
}: {
  label: string;
  value: string;
  unit?: string;
  hint?: string;
  big?: boolean;
}) {
  return (
    <div>
      <Label>{label}</Label>
      <div
        style={{
          font: `${big ? 600 : 500} ${big ? 34 : 19}px/1.1 var(--mono)`,
          marginTop: 7,
          color: "var(--ink)",
        }}
      >
        {value}
        {unit && (
          <span
            style={{
              font: "400 13px/1 var(--mono)",
              color: "var(--muted)",
              marginLeft: 5,
            }}
          >
            {unit}
          </span>
        )}
      </div>
      {hint && (
        <div
          style={{
            font: "400 11.5px/1.45 var(--sans)",
            color: "var(--muted)",
            marginTop: 5,
          }}
        >
          {hint}
        </div>
      )}
    </div>
  );
}

/** Caveats are first-class here — see prompt.md §7 and §9. */
export function Caveat({ children }: { children: ReactNode }) {
  return (
    <div
      style={{
        display: "flex",
        gap: 9,
        padding: "10px 12px",
        borderRadius: 8,
        background: "var(--soft)",
        border: "1px solid var(--line)",
        font: "400 12px/1.55 var(--sans)",
        color: "var(--muted)",
      }}
    >
      <span style={{ color: "var(--warn)", flexShrink: 0 }}>▲</span>
      <div>{children}</div>
    </div>
  );
}

export function Slider({
  label,
  value,
  min,
  max,
  step = 1,
  suffix,
  onChange,
}: {
  label: string;
  value: number;
  min: number;
  max: number;
  step?: number;
  suffix?: string;
  onChange: (v: number) => void;
}) {
  return (
    <div style={{ marginBottom: 13 }}>
      <div
        style={{
          display: "flex",
          justifyContent: "space-between",
          alignItems: "baseline",
          marginBottom: 6,
        }}
      >
        <span style={{ font: "400 12px/1 var(--sans)", color: "var(--muted)" }}>
          {label}
        </span>
        <span style={{ font: "500 12px/1 var(--mono)", color: "var(--ink)" }}>
          {value.toFixed(step < 1 ? 2 : 0)}
          {suffix}
        </span>
      </div>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(e) => onChange(parseFloat(e.target.value))}
      />
    </div>
  );
}
