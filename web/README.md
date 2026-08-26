# Understanding the web app

A beginner's walkthrough of this codebase. Written for someone who has never built
a web app. It assumes you can read Python but have never touched JavaScript,
React, or a browser's developer tools.

Read it top to bottom, once. The goal is that "frontend", "backend", "API",
"state", and "component" stop being vague words and become things you can point
at in a file.

---

## Contents

- [Part 0 — How a web app works](#part-0--how-a-web-app-works)
- [Part 1 — React (the language the UI is written in)](#part-1--react-the-language-the-ui-is-written-in)
  - [1.1 JavaScript, quickly](#11-javascript-quickly)
  - [1.2 JSX: markup that is actually code](#12-jsx-markup-that-is-actually-code)
  - [1.3 Components: the UI is built from functions](#13-components-the-ui-is-built-from-functions)
  - [1.4 Props: how components receive data](#14-props-how-components-receive-data)
  - [1.5 State and the render cycle](#15-state-and-the-render-cycle)
  - [1.6 `useState`](#16-usestate)
  - [1.7 `useEffect`](#17-useeffect)
  - [1.8 `useMemo` and `useRef`](#18-usememo-and-useref)
  - [1.9 `"use client"` and `dynamic()`](#19-use-client-and-dynamic)
  - [1.10 Line-by-line: `page.tsx`](#110-line-by-line-pagetsx)
  - [1.11 Line-by-line: `primitives.tsx`](#111-line-by-line-primitivestsx)
- [Part 2 — Three.js (the 3D)](#part-2--threejs-the-3d)
  - [2.1 What three.js is](#21-what-threejs-is)
  - [2.2 Geometry: `BufferGeometry` and `BufferAttribute`](#22-geometry-buffergeometry-and-bufferattribute)
  - [2.3 Materials and meshes](#23-materials-and-meshes)
  - [2.4 `@react-three/fiber`: three.js as JSX](#24-react-threefiber-threejs-as-jsx)
  - [2.5 `@react-three/drei`: ready-made helpers](#25-react-threedrei-ready-made-helpers)
  - [2.6 Line-by-line: `Viewport.tsx`](#26-line-by-line-viewporttsx)
  - [2.7 Line-by-line: `MeshView.tsx`](#27-line-by-line-meshviewtsx)
  - [2.8 Line-by-line: `usePly.ts`](#28-line-by-line-useplyts)
- [Part 3 — The app's glue](#part-3--the-apps-glue)
  - [3.1 The coordinate transform](#31-the-coordinate-transform)
  - [3.2 Line-by-line: `CutReview.tsx`](#32-line-by-line-cutreviewtsx)
  - [3.3 Line-by-line: `api.ts`](#33-line-by-line-apits)
  - [3.4 Line-by-line: `data.ts`](#34-line-by-line-datats)
  - [3.5 The screen state machine, end to end](#35-the-screen-state-machine-end-to-end)
- [Glossary](#glossary)

---

# Part 0 — How a web app works

## The two programs

When you run `./serve.sh`, you start **two separate programs**:

| | Frontend | Backend |
|---|---|---|
| Code | `web/` (this directory) | `service/` |
| Language | TypeScript (JavaScript) | Python |
| Framework | Next.js / React | FastAPI (uvicorn) |
| Runs on | your browser | the machine's CPU/GPU |
| Port | `3111` | `8000` |
| Job | draws the UI, takes input | does the heavy compute (VGGT, meshes, volumes) |

The split exists because the two jobs are very different. The **frontend** must
run inside whoever is looking at the page (you, or a phone on the wifi), so it
can only do things a browser can do: show text, buttons, and 3D graphics. The
**backend** must reach the GPU and run the pipeline, which a browser cannot do.

So they are two programs, running on the same machine, that talk to each other
over the network.

## What "the browser loads"

When you open `http://localhost:3111`, the browser asks the frontend program for
the page. The frontend (Next.js) replies with some HTML plus a pile of
JavaScript. The browser runs that JavaScript, and the JavaScript builds the page
you see and reacts to your clicks.

Crucially: **all the React code in `web/src/` runs in your browser.** The Python
code in `service/` runs as a separate process. They meet only through HTTP
requests.

## What an API is

"API" sounds grand. It just means: **a list of URLs the backend agrees to
answer.** Each URL is called an *endpoint*.

Open `web/src/lib/api.ts`. It defines the base address:

```ts
const API_PORT = 8000;
function defaultApi(): string {
  if (typeof window === "undefined") return "";
  return `${window.location.protocol}//${window.location.hostname}:${API_PORT}`;
}
export const API = process.env.NEXT_PUBLIC_API_URL || defaultApi();
```

`API` is just a string like `http://localhost:8000`. Every request the frontend
makes starts with that string. The endpoints are:

| Method | URL | What it does |
|---|---|---|
| `GET` | `/health` | "are you alive?" |
| `POST` | `/jobs` | upload a photo set, get a job id back |
| `GET` | `/jobs/{id}` | ask "what stage is this job on now?" |
| `POST` | `/jobs/{id}/run` | start measuring |
| `POST` | `/jobs/{id}/recut` | re-measure with edited cut planes |
| `GET` | `/jobs/{id}/files/...` | download a mesh or CSV |

`GET` = "give me data", `POST` = "here is data, do something with it".

## One concrete trip through the system

This is the whole app in one paragraph. Trace it in the files as you read:

1. You click **Start processing** on the Upload screen.
2. `Upload.tsx:32` calls `createJob(files)`, defined in `api.ts:69`.
3. `createJob` builds a `FormData` (your photos) and sends it to
   `http://localhost:8000/jobs`.
4. The Python backend (FastAPI in `service/app.py`) receives it, saves the
   photos, launches stage 0, and replies with `{ job_id, frames }`.
5. `createJob` returns that object. `Upload.tsx` calls `onStart(job_id, frames)`.
6. `page.tsx:152` handles `onStart`: it stores the job id in state and switches
   the screen to Framing.
7. The Framing screen polls `GET /jobs/{id}` once a second until stage 0
   finishes, then shows which photos passed.
8. You continue. The job runs. The Review screen fetches meshes and lets you
   move the cut plane. The Result screen fetches `volumes.csv` and prints the
   volume.

Every screen change is React state. Every piece of real data comes from an HTTP
call to the backend. That is the entire architecture.

---

# Part 1 — React (the language the UI is written in)

React is a library for building user interfaces out of small, reusable pieces
called **components**. Next.js wraps React and adds the server/routing half.
You will be reading React 19 syntax.

## 1.1 JavaScript, quickly

You must know a few JavaScript idioms before the React code makes sense. All of
these appear in `page.tsx`.

**`const` and `let` declare variables.** `const` cannot be reassigned; `let`
can. (Ignore `var`.)

```ts
const name = "cubit";   // cannot change
let count = 0;          // can change
```

**Arrow functions.** Three equivalent ways to write "a function that adds 1":

```ts
function add1(x) { return x + 1; }      // classic
const add1 = (x) => { return x + 1; };  // arrow
const add1 = (x) => x + 1;              // arrow, implicit return (no braces)
```

The last form — no braces, no `return` — is everywhere in React.

**Objects** are key/value maps:

```ts
const job = { id: "1d39d4580ef4", frames: 6 };
job.id;        // "1d39d4580ef4"
```

**Arrays and their helpers.** `map`, `filter`, `find`, `includes`:

```ts
[1, 2, 3].map((x) => x * 2);        // [2, 4, 6]  — transform each
[1, 2, 3].filter((x) => x > 1);     // [2, 3]      — keep some
[1, 2, 3].find((x) => x === 2);     // 2           — first match
```

`map` is the one that matters: you use it to turn an array of data into an
array of elements on screen. You will see it in `page.tsx:88`.

**Template literals.** Backticks let you splice values into strings:

```ts
`${API}/jobs/${id}`   // "http://localhost:8000/jobs/1d39d4580ef4"
```

**Ternary.** An `if/else` as an expression:

```ts
screen === t.id ? "active" : "inactive"   // if screen equals t.id, "active", else "inactive"
```

**Optional chaining `?.`.** Read a nested property only if the left side is not
null/undefined, else give `undefined`:

```ts
job?.id        // job.id if job exists, else undefined
```

**Nullish coalescing `??`.** Use the right side only when the left is
null/undefined:

```ts
job ? jobPrepBase(job.id) : `/samples/${dataset.id}/prep`   // ternary, same idea
color ?? "#ffffff"   // color, unless it's null/undefined, then white
```

**Spread `...`.** Copy all fields of an object into a new one:

```ts
{ ...prev, [id]: value }   // a copy of prev, with one field overridden
```

**TypeScript annotations.** The `: Type` bits are types, stripped away before
running. `useState<Screen>("samples")` means "this state holds a `Screen`".

## 1.2 JSX: markup that is actually code

A React component's `return` looks like HTML but is not. It is **JSX**, and it
compiles to JavaScript.

```tsx
return <div style={{ color: "red" }}>{name}</div>;
```

Rules that trip everyone up:

- **`{...}` means "evaluate this JavaScript".** Inside the braces you are back
  in code. `<div>{2 + 2}</div>` renders "4".
- **Every JSX tag must close.** `<input>` becomes `<input />`. `<img>` becomes
  `<img />`.
- **`class` is `className`**, because `class` is a reserved word in JS.
- **Styles are objects, not strings.** `style={{ color: "red" }}` — outer `{}`
  is "JSX expression", inner `{}` is a JavaScript object literal. It is an
  object with `camelCase` CSS keys (`backgroundColor`, not `background-color`).
- **A component returns exactly one top-level element**, though that element
  can contain anything.

That is why the code in this repo is full of `style={{ ... }}` — it is just a
JavaScript object holding CSS properties.

## 1.3 Components: the UI is built from functions

A component is **a function whose name starts with a capital letter and that
returns JSX.**

```tsx
function Greeting() {
  return <div>hello</div>;
}
```

You use it like a tag:

```tsx
<div>
  <Greeting />
</div>
```

That is the entire trick of React. You build one big component
(`Page`, `page.tsx:25`) out of many smaller ones (`Samples`, `Upload`,
`Viewport`, `Button`...). Each file in `web/src/components/` exports one or more
components.

Two ways to write the same thing:

```tsx
function Button() { ... }              // function declaration
const Button = () => { ... };          // arrow function
```

This codebase uses the first form (`export function Button(...)`).

**Named vs default export.** A file can export many things by name, or one
thing as `default`.

```tsx
export function Button() {}   // named: import { Button } from "./primitives";
export default function Page() {} // default: import Page from "./page";
```

`page.tsx:25` is a default export. The screens in `components/screens/` are
named exports, which is why `page.tsx:4` imports them with braces.

## 1.4 Props: how components receive data

Components need inputs. Those inputs are called **props** (short for
properties). They arrive as a single object argument.

```tsx
function Greeting({ name }: { name: string }) {
  return <div>hello {name}</div>;
}

<Greeting name="cubit" />   // renders "hello cubit"
```

The `{ name }` destructures the argument object — it pulls the `name` field out.
The `: { name: string }` is the TypeScript type of the argument.

Props flow **one direction**: a parent passes props down to a child. A child
cannot reach up and change its parent directly. Instead, parents pass **callback
functions** down, and the child calls them when something happens. That is what
`onBack`, `onContinue`, `onConfirm`, `onLoadError` are throughout this code — a
child says "this happened" by calling a function the parent gave it.

Look at `page.tsx:149-157`. `Page` renders `<Upload onStart={...} />`. It hands
`Upload` a function. When `Upload` finishes uploading, it calls that function,
and `Page` (the parent) responds by changing state. That is the standard React
communication pattern.

## 1.5 State and the render cycle

The single most important idea in React:

> **A component is a function that is re-run whenever its state changes.**

React keeps a copy of your data called **state**. When state changes, React
re-runs the component function, gets fresh JSX, and updates the screen to match.
You never tell React "make the button blue". You tell it "state `screen` is now
`"upload"`", and React re-renders and figures out the button for itself.

That is why React code reads like "describe the UI for this state" rather than
"do these steps". In `page.tsx:138`, `{screen === "samples" && <Samples .../>}`
is literally: "if the state says samples, the page shows the Samples component."

Consequences you should internalize:

- **You don't change the DOM directly.** You change state, React changes the DOM.
- **Re-render is cheap to write but you must not do side effects inside it.**
  A render should be pure: same props/state → same JSX, and nothing else (no
  network calls, no writing files). Side effects go in `useEffect`.

## 1.6 `useState`

`useState` is how you declare a piece of state. It returns a pair: the current
value, and a function to set it.

```ts
const [screen, setScreenRaw] = useState<Screen>("samples");
```

Break it down:

- `useState("samples")` — create state, initial value `"samples"`.
- It returns `[value, setter]`, which we destructure into `screen` and
  `setScreenRaw`.
- `screen` is the current value. Read it freely.
- `setScreenRaw("upload")` changes the value and triggers a re-render.
- `<Screen>` is the TypeScript type of the value.

Two hard rules:

1. **Never assign to the value directly.** `screen = "upload"` does nothing.
   Always call the setter.
2. **The setter can take a function** when the next value depends on the old:

```ts
setCount((prev) => prev + 1);   // correct under React's batching rules
setCount(count + 1);            // fine too, unless multiple updates race
```

You see the function form in `Review.tsx:158` and `Upload.tsx`'s siblings.

`useState` is called at the top of a component, once per piece of state, in the
same order every render. That ordering is how React knows which state is which.
(Which is why you never put `useState` inside an `if`.)

Every screen in this codebase opens with a stack of `useState` calls. `Result.tsx:27-31`
declares four pieces of state. `page.tsx:26-39` declares six.

## 1.7 `useEffect`

`useEffect` runs code **after** the render, and is the sanctioned place for side
effects (fetching, timers, talking to the browser).

```ts
useEffect(() => {
  fetch(`${API}/health`)
    .then((r) => setBackendUp(r.ok))
    .catch(() => setBackendUp(false));
}, []);
```

The first argument is the function to run. The second argument — the
**dependency array** — decides *when* to run it:

| Dependency array | Runs |
|---|---|
| `[]` | once, after the first render (like "on startup") |
| `[dataset]` | after first render, and again whenever `dataset` changes |
| (omitted) | after every render |

The classic pattern is "load data once on mount": declare state for the result,
then a `useEffect` with `[]` that fetches and calls the setter. See
`Result.tsx:33-35`:

```ts
const [rows, setRows] = useState<VolumeRow[] | null>(null);
useEffect(() => {
  loadVolumes(dataset.volumesCsv).then(setRows).catch((e) => setErr(String(e)));
}, [dataset]);
```

Read it as: "after the component first renders (and again if `dataset` changes),
fetch the volumes CSV, and when it arrives put it into `rows`." The re-render
caused by `setRows` is what actually shows the data.

**The cleanup function.** If the effect returns a function, React calls it
before the next run and when the component unmounts. This is how you cancel a
polling timer:

```ts
useEffect(() => {
  const id = setInterval(tick, POLL_MS);
  return () => clearInterval(id);   // called when leaving the screen
}, [jobId]);
```

See `Processing.tsx:41-69`. The `live` flag inside is another cleanup trick: it
stops a fetch whose result arrived after the component left.

## 1.8 `useMemo` and `useRef`

**`useMemo(fn, deps)`** remembers the result of an expensive calculation and only
recomputes it when a dependency changes.

```ts
const scale = useMemo(() => (rows ? linearScale(rows) : null), [rows]);
```

Read: "compute `linearScale(rows)`; remember it; recompute only when `rows`
changes." Without it, `linearScale` would run on every render even when nothing
relevant changed.

**`useRef(value)`** holds a mutable value that survives re-renders but, unlike
state, does *not* trigger a re-render when changed. `Processing.tsx:39` uses
`done.current` as a flag so the "hand off" happens exactly once. You will also
see refs in the 3D code to keep a camera handle.

## 1.9 `"use client"` and `dynamic()`

Two Next.js concepts, both about *where* code runs.

**`"use client"`** is the very first line of most files here. By default Next.js
(App Router) runs components on the **server** to generate HTML, then re-runs
them in the browser. A component that uses `useState`, `useEffect`, or touches
the browser (`window`, `document`) can only work in the browser, so it is
marked `"use client"`. If you add hooks to a file, it needs this directive.

**`dynamic(..., { ssr: false })`** loads a component only in the browser, never
on the server. The 3D components need a real browser (WebGL, the `three.js`
library), so the screens import them like this (`Review.tsx:11-18`):

```tsx
const Viewport = dynamic(
  () => import("@/components/three/Viewport").then((m) => m.Viewport),
  { ssr: false },
);
```

`ssr: false` = "skip this during server rendering, load it client-side." If the
3D code ran on the server it would crash, because there is no GPU and no
`document` there.

## 1.10 Line-by-line: `page.tsx`

Open `web/src/app/page.tsx`. This is the root component — the whole app. Every
other file exists to be shown by this one. `app/page.tsx` maps to the `/` route,
so it is the first component rendered when the page loads.

### Imports and constants (lines 1–23)

```tsx
"use client";
```
A Client Component (see 1.9): it uses hooks and the browser.

```tsx
import { useEffect, useState } from "react";
```
Pull the two hooks this file uses out of the `react` library.

```tsx
import { Samples } from "@/components/screens/Samples";
```
Import the `Samples` component. `@/` is an alias for `web/src/`, so this is
`web/src/components/screens/Samples.tsx`.

```tsx
import type { CutPlane, SampleDataset, Screen } from "@/lib/types";
```
Import TypeScript *types only* (`import type` vanishes at build time).

```tsx
const TABS: { id: Screen; label: string }[] = [ ... ];
```
A plain array of tab definitions, declared outside the component so it is
created once. Each entry has an `id` (a `Screen`) and a `label` (display text).

### The component and its state (lines 25–39)

```tsx
export default function Page() {
```
The component. `default` export because Next.js expects the route component to
be the default.

```tsx
const { theme, setTheme } = useTheme();
```
`useTheme()` is a custom hook (`theme.ts:12`) that returns a theme name and a
setter. A *custom hook* is just a function that itself uses hooks.

```tsx
const [screen, setScreenRaw] = useState<Screen>("samples");
```
State: which screen is showing. Initial value `"samples"`. `Screen` is a
string-union type (`types.ts:73`), so the value can only be one of those seven
strings.

```tsx
const [dataset, setDataset] = useState<SampleDataset>(SAMPLES[0]);
```
State: which dataset is selected. Starts as the first shipped sample.

```tsx
const [job, setJob] = useState<{ id: string; frames: number } | null>(null);
```
State: the live run, if any. `null` means "browsing a shipped sample, no
backend involved." This distinction — `job` vs `dataset` — runs through the
whole file: samples work offline, jobs need the service.

```tsx
const [afterRun, setAfterRun] = useState<Screen>("review");
const [phase, setPhase] = useState<"measure" | "cut">("measure");
```
Two more pieces of state that track the pipeline's two-pass flow (measure, then
review, then cut). Not important to understand yet.

### Reading the URL (lines 41–58)

```tsx
const setScreen = (s: Screen, d?: SampleDataset) => {
  setScreenRaw(s);
  const ds = d ?? dataset;
  const q = new URLSearchParams({ screen: s, dataset: ds.id });
  window.history.replaceState(null, "", `?${q}`);
};
```
A helper that changes the screen *and* mirrors it into the URL, so the current
view can be bookmarked or reloaded. `d ?? dataset` means "use `d` if given, else
the current `dataset`." `URLSearchParams` builds a `?screen=...&dataset=...`
query string; `history.replaceState` changes the URL without reloading the page.

```tsx
useEffect(() => {
  const q = new URLSearchParams(window.location.search);
  const s = q.get("screen") as Screen | null;
  const d = SAMPLES.find((x) => x.id === q.get("dataset"));
  if (d) setDataset(d);
  if (s && [...].includes(s)) setScreenRaw(s);
}, []);
```
Runs once on load (`[]`). Reads the URL, restores the saved screen and dataset.
This is the mirror of `setScreen` above: one writes the URL, the other reads it
back. (`Array.includes` guards against a bad value in the URL.)

### Probing the backend (lines 59–68)

```tsx
const [backendUp, setBackendUp] = useState(false);
useEffect(() => {
  if (!API) return;
  fetch(`${API}/health`, { signal: AbortSignal.timeout(2500) })
    .then((r) => setBackendUp(r.ok))
    .catch(() => setBackendUp(false));
}, []);
```
On load, ping the backend's `/health` endpoint with a 2.5s timeout. Store whether
it answered. The Upload screen uses `backendUp` to disable itself honestly when
the service is down, instead of letting the user fill a form that will fail.

`fetch` returns a **Promise** — a value that arrives later. `.then(fn)` runs
`fn` when it resolves; `.catch(fn)` runs when it fails. You'll see `.then`/`await`
for all network calls in this codebase. `async`/`await` is sugar over the same
thing (`api.ts:50`).

### The return: header (lines 70–127)

```tsx
return (
  <div style={{ minHeight: "100vh", ... }}>
```
One top-level element, as required. Everything else nests inside.

```tsx
<header style={{ ... }}>
  <div ...>cubit</div>
```
The top bar. Note `header` is a real HTML tag, lowercase. Lowercase tags are
HTML; capitalized tags are components (`<Button/>`, `<Panel/>`).

```tsx
{TABS.map((t) => (
  <button key={t.id} onClick={() => setScreen(t.id)} ...>
    {t.label}
  </button>
))}
```
This is the key loop pattern. `TABS.map(...)` turns the array of six tab
definitions into six `<button>` elements. `key={t.id}` tells React which element
is which across re-renders (required inside `map`). `onClick={() => setScreen(t.id)}`
attaches a click handler: clicking the button changes the screen state, which
re-renders the page with that screen showing.

```tsx
background: screen === t.id ? "var(--soft)" : "transparent",
```
A ternary in a style: the active tab gets a background, the others don't. This
is "describe the UI for the current state" in miniature.

```tsx
{THEMES.map((t) => (
  <button key={t.id} onClick={() => setTheme(t.id)} ... />
))}
```
The same `map` pattern for the three theme swatches. Clicking one calls
`setTheme`, which (via the `useTheme` effect in `theme.ts:14`) sets a
`data-theme` attribute on `<html>`, which CSS uses to switch palettes.

### The return: main (lines 129–207)

```tsx
<main style={{ ... }}>
```
The page body.

```tsx
{screen === "samples" && (
  <Samples onOpen={...} onUpload={() => setScreen("upload")} />
)}
```
**Conditional rendering.** `{cond && <X/>}` renders `X` only when `cond` is
true. So exactly one screen renders at a time, chosen by `screen`. This whole
block is a `switch` written as a sequence of `&&` expressions.

Each screen receives callbacks as props (see 1.4). `onUpload={() => setScreen("upload")}`
is `Page` handing `Samples` a function that changes the screen — the child says
"user clicked Upload", the parent does the navigating.

```tsx
{screen === "upload" && (
  <Upload
    backendUp={backendUp}
    onBack={() => setScreen("samples")}
    onStart={(id, frames) => {
      setJob({ id, frames });
      setDataset(jobDataset(id, frames));
      setScreen("framing");
    }}
  />
)}
```
`onStart` is where a live run begins. `Upload` calls it with a job id; `Page`
records the job, converts it into a dataset (`api.ts:139` dresses a job up with
URLs so the screens that only understand datasets can also show a live run), and
moves to Framing.

```tsx
{screen === "framing" && (
  <Framing
    jobId={job?.id ?? null}
    reportUrl={job ? `${jobPrepBase(job.id)}/framing.json`
                   : `/samples/${dataset.id}/prep/framing.json`}
    ...
    onContinue={async (strict) => {
      if (!job) return setScreen("result");
      await runJob(job.id, strict);
      setPhase("measure");
      setAfterRun("review");
      setScreen("processing");
    }}
  />
)}
```
Notice the `job ? ... : ...` ternaries: a live job reads its report from the
backend, a shipped sample reads a static file. Same component, two data sources.
`onContinue` is `async` because it must `await runJob(...)` — it waits for the
backend to accept the run before switching screens. `if (!job) return setScreen("result")`
handles the sample case: there is nothing to run, so just show the result.

The remaining screens follow the same shape:

```tsx
{screen === "processing" && <Processing jobId={job?.id ?? null} phase={phase}
  onDone={() => setScreen(afterRun)} onBack={...} />}
```
Processing polls the job and calls `onDone` when finished; `afterRun` was set
earlier to decide *where* it goes next (Review after a first measurement, Result
after a re-cut).

```tsx
{screen === "review" && <Review dataset={dataset} live={job !== null}
  onConfirm={async (planes) => { if (!job) return setScreen("result");
  await recut(job.id, planes); setPhase("cut"); setAfterRun("result");
  setScreen("processing"); }} />}
```
`live={job !== null}` tells Review whether it can actually re-cut (live job) or
is just previewing (shipped sample). `onConfirm` for a live job re-runs the cut
and sends the user back through Processing.

```tsx
{screen === "result" && <Result dataset={dataset} onBack={...} />}
{screen === "how" && <How />}
```
The last two screens. `Result` needs only a dataset — a live job is just a
dataset pointing at backend URLs, which is exactly why `jobDataset` exists.

That is the whole app's skeleton: **one state variable (`screen`) chooses which
component shows, and the components communicate with callbacks.** Everything
else in `web/src/` is a screen or a piece shared by the screens.

## 1.11 Line-by-line: `primitives.tsx`

Open `web/src/components/ui/primitives.tsx`. This file holds small, reusable UI
building blocks used by every screen. It is the best place to see props, JSX
styles, and composition in isolation.

### `Panel` (lines 11–21)

```tsx
export function Panel({ children, style, pad = 18 }: {
  children: ReactNode;
  style?: CSSProperties;
  pad?: number;
}) {
  return <div style={{ ...panel, padding: pad, ...style }}>{children}</div>;
}
```

- `children` is the special prop that holds whatever you put *inside* the tag:
  `<Panel>hello</Panel>` → `children` is `"hello"`. `ReactNode` is the type for
  "anything renderable".
- `style?` — the `?` means optional. Callers may omit it.
- `pad = 18` is a **default value**: omitted → 18.
- The style is spread in order: base `panel` (a shared object at line 5), then
  `padding: pad`, then the caller's `style`. Later spreads win, so a caller can
  override the padding. This "spread a base object, override on top" pattern is
  how every component here does styling.

### `Button` (lines 23–73)

```tsx
export function Button({ children, onClick, variant = "ghost", disabled, style, title }: {...}) {
  const base: CSSProperties = { ... };
  const variants: Record<string, CSSProperties> = {
    primary: { background: "var(--accent)", ... },
    ghost:   { ... },
    quiet:   { ... },
  };
  return (
    <button onClick={onClick} disabled={disabled} title={title}
      style={{ ...base, ...variants[variant], ...style }}>
      {children}
    </button>
  );
}
```

- `variant = "ghost"` picks a preset style from the `variants` map. `variant`
  is typed as the union `"primary" | "ghost" | "quiet"`, so TypeScript rejects
  typos.
- `style={{ ...base, ...variants[variant], ...style }}` — three objects merged;
  the caller's `style` wins over the preset.
- `onClick` is just passed through to the real `<button>`. React camelCases DOM
  attributes, so the HTML `onclick` is `onClick`.
- The colour values are CSS variables like `var(--accent)` — defined in
  `globals.css`, swapped by the theme system.

### `Stat` (lines 90–139)

```tsx
export function Stat({ label, value, unit, hint, big = false }: {...}) {
  return (
    <div>
      <Label>{label}</Label>
      <div style={{ font: `${big ? 600 : 500} ${big ? 34 : 19}px/1.1 var(--mono)`, ... }}>
        {value}
        {unit && <span ...>{unit}</span>}
      </div>
      {hint && <div ...>{hint}</div>}
    </div>
  );
}
```

Three things to notice:

- `big ? 600 : 34` — a ternary used to choose between two numbers, building a
  `font` string dynamically.
- `{unit && <span ...>}` — render the unit *only if* it was passed. This is the
  same `{cond && <X/>}` conditional you saw in `page.tsx`.
- It composes another component: `<Label>` is defined above in the same file.
  Composition — small pieces used inside other pieces — is the core React habit.

### `Slider` (lines 162–207)

```tsx
export function Slider({ label, value, min, max, step = 1, suffix, onChange }: {...}) {
  ...
  <input type="range" min={min} max={max} step={step} value={value}
    onChange={(e) => onChange(parseFloat(e.target.value))} />
```

A **controlled input**. The slider's position is not stored in the DOM; it comes
from the `value` prop, which comes from parent state. When the user drags it,
`onChange` fires with a DOM event `e`, and the component forwards the parsed
number up to the parent's `onChange` callback, which updates state, which
re-renders with a new `value`. This "state lives in the parent, the input just
reflects it" loop is the standard React form pattern. You see it in action in
`Review.tsx:360-390`, where moving the Height slider calls `update(...)`, which
recomputes the cut plane.

---

# Part 2 — Three.js (the 3D)

Three.js is a JavaScript library that draws 3D graphics in the browser using
WebGL (the browser's GPU interface). You give it a description of a scene — "a
camera here, lights there, an object made of these triangles" — and it figures
out what the pixels should be, every frame.

## 2.1 What three.js is

The core mental model is a **scene graph**: a tree of objects, where each object
has a position, rotation, and scale, and children are positioned relative to
their parent.

```
scene
 ├── camera        (where you look from)
 ├── lights         (how surfaces are lit)
 └── mesh           (an object you can see)
      ├── geometry  (the shape: a list of vertices)
      └── material  (what the surface looks like)
```

You do not draw triangles yourself. You assemble a tree, three.js walks it and
renders it. When you rotate an object, you change its node's rotation, and all
its children rotate with it — that is what "scene graph" buys you.

The four things every 3D scene needs, all visible in `Viewport.tsx`:

- **Camera** — the viewpoint. Has a position and a field of view (`fov`).
- **Light** — without it everything is black. Three has several kinds; this app
  uses a hemisphere (soft ambient sky/ground light) and two directional lights
  (like the sun, from a direction).
- **Mesh** — a visible object = geometry + material.
- **Renderer** — the part that turns the scene into pixels on a `<canvas>`. You
  rarely touch it directly; `@react-three/fiber` manages it.

## 2.2 Geometry: `BufferGeometry` and `BufferAttribute`

This is the part that confused you, and it is simpler than it looks.

A 3D object is made of **triangles**. Each triangle is three vertices, each
vertex is an `(x, y, z)` position. So the entire shape of a mesh is just a big
list of numbers.

`BufferGeometry` is exactly that: a bag of named arrays. The most important is
`position` — every vertex, as a flat `Float32Array`:

```
[x0, y0, z0,   x1, y1, z1,   x2, y2, z2,   ...]
 └── vertex 0 ──┘ └── vertex 1 ──┘
```

Nine numbers make a single triangle. A few thousand points = a few tens of
thousands of numbers. `Float32Array` is just a JavaScript "array of 32-bit
floats" — a compact, typed list of numbers.

A `BufferAttribute` says **how to read that array**. `new THREE.BufferAttribute(data, 3)`
means "take `data` and read it three numbers at a time, as x/y/z." That is the
link between "a flat array" and "vertices".

You see this precisely in `CutReview.tsx:108-111`:

```tsx
const g = new THREE.BufferGeometry();
g.setAttribute("position", new THREE.BufferAttribute(data, 3));
```

`data` is a flat `Float32Array` of points; the geometry now holds a `position`
attribute that three.js reads as vertices. That is the entire trick to turning a
raw list of numbers into a drawable cloud of points.

Besides `position`, a geometry can carry other attributes: `normal` (which way
each vertex faces, for lighting) and `color` (a per-vertex RGB). The PLY files
this app loads carry vertex colours, and `MeshView.tsx:35` checks for them with
`geometry.getAttribute("color")`.

**Why "Buffer"?** The name comes from GPU buffers — the arrays get uploaded to
the GPU as a block. You can ignore the name and remember: *a `BufferGeometry`
is a set of flat number arrays plus the instructions for reading them.*

## 2.3 Materials and meshes

A **material** decides how a surface is shaded. A **mesh** pairs a geometry with
a material.

Three materials appear in this app:

- `meshStandardMaterial` (`MeshView.tsx:40`) — the physically-based shader: it
  responds to lights, and takes `roughness`, `metalness`, `color`, `opacity`.
  This is the realistic-looking one.
- `meshBasicMaterial` (`CutReview.tsx:156`) — ignores lighting; renders a flat
  colour. Good for UI overlays like the cut-plane discs.
- `pointsMaterial` (`CutReview.tsx:116`) — used with `<points>` to draw point
  clouds: each vertex becomes a tiny square.

Useful material options you'll see:

- `vertexColors` — colour each vertex by the geometry's `color` attribute
  instead of one flat colour (`MeshView.tsx:41`).
- `wireframe` — draw the triangles' edges instead of filled faces.
- `transparent` / `opacity` — see-through surfaces (the plane discs).
- `side: THREE.DoubleSide` — draw both sides of a triangle (the default draws
  only the front, which makes an open surface invisible from behind).
- `depthWrite: false` — don't record this surface in the depth buffer, so two
  overlapping transparent discs don't hide each other.

## 2.4 `@react-three/fiber`: three.js as JSX

`@react-three/fiber` (imported as `Canvas` in `Viewport.tsx:3`) is the bridge
that lets you write three.js as JSX. Every JSX tag inside `<Canvas>` maps to a
three.js object, and props map to object properties.

```tsx
<Canvas camera={{ position: [0, 0, 10] }}>
  <mesh position={[0, 0, 0]}>
    <sphereGeometry args={[1, 32, 32]} />
    <meshStandardMaterial color="red" />
  </mesh>
</Canvas>
```

Reads as: create a renderer and camera, then a mesh, and give it a sphere
geometry and a red material. `args={[...]}` passes constructor arguments to the
underlying three.js class — `<circleGeometry args={[radius, 64]} />` calls
`new THREE.CircleGeometry(radius, 64)`.

Two facts to lock in:

1. **Nesting = the scene graph.** A `<mesh>` inside a `<group>` is a child of
   that group; moving the group moves the mesh. `CutReview.tsx:153` puts the
   plane discs inside a `<group position={...} quaternion={...}>` so both discs
   share one placement.
2. **This is not the DOM.** Inside `<Canvas>`, there is no HTML. You cannot
   render a `<div>` or a `<button>` in there — that is why `MeshView.tsx:26-28`
   comments that it must hand load errors up to a component that *does* live in
   the DOM (`Viewport`). The `<Canvas>` subtree and the HTML page are two
   separate worlds, glued by React state.

`<Canvas>` also handles the render loop: it re-renders every frame (60fps), the
same way a game does, so you never ask for "one draw" — you describe the scene
and it keeps drawing it.

## 2.5 `@react-three/drei`: ready-made helpers

`drei` is a companion library of useful three.js components. Three appear here:

- **`OrbitControls`** (`Viewport.tsx:143`) — the "drag to rotate, scroll to
  zoom, right-drag to pan" camera behaviour. `makeDefault` wires it up
  automatically. `minDistance`/`maxDistance` bound the zoom, and `maxPolarAngle`
  stops the camera going below the floor.
- **`Grid`** (`Viewport.tsx:128`) — an infinite floor grid with major/minor
  lines (`sectionSize` = major every 10 cells, `cellSize` = 1). `fadeDistance`
  fades it into the distance.
- **`Bounds`** (`MeshView.tsx:38`, `CutReview.tsx:296`) — watches its children
  and frames the camera so they fit the viewport. `fit` = move the camera to
  fit; `clip` = ignore points outside a margin; `observe` = refit when the
  geometry changes; `margin` = how much breathing room. This is what makes the
  mesh "auto-frame" to fill the window.

## 2.6 Line-by-line: `Viewport.tsx`

Open `web/src/components/three/Viewport.tsx`. This is the shared 3D stage: every
3D screen wraps its content in `<Viewport>`, which supplies the camera, lights,
grid, and controls.

### WebGL detection (lines 14–21)

```tsx
function webglAvailable(): boolean {
  try {
    const c = document.createElement("canvas");
    return !!(c.getContext("webgl2") || c.getContext("webgl"));
  } catch {
    return false;
  }
}
```

Creates a throwaway `<canvas>` and asks the browser for a WebGL context. If the
browser can't do 3D at all (some embedded webviews), this returns false. The
app checks this *before* rendering the 3D scene, so a missing GPU shows a
"cannot draw 3D" message instead of a silently blank box.

### The props (lines 35–50)

```tsx
export function Viewport({
  children,
  cameraPosition = [34, 26, 34],
  gridSize = 60,
  showGrid = true,
  className,
  error = null,
}: { children: ReactNode; ... }) {
```

`children` is whatever the caller puts inside `<Viewport>` — the actual mesh or
point cloud (`MeshView` or `CutReview`). The rest are optional knobs with
defaults. `error` is a string the caller passes when its geometry failed to
load, so the viewport can say *why* it is blank.

### Probing WebGL after mount (lines 53–54)

```tsx
const [noWebgl, setNoWebgl] = useState(false);
useEffect(() => setNoWebgl(!webglAvailable()), []);
```

`webglAvailable()` touches `document`, which only exists in the browser, so it
must run in an effect after mount — not during render (the server render would
crash, and a mismatch between server and client breaks hydration). The result is
stored in state.

### The error branch (lines 56–102)

```tsx
const problem = noWebgl ? { title: "This browser cannot draw 3D", detail: "..." }
  : error ? { title: "Nothing to draw", detail: error }
  : null;
if (problem) { return <div ...>{problem.title}...{problem.detail}</div>; }
```

A small decision tree: no WebGL → one message; a load error → another; neither →
render the real scene. Notice the component returns **early** with a plain HTML
`<div>` — this is the "component can only draw in the DOM" half of the 3D/DOM
split.

### The scene (lines 104–154)

```tsx
<Canvas dpr={[1, 2]} camera={{ position: cameraPosition, fov: 40, near: 0.1, far: 2000 }}
        gl={{ antialias: true }}>
```

The renderer + camera. `dpr={[1, 2]}` = device pixel ratio capped at 2 (crisp
on retina screens without waste). `fov: 40` is a fairly narrow field of view —
less distortion, like a portrait lens. `near`/`far` are the clipping planes:
things closer than 0.1 or farther than 2000 units are not drawn.

```tsx
<hemisphereLight intensity={0.55} groundColor="#404040" />
<directionalLight position={[18, 30, 14]} intensity={1.5} />
<directionalLight position={[-16, 12, -12]} intensity={0.45} />
```

Three lights: a hemisphere for soft overall light, and two directional lights —
a strong key light from one side and a weak fill light from the other — so the
mesh has shading and depth.

```tsx
<Suspense fallback={null}>{children}</Suspense>
```

`Suspense` is React's "wait for something that loads slowly" boundary. The
`children` (the mesh) load their PLY files asynchronously; while they load,
render `null` (nothing), then swap in the geometry when ready. `usePly` (below)
is what actually suspends by returning `null` until loaded.

```tsx
{showGrid && (
  <Grid args={[gridSize, gridSize]} cellSize={1} cellThickness={0.5}
        sectionSize={10} sectionThickness={1} infiniteGrid
        fadeDistance={gridSize * 2.2} fadeStrength={1.5}
        cellColor="#8a8a8a" sectionColor="#5f5f5f" followCamera={false} />
)}
```

The floor grid. `cellSize={1}` means one grid square = 1 unit — and because the
geometry is scaled to centimetres on load (Part 3), one square = 1 cm. That is
why the UI can say "1 grid square = 1 cm": the grid and the geometry live in the
same unit system.

```tsx
<OrbitControls makeDefault enableDamping dampingFactor={0.08}
               minDistance={4} maxDistance={400}
               maxPolarAngle={Math.PI / 2 - 0.02}
               target={[0, 6, 0]} />
```

The interaction controls. `enableDamping` smooths rotation (a slight inertia).
`maxPolarAngle={Math.PI / 2 - 0.02}` keeps the camera at or above the floor —
the "polar angle" is measured from the +Y axis, so capping it just under 90°
prevents looking up through the grid from below. `target` is the point the
camera orbits around.

## 2.7 Line-by-line: `MeshView.tsx`

Open `web/src/components/three/MeshView.tsx`. This is the simplest 3D component:
load one PLY, show it as a shaded mesh, auto-frame it.

```tsx
const { geometry, error } = usePly(url, scale);
```
`usePly` is the custom hook (2.8) that loads the PLY and applies the coordinate
transform. It returns the loaded geometry and any load error.

```tsx
useEffect(() => {
  onLoadError?.(error);
}, [error, onLoadError]);
```
This component draws inside the `<Canvas>`, so it cannot show an error message
itself (no DOM in there). It forwards the error up to the parent via the
`onLoadError` callback. `onLoadError?.(...)` is optional-call syntax: call it
only if it was provided.

```tsx
if (!geometry) return null;
```
Until the PLY loads, `geometry` is null, so render nothing. Combined with the
`<Suspense>` in `Viewport`, this is what makes the mesh "appear" once ready.

```tsx
const hasColors = !!geometry.getAttribute("color");
```
Does this geometry carry per-vertex colours? `getAttribute("color")` returns the
attribute or `undefined`; `!!` coerces to boolean.

```tsx
<Bounds fit clip observe margin={1.35}>
  <mesh geometry={geometry} castShadow receiveShadow>
    <meshStandardMaterial
      vertexColors={hasColors && !color}
      color={color ?? "#ffffff"}
      roughness={0.72} metalness={0.02}
      wireframe={wireframe} transparent={opacity < 1} opacity={opacity}
      side={THREE.DoubleSide} />
  </mesh>
</Bounds>
```

- `<Bounds>` auto-frames the camera around the mesh.
- `<mesh geometry={geometry}>` attaches the loaded geometry. `castShadow` /
  `receiveShadow` are inert here (no shadows configured) but harmless.
- `vertexColors={hasColors && !color}` — use the vertex colours *only if* they
  exist and the caller didn't override with a flat `color`.
- `color ?? "#ffffff"` — a flat colour, defaulting to white. When `color` is
  set, `vertexColors` is false, so every vertex takes this colour (the "ghost"
  / overlay look).
- `transparent={opacity < 1}` — only pay the cost of transparency when opacity
  is actually below 1.
- `side={THREE.DoubleSide}` — draw both faces, since reconstructed meshes are
  often not closed (a one-sided default would hide the inside).

## 2.8 Line-by-line: `usePly.ts`

Open `web/src/components/three/usePly.ts`. This is the heart of the 3D half: it
loads a PLY file and moves it from "the pipeline's coordinates" to "scene
coordinates". The coordinate math is explained in full in Part 3; here I cover
the *mechanics* of loading and the hook.

### `transformToScene` (lines 23–41)

```tsx
export function transformToScene(geom: THREE.BufferGeometry, scale: number) {
  geom.rotateX(-Math.PI / 2);          // Z-up -> Y-up
  geom.scale(scale, scale, scale);     // mesh units -> cm
  geom.computeBoundingBox();
  const bb = geom.boundingBox!;
  const cx = (bb.min.x + bb.max.x) / 2;
  const cz = (bb.min.z + bb.max.z) / 2;
  const offset = new THREE.Vector3(-cx, -bb.min.y, -cz);
  geom.translate(offset.x, offset.y, offset.z);
  geom.computeVertexNormals();
  geom.computeBoundingBox();
  geom.userData.sceneOffset = offset;
  geom.userData.sceneScale = scale;
  return geom;
}
```

It mutates the geometry in place, applying three operations in order: rotate
(lie the object upright), scale (convert units to cm), translate (recentre and
drop to the floor). `geom.boundingBox!` — the `!` tells TypeScript "I know this
is not null" (it was just computed). `computeVertexNormals()` recomputes face
normals, because the exported PLYs may not carry them and lighting needs them.

The last two lines are crucial: it stores the offset and scale **on the
geometry** (`userData` is three.js's "attach arbitrary data" slot). Why?
Because other objects — the cut planes — must receive the *same* transform to
line up with the cloud. Storing it here lets `CutReview.tsx` read it back later
instead of recomputing it and risking a mismatch.

### `pointToScene` and `dirToScene` (lines 44–60)

```tsx
export function pointToScene(p, scale, offset) {
  return new THREE.Vector3(p[0], p[1], p[2])
    .applyAxisAngle(new THREE.Vector3(1, 0, 0), -Math.PI / 2)
    .multiplyScalar(scale)
    .add(offset);
}
export function dirToScene(d) {
  return new THREE.Vector3(d[0], d[1], d[2])
    .applyAxisAngle(new THREE.Vector3(1, 0, 0), -Math.PI / 2)
    .normalize();
}
```

These re-apply the same transform to **single points** (like a plane's centre)
and **directions** (like a plane's normal). A point is rotated, scaled, and
translated — all three. A direction is rotated only: a direction has no length
to scale and no position to move. `applyAxisAngle(axis, angle)` rotates around
an axis by an angle; `.normalize()` rescales a vector to length 1.

### The hook (lines 62–112)

```tsx
export function usePly(url: string | null, scale = 1): PlyState {
  const [state, setState] = useState<PlyState>({
    geometry: null, error: null, loading: !!url,
  });
  useEffect(() => {
    if (!url) { setState({ geometry: null, error: null, loading: false }); return; }
    let cancelled = false;
    let created: THREE.BufferGeometry | null = null;
    setState({ geometry: null, error: null, loading: true });

    new PLYLoader().load(
      url,
      (geom) => {
        if (cancelled) { geom.dispose(); return; }
        created = transformToScene(geom, scale);
        setState({ geometry: created, error: null, loading: false });
      },
      undefined,
      () => {
        if (!cancelled)
          setState({ geometry: null, error: `Could not load ${url.split("/").pop()}`, loading: false });
      },
    );

    return () => { cancelled = true; created?.dispose(); };
  }, [url, scale]);

  return state;
}
```

- `PLYLoader().load(url, onSuccess, onProgress, onError)` — three.js's PLY
  loader. It fetches the file, parses it, and calls `onSuccess` with a fresh
  `BufferGeometry`. `onProgress` is `undefined` (we don't need it). The error
  callback builds a message from the filename (`url.split("/").pop()` = the last
  path segment).
- **`cancelled` flag + cleanup**: if the component unmounts (user navigates
  away) before the load finishes, the cleanup function sets `cancelled = true`.
  The `onSuccess` handler then `dispose()`s the geometry (frees GPU memory) and
  skips `setState` — you must not set state on an unmounted component. The
  cleanup also disposes any geometry that *was* created, so repeated navigation
  doesn't leak GPU memory. This is the standard "async in a hook" pattern.
- The dependency array `[url, scale]` means: re-run this load whenever the URL
  or scale changes (e.g. switching dataset, or once `scale` becomes known).

The `positionsOf` helper at the bottom (lines 115–117) just extracts the flat
position array from a geometry — used by the cut test in Part 3.

---

# Part 3 — The app's glue

Part 2 explained the 3D machinery. This part explains what ties it to *this*
pipeline: the coordinate transform, the cut rule, the API client, and the data
wrangling.

## 3.1 The coordinate transform

This is the single most important piece of geometry in the app, and it is
explained in the comment at the top of `usePly.ts:7-22`. Two coordinate systems
are involved, and they disagree on which way is "up":

- **The pipeline** (Python) levels the scene so that **Z is up**.
- **three.js** assumes **Y is up**.

If you load a pipeline PLY into three.js without correcting this, the limb lies
on its side and floats nowhere near the floor grid. So `transformToScene` does,
in order:

1. **Rotate −90° about X.** `geom.rotateX(-Math.PI / 2)` maps the pipeline's
   +Z (up) onto three.js's +Y. Now the object is upright.
2. **Scale to centimetres.** `geom.scale(scale, scale, scale)`. The mesh is in
   arbitrary "mesh units"; `scale` (from `linearScale`, 3.4) converts to cm. Do
   it once here, and every downstream number and the grid are already real.
3. **Recentre and drop to the floor.** Compute the bounding box, centre the
   object on X and Z, and translate it down so its lowest point (`bb.min.y`)
   sits at `y = 0`. Result: the object stands *on* the grid.

The stored `sceneOffset` and `sceneScale` let `pointToScene` / `dirToScene`
apply the identical transform to any point the pipeline hands back — most
notably the **cut planes**, whose centres and normals are in pipeline
coordinates. Without this, the plane disc would float away from the cloud by
exactly the offset amount (the comment in `usePly.ts:35-38` says this explicitly).

One more subtlety the code leans on: `Review.tsx` sliders work in scene cm, but
the plane data is stored in mesh units. The slider code (`Review.tsx:40-44`)
inverts the transform — `(heightCm - offsetY) / scale` — to turn a cm height
back into a mesh-Z coordinate, so the plane data stays in one canonical space
while the UI shows human units.

## 3.2 Line-by-line: `CutReview.tsx`

Open `web/src/components/three/CutReview.tsx`. This is the Review screen's 3D
view: it draws the point cloud split into "kept" and "discarded" by the cut
planes, plus the plane widgets you drag.

### The cut rule (lines 27–71)

```tsx
export function splitByPlanes(positions, planes, sceneScale, offset) {
  const prepared = planes.slice(0, MAX_PLANES).flatMap((p) => {
    const c = pointToScene(p.centroid, sceneScale, offset);
    const nrm = dirToScene(p.normal);
    const vert = nrm.dot(UP);
    if (Math.abs(vert) < 1e-3) return [];
    if (vert < 0) nrm.negate();
    return [{ d0: nrm.dot(c), nrm }];
  });
  ...
  for (let i = 0; i < n; i++) {
    ...
    if (prepared.length === 1) {
      kept = prepared[0].nrm.dot(v) - prepared[0].d0 <= 0;
    } else {
      const a = prepared[0].nrm.dot(v) - prepared[0].d0 <= 0;
      const b = prepared[1].nrm.dot(v) - prepared[1].d0 <= 0;
      kept = a !== b;
    }
    ...
  }
  return { keep, drop };
}
```

The math is the classic "which side of a plane is a point on" test. A plane is
defined by a point `c` on it and a unit normal `nrm`. For any point `v`, the
signed distance to the plane is `nrm · v - nrm · c`. The dot product `nrm · v`
is positive when `v` is on the side the normal points toward. So:

- **0 planes** → keep everything (no cut).
- **1 plane** → keep points *below* it (`distance ≤ 0`, after the normal is
  flipped to point up).
- **2 planes** → keep points *between* them (below the upper **and** above the
  lower — written as `a !== b` because "below one but not the other" is exactly
  "between").

Two defensive details worth noting:

- `nrm.negate()` if the normal points down — a plane is unchanged by flipping
  its whole normal, so "below" is always unambiguous regardless of which sign
  the detector reported.
- A plane standing *vertical* (`vert ≈ 0`) has no "below", so it is skipped
  rather than guessing a side (`if (Math.abs(vert) < 1e-3) return []`).

This mirrors the pipeline's own cut code (`core/segmentation.py:apply_marker_cut`)
exactly, so the browser preview matches what the backend will compute. The
per-point loop is just a dot product per plane — thousands of points, cheap, no
debouncing needed.

### `midAxis` (lines 81–95)

```tsx
function midAxis(geom) {
  const pos = geom.getAttribute("position").array;
  const bb = geom.boundingBox!;
  const mid = (bb.min.y + bb.max.y) / 2;
  const half = (bb.max.y - bb.min.y) * 0.1;
  let sx = 0, sz = 0, k = 0;
  for (let i = 0; i < pos.length; i += 3) {
    if (Math.abs(pos[i + 1] - mid) > half) continue;
    sx += pos[i]; sz += pos[i + 2]; k++;
  }
  ...
}
```

Finds the object's horizontal centre *at mid height* — averaging only points
within a band around the middle. The comment (`CutReview.tsx:74-80`) explains
why: on a leg, the foot juts forward, so the bounding-box centre sits ahead of
the calf. A manually-added plane should sit on the limb's own axis, which this
band-averaged centre approximates. (Note it reads `pos[i + 1]` for Y because
this geometry is already in scene space, where Y is up.)

### The `Points` component (lines 97–125)

```tsx
function Points({ data, color, size, opacity = 1 }) {
  const geom = useMemo(() => {
    const g = new THREE.BufferGeometry();
    g.setAttribute("position", new THREE.BufferAttribute(data, 3));
    return g;
  }, [data]);
  if (!data.length) return null;
  return (
    <points geometry={geom}>
      <pointsMaterial size={size} color={color} sizeAttenuation
                      transparent={opacity < 1} opacity={opacity} />
    </points>
  );
}
```

This is the `BufferGeometry`/`BufferAttribute` construction from 2.2, used in
earnest: it wraps a flat point array in a geometry and renders it as a point
cloud (`<points>`). `useMemo` rebuilds the geometry only when `data` changes.
`sizeAttenuation` makes points smaller when far away (perspective-correct).

### The `PlaneWidget` (lines 128–176)

```tsx
const { position, quaternion } = useMemo(() => {
  const pos = pointToScene(plane.centroid, sceneScale, offset);
  const nrm = dirToScene(plane.normal);
  const q = new THREE.Quaternion().setFromUnitVectors(
    new THREE.Vector3(0, 0, 1), nrm);
  return { position: pos, quaternion: q };
}, [plane, sceneScale, offset]);
```

Converts the plane's centre to scene space (`pointToScene`), and builds a
**quaternion** that rotates the plane's default facing (+Z) onto the plane's
normal. A quaternion is three.js's rotation representation (it avoids the
"gimbal lock" of Euler angles) — you don't need the math, just know it's "the
rotation that turns one direction into another". The disc is then a `<group
position quaternion>` containing a filled circle and a ring.

### The main component (lines 237–328)

```tsx
const { geometry, error } = usePly(url, scale);
const split = useMemo(() => {
  if (!geometry) return null;
  const pos = geometry.getAttribute("position").array;
  const off = geometry.userData.sceneOffset ?? new THREE.Vector3();
  const r = splitByPlanes(pos, planes, scale, off);
  onCounts?.(r.keep.length / 3, r.drop.length / 3);
  return r;
}, [geometry, planes, scale]);
```

Loads the mesh, then splits its points by the planes. `sceneOffset` is read back
from `userData` (stored by `transformToScene`) — this is the "give the cut
planes the identical transform" handoff from 3.1. `onCounts` reports how many
points are kept/dropped so the Review screen can print "N kept · M discarded".

```tsx
useEffect(() => {
  if (!geometry) return;
  const bb = geometry.boundingBox!;
  const off = geometry.userData.sceneOffset;
  onExtent?.(bb.min.y, bb.max.y, off ?? new THREE.Vector3(), midAxis(geometry));
}, [geometry]);
```

Once loaded, reports the cloud's vertical extent, the offset, and the mid-height
axis back to Review — these become the slider bounds and the manual-plane
position. (The `// eslint-disable-next-line react-hooks/exhaustive-deps`
comments here are a deliberate choice: these callbacks are intentionally read
fresh rather than listed as dependencies, to avoid re-running on every render.)

Finally it renders, inside `<Bounds>`: two `Points` clouds (kept in blue, dropped
in grey at low opacity), one `OriginWidget` per plane (yellow, the *detected*
marker line — a fixed record that never moves), and one `PlaneWidget` per plane
(green, the *proposed* cut — draggable via the sliders). The yellow/green pair
is the whole point of the review: you see the measurement as detected *and* your
edit side by side.

## 3.3 Line-by-line: `api.ts`

Open `web/src/lib/api.ts`. This is the frontend's entire vocabulary for talking
to the backend. It has no React — just plain functions that make HTTP requests
and return data.

### The base URL (lines 16–21)

```tsx
function defaultApi(): string {
  if (typeof window === "undefined") return "";
  return `${window.location.protocol}//${window.location.hostname}:${API_PORT}`;
}
export const API = process.env.NEXT_PUBLIC_API_URL || defaultApi();
```

`API` is the service's address. The clever bit (comment at `api.ts:6-13`): it
defaults to *the page's own hostname*, not `localhost`. If a phone loads the
page from `http://192.168.1.5:3111`, then `API` becomes `http://192.168.1.5:8000`
— reachable from the phone. `localhost` would point the phone at itself.
`NEXT_PUBLIC_API_URL` (an environment variable, prefixed `NEXT_PUBLIC_` so Next
exposes it to the browser) overrides this when the service lives elsewhere.

### The status shape (lines 28–48)

```tsx
export interface JobStatus {
  job_id: string;
  state: "queued" | "prep" | "awaiting-framing" | "running" | "awaiting-cut" | "done" | "failed";
  stage: number;
  ...
}
```

A TypeScript `interface` describing the JSON the backend returns for a job. It is
a *type*, not data — it exists only to catch mistakes at compile time. The
`state` field is the job's lifecycle, and the screens poll it to decide what to
show.

### The `json` helper (lines 50–65)

```tsx
async function json<T>(res: Response): Promise<T> {
  if (!res.ok) {
    let detail = `${res.status}`;
    try {
      const body = await res.json();
      if (body?.detail) detail = String(body.detail);
    } catch { /* not JSON */ }
    throw new Error(detail);
  }
  return res.json() as Promise<T>;
}
```

Wraps every response: if the status is not OK, throw an error. It tries to read
FastAPI's `detail` field (the human-readable reason, written for the person
holding the camera) and surfaces that verbatim rather than a bare status code.
The `<T>` is a generic — "this function returns whatever type you asked for".

### `createJob` (lines 69–98)

```tsx
export function createJob(files, onProgress?) {
  const form = new FormData();
  files.forEach((f) => form.append("files", f, f.name));
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    xhr.open("POST", `${API}/jobs`);
    xhr.upload.onprogress = (e) => { ... onProgress(e.loaded / e.total); };
    ...
    xhr.send(form);
  });
}
```

Uploads the photos. Two things to notice:

- **`FormData`** is the browser's way to send files over HTTP (like a `<form>`
  with a file input). Each file is appended under the key `files`.
- **It uses `XMLHttpRequest`, not `fetch`**, because `fetch` still cannot report
  upload progress, and uploading a dozen full-resolution HEICs over wifi is slow
  enough to need a progress bar (the comment at `api.ts:76-78`). `xhr.upload.onprogress`
  fires repeatedly with `loaded / total`, which the Upload screen turns into a
  percentage.

It returns a `Promise` that resolves with `{ job_id, frames }` on success.

### The rest (lines 100–160)

```tsx
export async function getJob(id) { ... }                 // GET /jobs/{id}
export async function runJob(id, strict) { ... }         // POST /jobs/{id}/run
export async function recut(id, planes) { ... }          // POST /jobs/{id}/recut
```

Three thin wrappers. `runJob` sends `{ strict }` — `strict: false` is the user
overruling the framing gate. `recut` sends the edited planes (mapping each to
just `centroid`, `normal`, `npts` — the fields the backend expects).

```tsx
export function jobDataset(id, frames): SampleDataset {
  const f = (name) => `${API}/jobs/${id}/files/${name}`;
  return { id, label: "Your upload", subject: ..., nominalMl: null, frames,
    meshes: { leg: f("leg_mesh.ply"), ... },
    volumesCsv: f("volumes.csv"), cuttingLine: f("cutting_line.json") };
}
```

The key abstraction. A live job and a shipped sample differ *only* in where
their files live. `jobDataset` dresses a job up as a `SampleDataset` whose URLs
point at the backend's `/files/...` endpoint. Result and Review then need to
know nothing about jobs — they just load "a dataset". `nominalMl: null` because
an uploaded object has no known truth, and the UI must never invent one.

## 3.4 Line-by-line: `data.ts`

Open `web/src/lib/data.ts`. This file holds the shipped samples, the stage
lists, and the parsing/derivation helpers.

### `SAMPLES` (lines 5–36)

An array of two precomputed runs, each a `SampleDataset` with URLs under
`/samples/...` (static files served by Next, no backend). These are what make
the site work offline.

### The stage lists (lines 46–63)

`MEASURE_STAGES` and `CUT_STAGES` are the two halves of the pipeline, displayed
by the Processing screen. They overlap in stage *numbers* (3–6) but not in
meaning: the first pass measures the reference cube, the second applies the
confirmed cut and measures the object. The screen is told which pass it is via
the `phase` prop (`page.tsx:39`). The `seconds` fields are expectations, not
timers — the actual progress comes from polling the job's `stage` field.

### `parseVolumesCsv` (lines 68–107)

```tsx
export function parseVolumesCsv(text: string): VolumeRow[] {
  const [header, ...lines] = text.trim().split("\n");
  const cols = header.split(",");
  return lines.filter((l) => l.trim()).map((line) => {
    const cells = line.split(",");
    const has = (k) => cols.indexOf(k) >= 0;
    const get = (k) => cells[cols.indexOf(k)];
    const num = (k) => { const v = parseFloat(get(k)); return Number.isFinite(v) ? v : 0; };
    ...
  });
}
```

A minimal CSV parser. `split("\n")` turns the file into lines; the first line is
the header; each remaining line becomes an object. `[header, ...lines]` is array
destructuring with a rest spread. `has`/`get`/`num` are small closures that
look up a column by name (so the column order doesn't matter). The block around
`aabb` (`data.ts:87-98`) handles two Stage 6 versions naming their columns
differently, normalising them to one shape.

### `linearScale` (lines 129–148)

```tsx
export function linearScale(rows: VolumeRow[]): number | null {
  const ref = rows.find((r) => r.is_ref);
  if (!ref) return null;
  let scale = REFERENCE_CM / ((ref.obb_b + ref.obb_c) / 2);
  if (ref.aabb) {
    scale = Math.cbrt(ref.real_vol_cm3 / ref.volume);
  }
  return Number.isFinite(scale) && scale > 0 ? scale : null;
}
```

This is where "cm per mesh unit" comes from — the number that gets fed to
`usePly` to scale the whole scene. The reference cube is 14 cm; its *measured*
mesh-unit size gives the conversion.

Two subtleties, both documented in the comments:

- **Use the horizontal edges only.** `obb_b`/`obb_c` are the horizontal
  extents; `obb_a` is vertical, which the floor truncates, so averaging it in
  would bias the scale small.
- **Axis-aligned fallback.** If the CSV came from the axis-aligned Stage 6, the
  extents read the cube's *diagonal* (wrong by ~26% on a tilted cube). In that
  case it falls back to the volume ratio's cube root — the same derivation that
  Stage 6 used, so the viewer stays consistent with the printed number.

The guard at the end (`isFinite && scale > 0`) matters: the old code returned
`14 / 0` here, handing `Infinity` to the loader, which multiplied every vertex
by it and produced a scene of `NaN`s — a blank viewport indistinguishable from a
browser with no WebGL at all. Returning `null` lets the caller say "cannot draw
to scale" instead.

### `loadCutPlanes` (lines 155–174)

```tsx
export async function loadCutPlanes(url: string): Promise<CutPlane[]> {
  try {
    const res = await fetch(url);
    if (!res.ok) return [];
    const json = await res.json();
    return (json.markers ?? []).map((m) => ({ ... }));
  } catch { return []; }
}
```

Loads the detected cut planes from `cutting_line.json`, mapping each to a
`CutPlane` with an `id`, a `centroid`, a `normal`, and — importantly — an
`origin` copy. The `origin` preserves where detection *originally* put the
plane, so the Review screen can show "the marker was here" next to "you moved it
here". On any failure it returns `[]` (no planes), which the UI reads as "no
marker detected".

## 3.5 The screen state machine, end to end

Tie it all together. Here is the full lifecycle of a live upload, each step
naming the file and state change that makes it happen:

1. **Samples** (`page.tsx:138`). You click "Upload a photo set" → `onUpload` →
   `setScreen("upload")`. `page.tsx` re-renders showing `<Upload>`.
2. **Upload** (`Upload.tsx`). You pick files, click Start. `send()` calls
   `createJob` (`api.ts:69`), showing a progress bar via `setProgress`. On
   success it calls `onStart(job_id, frames)`.
3. **Page records the job** (`page.tsx:152`): `setJob(...)`,
   `setDataset(jobDataset(...))`, `setScreen("framing")`.
4. **Framing** (`Framing.tsx`). No job report exists yet, so it polls
   `getJob` (`api.ts:100`) once a second (`Framing.tsx:99-121`). When the
   backend finishes stage 0 and reports `framing`, it shows each photo's
   verdict. You click Continue → `onContinue` → `runJob` (`api.ts:106`) →
   `setScreen("processing")`.
5. **Processing** (`Processing.tsx`). Polls `getJob`, advancing the stage list
   as the backend reports each stage finishing. On `done`/`awaiting-cut` it
   calls `onDone` → `setScreen(afterRun)` (Review, first time).
6. **Review** (`Review.tsx`). Loads `volumes.csv` (`loadVolumes`) and
   `cutting_line.json` (`loadCutPlanes`); renders `<CutReview>`, which loads the
   mesh via `usePly` and shows the kept/dropped split. Moving a slider calls
   `update` (`Review.tsx:157`), recomputing the plane and re-splitting the cloud
   live. You click Confirm → `onConfirm` → `recut` (`api.ts:118`) →
   `setScreen("processing")` with `phase = "cut"`.
7. **Processing** (again, cut phase) → `onDone` → `setScreen("result")`.
8. **Result** (`Result.tsx`). Loads `volumes.csv`, computes `linearScale`,
   renders `<MeshView>`, and prints the measured volume, dimensions, and the
   reference self-check.

Two threads run through every step:

- **Navigation is always a state change.** `setScreen` is the only way screens
  change. There are no page loads, no links, no routing in the traditional
  sense — it is one component re-rendering with a different `screen` value.
- **Data is always an HTTP call.** Every real number and every mesh comes from
  `fetch`/`XHR` to the backend (or a static `/samples/...` file for the shipped
  samples). The screens never compute the pipeline themselves; they only display
  and let you steer it.

Read this section again after you have seen the app run. The names will stop
being abstract once you have watched a real job move through these screens.

---

# Glossary

- **API** — a set of URLs a server answers. "The backend's API" = the list of
  `/jobs`, `/health`, etc.
- **Backend** — the server program (here, the Python compute service). Runs the
  heavy work, answers HTTP requests.
- **Component** — a function returning JSX, the unit of a React UI.
- **CSS variable** — `--name` values defined in CSS and read as `var(--name)`.
- **Endpoint** — one URL in an API.
- **Fetch** — the browser function that makes an HTTP request; returns a Promise.
- **Frontend** — the code that runs in the browser (here, the React app).
- **Hook** — a `useXxx` function that lets components use React features
  (`useState`, `useEffect`, `useMemo`, custom ones like `useTheme`).
- **JSX** — the markup-like syntax React components return; compiles to JS.
- **Mount** — the first time a component renders and appears on the page.
- **Promise** — a value that will exist later; chain with `.then`/`.catch` or
  `await`.
- **Prop** — an input to a component.
- **Re-render** — re-running a component function after state changes.
- **State** — data React tracks; changing it re-renders the component.
- **SSR** — server-side rendering: running components on the server to generate
  the initial HTML.
