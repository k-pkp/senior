/**
 * Circumference at a cutting plane, fitted in the browser while the plane moves.
 *
 * A port of `pipeline/core/crosssection.py`: Halir-Flusser direct least-squares
 * ellipse fit, Ramanujan's second approximation for the perimeter, plus the two
 * diagnostics that say whether the number means anything — angular coverage and
 * an independent median-radius polygon.
 *
 * Why a port and not a call to the service: the review screen already splits the
 * cloud client-side so the cut updates under the pointer, and a circumference
 * that arrived a round-trip later would describe a plane the user had already
 * moved. The maths is a few hundred flops per point on ~11k points.
 *
 * **Scene space is centimetres.** `usePly` scales mesh units by `linearScale`
 * (cm per unit, the same figure Stage 6 reports volumes on) and rotates Z-up to
 * Y-up, so everything here is already in cm and no scale factor appears in the
 * arithmetic. That is the one thing to preserve if the loader ever changes.
 *
 * Two differences from the Python, both deliberate:
 *
 *  - It slices `leg_no_cut.ply`, the cloud the review screen draws, where the
 *    pipeline slices `leg_open.ply`. They are the same points except near the
 *    floor, where `leg_no_cut` carries the fabricated skirt and bottom cap. A
 *    plane low enough to cut through those is flagged rather than silently
 *    measured — see `nearFloor`.
 *  - It reports a reason instead of raising, because a plane being dragged
 *    passes through positions where no ellipse exists and the panel has to say
 *    so without the screen going blank.
 */

/** Half-thickness of the slice, in real millimetres — `SLAB_HALF_MM`. */
export const SLAB_HALF_MM = 4.0;

/** Above this largest angular gap the fit extrapolates across a hole. */
export const MAX_ARC_GAP_DEG = 45.0;

/** Points nearer the cloud's base than this (cm) may hit fabricated geometry. */
export const FLOOR_GUARD_CM = 2.0;

export interface CrossSection {
  ok: true;
  circumferenceCm: number;
  aCm: number;
  bCm: number;
  polygonCm: number;
  nSlab: number;
  coverage: number;
  maxGapDeg: number;
  residRmsMm: number;
  partialArc: boolean;
  nearFloor: boolean;
}

export interface CrossSectionFailure {
  ok: false;
  reason: string;
}

export type CrossSectionResult = CrossSection | CrossSectionFailure;

type Vec3 = readonly [number, number, number];
type Mat3 = number[][];

// Multiplies two 3x3 matrices.
function mul(a: Mat3, b: Mat3): Mat3 {
  const out: Mat3 = [[0, 0, 0], [0, 0, 0], [0, 0, 0]];
  for (let i = 0; i < 3; i++)
    for (let k = 0; k < 3; k++) {
      const aik = a[i][k];
      if (aik === 0) continue;
      for (let j = 0; j < 3; j++) out[i][j] += aik * b[k][j];
    }
  return out;
}

// Transposes a 3x3 matrix.
function transpose(m: Mat3): Mat3 {
  return [
    [m[0][0], m[1][0], m[2][0]],
    [m[0][1], m[1][1], m[2][1]],
    [m[0][2], m[1][2], m[2][2]],
  ];
}

// Determinant of a 3x3 matrix.
function det3(m: Mat3): number {
  return (
    m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1]) -
    m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0]) +
    m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0])
  );
}

/** Inverse by cofactors. 3x3 only, so the closed form beats a factorisation. */
function inv3(m: Mat3): Mat3 | null {
  const d = det3(m);
  if (!Number.isFinite(d) || Math.abs(d) < 1e-300) return null;
  const c: Mat3 = [
    [
      m[1][1] * m[2][2] - m[1][2] * m[2][1],
      m[0][2] * m[2][1] - m[0][1] * m[2][2],
      m[0][1] * m[1][2] - m[0][2] * m[1][1],
    ],
    [
      m[1][2] * m[2][0] - m[1][0] * m[2][2],
      m[0][0] * m[2][2] - m[0][2] * m[2][0],
      m[0][2] * m[1][0] - m[0][0] * m[1][2],
    ],
    [
      m[1][0] * m[2][1] - m[1][1] * m[2][0],
      m[0][1] * m[2][0] - m[0][0] * m[2][1],
      m[0][0] * m[1][1] - m[0][1] * m[1][0],
    ],
  ];
  return c.map((row) => row.map((v) => v / d));
}

/**
 * Real eigenvalues of a general 3x3, via the characteristic cubic.
 *
 * `numpy.linalg.eig` has no counterpart in the browser and pulling a linear
 * algebra library in for one 3x3 would cost more than the closed form. Complex
 * pairs are dropped: the constraint below only ever selects a real eigenvector,
 * and a matrix whose only elliptical candidate is complex has no ellipse in it.
 */
function realEigenvalues(m: Mat3): number[] {
  // λ³ + c2λ² + c1λ + c0 = 0
  const c2 = -(m[0][0] + m[1][1] + m[2][2]);
  const c1 =
    m[0][0] * m[1][1] - m[0][1] * m[1][0] +
    m[0][0] * m[2][2] - m[0][2] * m[2][0] +
    m[1][1] * m[2][2] - m[1][2] * m[2][1];
  const c0 = -det3(m);

  // Depressed cubic t³ + pt + q, with λ = t - c2/3.
  const shift = c2 / 3;
  const p = c1 - (c2 * c2) / 3;
  const q = (2 * c2 * c2 * c2) / 27 - (c2 * c1) / 3 + c0;

  const roots: number[] = [];
  const discriminant = (q * q) / 4 + (p * p * p) / 27;
  if (discriminant > 0) {
    const sqrtD = Math.sqrt(discriminant);
    const u = Math.cbrt(-q / 2 + sqrtD);
    const v = Math.cbrt(-q / 2 - sqrtD);
    roots.push(u + v - shift);
  } else {
    // Three real roots — the trigonometric form, which avoids the cube roots of
    // complex numbers the algebraic form would need here.
    const r = Math.sqrt(Math.max(-(p * p * p) / 27, 0));
    const cosArg = r === 0 ? 0 : Math.min(1, Math.max(-1, -q / (2 * r)));
    const phi = Math.acos(cosArg);
    const scale = 2 * Math.sqrt(Math.max(-p / 3, 0));
    for (let k = 0; k < 3; k++) {
      roots.push(scale * Math.cos((phi - 2 * Math.PI * k) / 3) - shift);
    }
  }
  return roots.filter((x) => Number.isFinite(x));
}

/** Null vector of (m - λI), as the largest cross product of two of its rows. */
function eigenvector(m: Mat3, lambda: number): Vec3 | null {
  const a: Mat3 = [
    [m[0][0] - lambda, m[0][1], m[0][2]],
    [m[1][0], m[1][1] - lambda, m[1][2]],
    [m[2][0], m[2][1], m[2][2] - lambda],
  ];
  const cross = (u: number[], v: number[]): number[] => [
    u[1] * v[2] - u[2] * v[1],
    u[2] * v[0] - u[0] * v[2],
    u[0] * v[1] - u[1] * v[0],
  ];
  let best: number[] | null = null;
  let bestNorm = 0;
  for (const [i, j] of [[0, 1], [0, 2], [1, 2]]) {
    const c = cross(a[i], a[j]);
    const n = Math.hypot(c[0], c[1], c[2]);
    if (n > bestNorm) {
      bestNorm = n;
      best = c;
    }
  }
  if (!best || bestNorm < 1e-12) return null;
  return [best[0] / bestNorm, best[1] / bestNorm, best[2] / bestNorm];
}

/**
 * Halir-Flusser direct least-squares ellipse fit.
 *
 * Returns the conic (a, b, c, d, e, f) of a x² + bxy + cy² + dx + ey + f = 0,
 * constrained to an ellipse by 4ac - b² = 1. The design matrix is split into
 * quadratic and linear halves so the eigenproblem is a well-conditioned 3x3
 * rather than a near-singular 6x6.
 */
export function fitEllipseDirect(x: number[], y: number[]): number[] | null {
  const n = x.length;
  const s1: Mat3 = [[0, 0, 0], [0, 0, 0], [0, 0, 0]];
  const s2: Mat3 = [[0, 0, 0], [0, 0, 0], [0, 0, 0]];
  const s3: Mat3 = [[0, 0, 0], [0, 0, 0], [0, 0, 0]];
  for (let i = 0; i < n; i++) {
    const d1 = [x[i] * x[i], x[i] * y[i], y[i] * y[i]];
    const d2 = [x[i], y[i], 1];
    for (let r = 0; r < 3; r++) {
      for (let c = 0; c < 3; c++) {
        s1[r][c] += d1[r] * d1[c];
        s2[r][c] += d1[r] * d2[c];
        s3[r][c] += d2[r] * d2[c];
      }
    }
  }

  const s3inv = inv3(s3);
  if (!s3inv) return null;

  // T = -S3⁻¹ S2ᵀ, M = S1 + S2 T
  const t = mul(s3inv, transpose(s2)).map((row) => row.map((v) => -v));
  const m = mul(s2, t).map((row, i) => row.map((v, j) => v + s1[i][j]));

  // Pre-multiply by inv(C1) for the constraint 4ac - b² = 1.
  const constrained: Mat3 = [
    [m[2][0] / 2, m[2][1] / 2, m[2][2] / 2],
    [-m[1][0], -m[1][1], -m[1][2]],
    [m[0][0] / 2, m[0][1] / 2, m[0][2] / 2],
  ];

  for (const lambda of realEigenvalues(constrained)) {
    const vec = eigenvector(constrained, lambda);
    if (!vec) continue;
    if (4 * vec[0] * vec[2] - vec[1] * vec[1] <= 0) continue;
    const tail = t.map((row) => row[0] * vec[0] + row[1] * vec[1] + row[2] * vec[2]);
    return [vec[0], vec[1], vec[2], tail[0], tail[1], tail[2]];
  }
  return null;
}

/** Conic coefficients to centre, semi-major, semi-minor, tilt (radians). */
export function ellipseGeometry(coef: number[]):
  | { cx: number; cy: number; a: number; b: number; angle: number }
  | null {
  const [a, bRaw, c, d, e, f] = coef;
  const b = bRaw / 2;
  const dd = d / 2;
  const ee = e / 2;

  const denom = a * c - b * b;
  if (!Number.isFinite(denom) || Math.abs(denom) < 1e-300) return null;

  const cx = (b * ee - c * dd) / denom;
  const cy = (b * dd - a * ee) / denom;
  const fc = f + dd * cx + ee * cy;

  // Eigen-decomposition of the symmetric 2x2 [[a, b], [b, c]], closed form.
  const mean = (a + c) / 2;
  const diff = Math.sqrt(((a - c) / 2) ** 2 + b * b);
  const l1 = mean + diff;
  const l2 = mean - diff;
  const ax1 = -fc / l1;
  const ax2 = -fc / l2;
  if (!(ax1 > 0) || !(ax2 > 0) || !Number.isFinite(ax1) || !Number.isFinite(ax2)) {
    return null;
  }
  const r1 = Math.sqrt(ax1);
  const r2 = Math.sqrt(ax2);

  // Eigenvector for the axis that turns out to be the major one.
  const vecFor = (l: number): [number, number] =>
    Math.abs(b) > 1e-300 ? [l - c, b] : a <= c ? [1, 0] : [0, 1];
  const [major, minor, majorVec] =
    r1 >= r2 ? [r1, r2, vecFor(l1)] : [r2, r1, vecFor(l2)];

  return {
    cx,
    cy,
    a: major,
    b: minor,
    angle: Math.atan2(majorVec[1], majorVec[0]),
  };
}

/** Ramanujan's second approximation to an ellipse's perimeter. */
export function ramanujan2(a: number, b: number): number {
  if (a + b <= 0) return NaN;
  const h = ((a - b) ** 2) / ((a + b) ** 2);
  return Math.PI * (a + b) * (1 + (3 * h) / (10 + Math.sqrt(4 - 3 * h)));
}

/** Orthonormal (u, v) spanning the plane with the given normal. */
function planeBasis(normal: Vec3): { u: Vec3; v: Vec3; n: Vec3 } {
  const len = Math.hypot(normal[0], normal[1], normal[2]) || 1;
  const n: Vec3 = [normal[0] / len, normal[1] / len, normal[2] / len];
  // Seed against the axis the normal is least aligned with, so the cross
  // product is never taken between near-parallel vectors.
  const abs = [Math.abs(n[0]), Math.abs(n[1]), Math.abs(n[2])];
  const seedIndex = abs.indexOf(Math.min(...abs));
  const seed: Vec3 = [seedIndex === 0 ? 1 : 0, seedIndex === 1 ? 1 : 0, seedIndex === 2 ? 1 : 0];
  const ux = n[1] * seed[2] - n[2] * seed[1];
  const uy = n[2] * seed[0] - n[0] * seed[2];
  const uz = n[0] * seed[1] - n[1] * seed[0];
  const ulen = Math.hypot(ux, uy, uz) || 1;
  const u: Vec3 = [ux / ulen, uy / ulen, uz / ulen];
  const v: Vec3 = [
    n[1] * u[2] - n[2] * u[1],
    n[2] * u[0] - n[0] * u[2],
    n[0] * u[1] - n[1] * u[0],
  ];
  return { u, v, n };
}

/**
 * Fit one cross-section. `positions` and `centre` are scene centimetres.
 *
 * `floorY` is the base of the cloud, used only to flag a plane low enough to be
 * slicing the fabricated skirt rather than reconstructed surface.
 */
export function fitSlice(
  positions: Float32Array,
  centre: Vec3,
  normal: Vec3,
  floorY: number | null = null,
  slabHalfMm: number = SLAB_HALF_MM,
): CrossSectionResult {
  const { u, v, n } = planeBasis(normal);
  const halfCm = slabHalfMm / 10;

  const xs: number[] = [];
  const ys: number[] = [];
  for (let i = 0; i < positions.length; i += 3) {
    const px = positions[i] - centre[0];
    const py = positions[i + 1] - centre[1];
    const pz = positions[i + 2] - centre[2];
    if (Math.abs(px * n[0] + py * n[1] + pz * n[2]) > halfCm) continue;
    xs.push(px * u[0] + py * u[1] + pz * u[2]);
    ys.push(px * v[0] + py * v[1] + pz * v[2]);
  }
  if (xs.length < 12) {
    return { ok: false, reason: `only ${xs.length} points in the slab` };
  }

  // Normalise to unit RMS radius before fitting. The design matrix holds fourth
  // powers of the coordinates, so a slice ~5 cm from its own origin spans eight
  // orders of magnitude and the 3x3 loses digits it does not need to lose. The
  // ellipse's axes scale linearly, so the scale comes straight back out.
  let rms = 0;
  for (let i = 0; i < xs.length; i++) rms += xs[i] * xs[i] + ys[i] * ys[i];
  rms = Math.sqrt(rms / xs.length) || 1;
  const nx = xs.map((value) => value / rms);
  const ny = ys.map((value) => value / rms);

  const conic = fitEllipseDirect(nx, ny);
  if (!conic) return { ok: false, reason: "no elliptical solution — the slice is not closed" };
  const geom = ellipseGeometry(conic);
  if (!geom) return { ok: false, reason: "the conic solved to a hyperbola" };

  const aCm = geom.a * rms;
  const bCm = geom.b * rms;
  const cx = geom.cx * rms;
  const cy = geom.cy * rms;

  const phi: number[] = [];
  const radius: number[] = [];
  let residSq = 0;
  for (let i = 0; i < xs.length; i++) {
    const dx = xs[i] - cx;
    const dy = ys[i] - cy;
    phi.push(((Math.atan2(dy, dx) % (2 * Math.PI)) + 2 * Math.PI) % (2 * Math.PI));
    const r = Math.hypot(dx, dy);
    radius.push(r);
    // Radial residual against the fitted ellipse, in millimetres. An algebraic
    // fit always returns something; this is what says whether "ellipse"
    // described the slice at all.
    const th = Math.atan2(dy, dx) - geom.angle;
    const rEll =
      1 / Math.sqrt((Math.cos(th) / aCm) ** 2 + (Math.sin(th) / bCm) ** 2);
    residSq += ((r - rEll) * 10) ** 2;
  }

  // Angular coverage. A conic fit to a partial arc still returns an ellipse —
  // it extrapolates the missing side, with a *better* residual than the truth.
  // This is the only diagnostic that catches a limb whose back was never
  // reconstructed.
  const sorted = [...phi].sort((p, q) => p - q);
  let maxGap = sorted[0] + 2 * Math.PI - sorted[sorted.length - 1];
  for (let i = 1; i < sorted.length; i++) {
    maxGap = Math.max(maxGap, sorted[i] - sorted[i - 1]);
  }
  const sectors = new Set(sorted.map((p) => Math.floor((p / (2 * Math.PI)) * 72) % 72));

  // Independent cross-check: median radius per wedge, joined into a closed
  // polygon. It assumes no shape model, so a large disagreement means the
  // cross-section is not elliptical — not that the fit failed.
  const wedges = 36;
  const buckets: number[][] = Array.from({ length: wedges }, () => []);
  for (let i = 0; i < phi.length; i++) {
    buckets[Math.floor((phi[i] / (2 * Math.PI)) * wedges) % wedges].push(radius[i]);
  }
  const ring: Array<[number, number]> = [];
  buckets.forEach((values, k) => {
    if (!values.length) return;
    values.sort((p, q) => p - q);
    // Mean of the two middle values on an even count, matching numpy.median —
    // taking the upper one instead moved the polygon cross-check by 0.26% on a
    // real slice, which is the same size as the disagreement it exists to
    // report.
    const mid = values.length >> 1;
    const median =
      values.length % 2 ? values[mid] : (values[mid - 1] + values[mid]) / 2;
    const centreAngle = ((k + 0.5) / wedges) * 2 * Math.PI;
    ring.push([median * Math.cos(centreAngle), median * Math.sin(centreAngle)]);
  });
  let polygonCm = 0;
  for (let i = 0; i < ring.length; i++) {
    const [x0, y0] = ring[i];
    const [x1, y1] = ring[(i + 1) % ring.length];
    polygonCm += Math.hypot(x1 - x0, y1 - y0);
  }

  const maxGapDeg = (maxGap * 180) / Math.PI;
  return {
    ok: true,
    circumferenceCm: ramanujan2(aCm, bCm),
    aCm,
    bCm,
    polygonCm,
    nSlab: xs.length,
    coverage: sectors.size / 72,
    maxGapDeg,
    residRmsMm: Math.sqrt(residSq / xs.length),
    partialArc: maxGapDeg > MAX_ARC_GAP_DEG,
    nearFloor: floorY !== null && centre[1] - floorY < FLOOR_GUARD_CM,
  };
}
