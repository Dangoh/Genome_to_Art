#!/usr/bin/env python3
"""
fasta_painter.py

FASTA-driven abstract paintings where randomness is deterministically
driven by nucleotide content (k-mers). Same FASTA + same args => same image.

Features:
- Many --style and --theme options
- --all: run selected styles × selected themes
- --prefix + --outdir: auto-named outputs: <prefix><sep><style><sep><theme>.png
- Deterministic per-combo seeding (style+theme mixed into seed)
- PRNG mixer uses Python ints -> no NumPy overflow warnings
- Psychedelic theme family (psy_*) + --themes psy
- --styles and --themes selectors for --all
- --sep to customize output filenames
- --fill: boost coverage (less white) with a single flag
- Circular-genome inspired styles: plasmid, chromosome, mandala, radial_rings

Styles:
  papercut      : layered paper-cut bands
  ribbons       : sweeping ribbon loop (papercut-like)
  swirl         : spiral loop with stacked layers (papercut-like)
  fluid         : smooth flowing strokes (acrylic/smoke)
  fluid_bold    : like fluid but thicker, more opaque
  fluid_wispy   : like fluid but thinner, more translucent
  turbulence    : more chaotic flow field, energetic
  streamlines   : many thin streamlines (airy, filamentous)
  bloom         : clustered releases (center-biased)
  storm         : strong directional shearing + vortices
  doodle        : dense curvy doodles
  ink           : monochrome ink streamlines (forces ink/mono feel)
  topo          : topographic contour lines from a scalar field
  marble        : marbled contour bands (contour-like ribbons)

Marble-adjacent contour/texture styles:
  lava          : thick molten contour bands
  tides         : flowing layered contour waves
  quartz        : crystalline contour facets
  cells         : organic cellular Voronoi-like blobs
  nebula        : smooth cloudy scalar diffusion (raster pigment fog)

Additional doodle-friendly styles:
  scribble      : dense fast pen loops
  hatch         : cross-hatch streamlines
  stipple       : pointillist field
  worms         : short curvy worms

Circular-genome inspired styles:
  plasmid       : concentric circular arcs + bead dots (plasmid map vibe)
  chromosome    : circular ring-flow strokes around a center (circular DNA motion)
  mandala       : radial symmetry rings + layered contours (psychedelic mandala)
  radial_rings  : dense circular contour rings (genome-ring / circos-y feel)

Themes:
  vivid, pastel, ocean, sunset, forest, ice, earth, berry, neon,
  mono, ink, fire, aurora, desert,
  psy_acid, psy_uv, psy_candy, psy_cyber, psy_rainbow, psy_lava, psy_jungle, psy_galaxy

Examples:
  python fasta_painter.py --fasta WHO_F_2024.fasta --style fluid --theme vivid --outdir out --prefix WHO_F --fill
  python fasta_painter.py --fasta WHO_F_2024.fasta --style chromosome --theme psy_uv --outdir out --prefix WHO_F --fill
  python fasta_painter.py --fasta WHO_F_2024.fasta --all --themes psy --styles plasmid,chromosome,mandala --outdir out --prefix WHO_F --fill
"""

import argparse
import hashlib
import math
import os
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import matplotlib.pyplot as plt


# ----------------------------
# FASTA parsing + sequence prep
# ----------------------------

def read_fasta(path: str) -> str:
    seq_chunks = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                continue
            seq_chunks.append(line)
    seq = "".join(seq_chunks).upper()
    seq = "".join([c for c in seq if c in "ACGT"])  # drop ambiguous bases
    return seq


# ----------------------------
# Deterministic RNG driven by kmers
# ----------------------------

def sha256_int(data: bytes) -> int:
    return int.from_bytes(hashlib.sha256(data).digest(), byteorder="big", signed=False)

def blake2_u64(data: bytes) -> np.uint64:
    h = hashlib.blake2b(data, digest_size=8).digest()
    return np.uint64(int.from_bytes(h, "big", signed=False))

def kmer_to_u64_hash(kmer: str) -> np.uint64:
    return blake2_u64(kmer.encode("ascii"))

def iter_kmers(seq: str, k: int, stride: int) -> np.ndarray:
    if len(seq) < k:
        return np.array([], dtype=np.uint64)
    stride = max(1, stride)
    out = np.empty(((len(seq) - k) // stride) + 1, dtype=np.uint64)
    j = 0
    for i in range(0, len(seq) - k + 1, stride):
        out[j] = kmer_to_u64_hash(seq[i:i+k])
        j += 1
    return out[:j]

@dataclass
class KmerRNG:
    kmers: np.ndarray
    state: np.uint64
    idx: int = 0

    @staticmethod
    def from_sequence(seq: str, k: int, stride: int, user_seed: int = 0) -> "KmerRNG":
        km = iter_kmers(seq, k, stride)
        if km.size == 0:
            h = sha256_int(seq.encode("utf-8"))
            km = np.array([(h >> (i * 16)) & 0xFFFFFFFFFFFFFFFF for i in range(32)], dtype=np.uint64)

        global_seed = sha256_int(seq.encode("utf-8")) ^ (user_seed & ((1 << 256) - 1))
        st = np.uint64(global_seed & 0xFFFFFFFFFFFFFFFF)
        return KmerRNG(km, st, 0)

    def _mix(self, x: np.uint64) -> np.uint64:
        # SplitMix64-like mixing, using Python ints and masking -> no NumPy overflow warnings
        z = (int(self.state) + 0x9E3779B97F4A7C15 + int(x)) & 0xFFFFFFFFFFFFFFFF
        z ^= (z >> 30)
        z = (z * 0xBF58476D1CE4E5B9) & 0xFFFFFFFFFFFFFFFF
        z ^= (z >> 27)
        z = (z * 0x94D049BB133111EB) & 0xFFFFFFFFFFFFFFFF
        z ^= (z >> 31)
        self.state = np.uint64(z)
        return self.state

    def u64(self) -> np.uint64:
        x = self.kmers[self.idx % self.kmers.size]
        self.idx += 1
        return self._mix(x)

    def uniform(self, a: float = 0.0, b: float = 1.0) -> float:
        r = int(self.u64() >> np.uint64(11)) / float(1 << 53)
        return a + (b - a) * r

    def normal(self, mu: float = 0.0, sigma: float = 1.0) -> float:
        u1 = max(1e-12, self.uniform())
        u2 = self.uniform()
        z = math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)
        return mu + sigma * z

    def choice(self, n: int) -> int:
        return int(self.u64() % np.uint64(n))


# ----------------------------
# Color utilities
# ----------------------------

def hsv_to_rgb(h, s, v):
    import colorsys
    return colorsys.hsv_to_rgb(h % 1.0, np.clip(s, 0, 1), np.clip(v, 0, 1))

def make_palette(rng: KmerRNG, n: int, theme: str) -> List[Tuple[float, float, float]]:
    colors = []

    def ramp(base_h, span, s_lo, s_hi, v_lo, v_hi, jitter=0.02):
        for i in range(n):
            h = base_h + (i / max(1, n - 1)) * span + rng.normal(0, jitter)
            s = rng.uniform(s_lo, s_hi)
            v = rng.uniform(v_lo, v_hi)
            colors.append(hsv_to_rgb(h, s, v))

    # classic themes
    if theme == "ocean":
        base_h = rng.uniform(0.46, 0.58)
        ramp(base_h, span=0.18, s_lo=0.65, s_hi=0.95, v_lo=0.70, v_hi=1.00, jitter=0.015)
    elif theme == "sunset":
        base_h = rng.uniform(0.00, 0.10)
        ramp(base_h, span=0.22, s_lo=0.70, s_hi=0.98, v_lo=0.70, v_hi=1.00, jitter=0.015)
    elif theme == "pastel":
        base_h = rng.uniform(0.00, 1.00)
        ramp(base_h, span=1.00, s_lo=0.25, s_hi=0.55, v_lo=0.88, v_hi=1.00, jitter=0.02)
    elif theme == "vivid":
        base_h = rng.uniform(0.00, 1.00)
        ramp(base_h, span=1.00, s_lo=0.75, s_hi=1.00, v_lo=0.75, v_hi=1.00, jitter=0.012)
    elif theme == "forest":
        base_h = rng.uniform(0.25, 0.42)
        ramp(base_h, span=0.20, s_lo=0.55, s_hi=0.95, v_lo=0.55, v_hi=0.95, jitter=0.012)
    elif theme == "ice":
        base_h = rng.uniform(0.52, 0.66)
        ramp(base_h, span=0.18, s_lo=0.25, s_hi=0.55, v_lo=0.90, v_hi=1.00, jitter=0.012)
    elif theme == "earth":
        base_h = rng.uniform(0.05, 0.14)
        ramp(base_h, span=0.18, s_lo=0.45, s_hi=0.85, v_lo=0.45, v_hi=0.90, jitter=0.012)
    elif theme == "berry":
        base_h = rng.uniform(0.80, 0.95)
        ramp(base_h, span=0.25, s_lo=0.55, s_hi=0.95, v_lo=0.60, v_hi=1.00, jitter=0.015)
    elif theme == "neon":
        base_h = rng.uniform(0.00, 1.00)
        ramp(base_h, span=1.00, s_lo=0.90, s_hi=1.00, v_lo=0.90, v_hi=1.00, jitter=0.02)
    elif theme == "mono":
        for _ in range(n):
            v = rng.uniform(0.15, 0.95)
            colors.append((v, v, v))
    elif theme == "ink":
        for _ in range(n):
            v = rng.uniform(0.02, 0.25) if rng.uniform() < 0.9 else rng.uniform(0.25, 0.55)
            colors.append((v, v, v))
    elif theme == "fire":
        base_h = rng.uniform(0.00, 0.08)
        ramp(base_h, span=0.15, s_lo=0.75, s_hi=1.00, v_lo=0.60, v_hi=1.00, jitter=0.010)
    elif theme == "aurora":
        base_h = rng.uniform(0.40, 0.80)
        ramp(base_h, span=0.45, s_lo=0.55, s_hi=0.95, v_lo=0.75, v_hi=1.00, jitter=0.02)
    elif theme == "desert":
        base_h = rng.uniform(0.08, 0.18)
        ramp(base_h, span=0.18, s_lo=0.30, s_hi=0.70, v_lo=0.85, v_hi=1.00, jitter=0.012)

    # psychedelic family (curated vivid)
    elif theme == "psy_acid":
        for _ in range(n):
            h = rng.uniform(0.18, 0.35) if rng.uniform() < 0.72 else rng.uniform(0.82, 0.98)
            colors.append(hsv_to_rgb(h, rng.uniform(0.85, 1.00), rng.uniform(0.80, 1.00)))
    elif theme == "psy_uv":
        for _ in range(n):
            h = rng.uniform(0.62, 0.88) if rng.uniform() < 0.80 else rng.uniform(0.20, 0.32)
            colors.append(hsv_to_rgb(h, rng.uniform(0.80, 1.00), rng.uniform(0.75, 1.00)))
    elif theme == "psy_candy":
        base_h = rng.uniform(0.00, 1.00)
        for i in range(n):
            h = base_h + rng.uniform(-0.18, 0.18) + (i / max(1, n - 1)) * 0.60
            colors.append(hsv_to_rgb(h, rng.uniform(0.70, 0.95), rng.uniform(0.88, 1.00)))
    elif theme == "psy_cyber":
        for _ in range(n):
            r = rng.uniform()
            if r < 0.40:
                h = rng.uniform(0.50, 0.58)
            elif r < 0.85:
                h = rng.uniform(0.82, 0.95)
            else:
                h = rng.uniform(0.10, 0.16)
            colors.append(hsv_to_rgb(h, rng.uniform(0.85, 1.00), rng.uniform(0.75, 1.00)))
    elif theme == "psy_rainbow":
        base_h = rng.uniform(0.00, 1.00)
        for i in range(n):
            h = base_h + (i / max(1, n - 1)) * 1.00 + rng.normal(0, 0.01)
            colors.append(hsv_to_rgb(h, rng.uniform(0.80, 1.00), rng.uniform(0.78, 1.00)))
    elif theme == "psy_lava":
        for _ in range(n):
            h = rng.uniform(0.00, 0.12) if rng.uniform() < 0.86 else rng.uniform(0.48, 0.58)
            colors.append(hsv_to_rgb(h, rng.uniform(0.85, 1.00), rng.uniform(0.70, 1.00)))
    elif theme == "psy_jungle":
        for _ in range(n):
            r = rng.uniform()
            if r < 0.65:
                h = rng.uniform(0.20, 0.40)
            elif r < 0.82:
                h = rng.uniform(0.04, 0.10)
            else:
                h = rng.uniform(0.78, 0.92)
            colors.append(hsv_to_rgb(h, rng.uniform(0.80, 1.00), rng.uniform(0.70, 1.00)))
    elif theme == "psy_galaxy":
        for _ in range(n):
            h = rng.uniform(0.62, 0.78) if rng.uniform() < 0.82 else rng.uniform(0.84, 0.96)
            colors.append(hsv_to_rgb(h, rng.uniform(0.75, 1.00), rng.uniform(0.65, 1.00)))
    else:
        raise ValueError(f"Unknown theme: {theme}")

    return colors


# ----------------------------
# Plot helpers
# ----------------------------

def setup_figure(width: float, height: float, dpi: int, background: str):
    fig = plt.figure(figsize=(width, height), dpi=dpi)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_facecolor(background)
    fig.patch.set_facecolor(background)
    ax.set_aspect("equal")
    ax.axis("off")
    return fig, ax

def finalize_and_save(fig, ax, out: str, pad_frac: float):
    ax.relim()
    ax.autoscale_view()
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    pad = float(pad_frac) * max(x1 - x0, y1 - y0)
    ax.set_xlim(x0 - pad, x1 + pad)
    ax.set_ylim(y0 - pad, y1 + pad)
    fig.savefig(out, dpi=fig.dpi, facecolor=fig.get_facecolor(), bbox_inches="tight", pad_inches=0)
    plt.close(fig)


# ----------------------------
# Flow field for stroke styles
# ----------------------------

@dataclass
class FlowField:
    a: np.ndarray
    kx: np.ndarray
    ky: np.ndarray
    phx: np.ndarray
    phy: np.ndarray
    gain: float

    def vel(self, x: float, y: float) -> Tuple[float, float]:
        sx = np.sin(self.kx * x + self.phx)
        cx = np.cos(self.kx * x + self.phx)
        sy = np.sin(self.ky * y + self.phy)
        cy = np.cos(self.ky * y + self.phy)

        dfdx = np.sum(self.a * self.kx * cx * cy)
        dfdy = np.sum(self.a * (-self.ky) * sx * sy)
        vx = dfdy
        vy = -dfdx
        return (self.gain * float(vx), self.gain * float(vy))

def make_flow_field(rng: KmerRNG, components: int, gain: float, chaotic: float = 0.0) -> FlowField:
    a = np.array([rng.uniform(0.6, 1.2) * (0.88 ** i) for i in range(components)], dtype=float)

    k_scale = 1.0 + 1.8 * chaotic
    jitter = 0.0 + 1.2 * chaotic

    kx = np.array([rng.uniform(0.8, 3.2) * k_scale * (1.18 ** (i * 0.22)) for i in range(components)], dtype=float)
    ky = np.array([rng.uniform(0.8, 3.2) * k_scale * (1.18 ** (i * 0.22)) for i in range(components)], dtype=float)
    phx = np.array([rng.uniform(0, 2 * math.pi) + rng.normal(0, jitter) for _ in range(components)], dtype=float)
    phy = np.array([rng.uniform(0, 2 * math.pi) + rng.normal(0, jitter) for _ in range(components)], dtype=float)

    for i in range(components):
        if rng.uniform() < 0.5:
            kx[i] *= -1.0
        if rng.uniform() < 0.5:
            ky[i] *= -1.0

    return FlowField(a=a, kx=kx, ky=ky, phx=phx, phy=phy, gain=gain)

def smooth_polyline(pts: np.ndarray, k: int = 9) -> np.ndarray:
    if pts.shape[0] < k:
        return pts
    kernel = np.ones(k) / k
    xs = np.convolve(pts[:, 0], kernel, mode="same")
    ys = np.convolve(pts[:, 1], kernel, mode="same")
    return np.column_stack([xs, ys])

def draw_flow_strokes(
    ax,
    rng: KmerRNG,
    palette: List[Tuple[float, float, float]],
    strokes: int,
    steps: int,
    step_size: float,
    field_components: int,
    field_gain: float,
    alpha: float,
    lw_min: float,
    lw_max: float,
    fill_box: float,
    chaotic: float = 0.0,
    jitter: float = 0.0007,
    directional: float = 0.0,
    mode: str = "line",  # line | dots
    dot_size: float = 4.0,
):
    field = make_flow_field(rng, components=field_components, gain=field_gain, chaotic=chaotic)

    def vel(x, y):
        vx, vy = field.vel(x, y)
        if directional != 0.0:
            vx += directional * 0.45 * (0.5 + 0.5 * math.tanh(2.2 * y))
        return vx, vy

    for i in range(strokes):
        x0 = rng.uniform(-fill_box, fill_box)
        y0 = rng.uniform(-fill_box, fill_box)

        pts = np.zeros((steps, 2), dtype=float)
        x, y = x0, y0
        m = 0
        for _ in range(steps):
            pts[m] = (x, y)
            vx, vy = vel(x, y)

            xm = x + 0.5 * step_size * vx
            ym = y + 0.5 * step_size * vy
            vxm, vym = vel(xm, ym)

            x += step_size * vxm
            y += step_size * vym

            x += rng.normal(0, jitter)
            y += rng.normal(0, jitter)

            m += 1
            if abs(x) > 1.10 or abs(y) > 1.10:
                break

        pts = pts[:m]
        if pts.shape[0] < 10:
            continue

        pts = smooth_polyline(pts, k=9)
        col = palette[i % len(palette)]

        if mode == "dots":
            take = max(8, int(pts.shape[0] * 0.33))
            idxs = np.linspace(0, pts.shape[0] - 1, take).astype(int)
            ax.scatter(pts[idxs, 0], pts[idxs, 1], s=dot_size, alpha=alpha, c=[col], linewidths=0, zorder=2)
            continue

        phase = i / max(12.0, rng.uniform(18, 55))
        lw = lw_min + (lw_max - lw_min) * (0.50 + 0.50 * math.sin(phase))

        ax.plot(
            pts[:, 0], pts[:, 1],
            linewidth=lw,
            alpha=alpha,
            solid_capstyle="round",
            solid_joinstyle="round",
            color=col,
            zorder=2,
        )


# ----------------------------
# Scalar fields + contour/raster rendering
# ----------------------------

def make_scalar_field(rng: KmerRNG, components: int, chaotic: float = 0.0):
    a = np.array([rng.uniform(0.6, 1.2) * (0.90 ** i) for i in range(components)], dtype=float)
    k_scale = 1.0 + 2.2 * chaotic
    kx = np.array([rng.uniform(0.8, 3.0) * k_scale * (1.14 ** (i * 0.25)) for i in range(components)], dtype=float)
    ky = np.array([rng.uniform(0.8, 3.0) * k_scale * (1.14 ** (i * 0.25)) for i in range(components)], dtype=float)
    phx = np.array([rng.uniform(0, 2 * math.pi) for _ in range(components)], dtype=float)
    phy = np.array([rng.uniform(0, 2 * math.pi) for _ in range(components)], dtype=float)

    def f(x, y):
        return np.sum(a * np.sin(kx * x + phx) * np.cos(ky * y + phy))
    return f

def scalar_grid(f, n: int = 420, span: float = 1.05):
    xs = np.linspace(-span, span, n)
    ys = np.linspace(-span, span, n)
    X, Y = np.meshgrid(xs, ys)
    Z = np.vectorize(f)(X, Y)
    return X, Y, Z

def draw_contours(ax, X, Y, Z, palette, levels: int, lw: float, alpha: float):
    cs = ax.contour(X, Y, Z, levels=levels, linewidths=lw, alpha=alpha)
    for i, c in enumerate(cs.collections):
        c.set_color([palette[i % len(palette)]])
        c.set_capstyle("round")
        c.set_joinstyle("round")

def draw_filled_contours(ax, X, Y, Z, palette, levels: int, alpha: float):
    lvl_vals = np.linspace(Z.min(), Z.max(), levels)
    cs = ax.contourf(X, Y, Z, levels=lvl_vals, alpha=alpha)
    for i, c in enumerate(cs.collections):
        c.set_color([palette[i % len(palette)]])
        c.set_edgecolor("none")

def box_blur(img: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return img
    r = int(radius)
    pad = np.pad(img, ((0, 0), (r, r)), mode="edge")
    c = np.cumsum(pad, axis=1)
    out = (c[:, 2*r:] - c[:, :-2*r]) / float(2*r)
    pad2 = np.pad(out, ((r, r), (0, 0)), mode="edge")
    c2 = np.cumsum(pad2, axis=0)
    out2 = (c2[2*r:, :] - c2[:-2*r, :]) / float(2*r)
    return out2

def palette_map(Z: np.ndarray, palette: List[Tuple[float, float, float]]):
    z = (Z - Z.min()) / (Z.max() - Z.min() + 1e-12)
    idx = z * (len(palette) - 1)
    lo = np.floor(idx).astype(int)
    hi = np.clip(lo + 1, 0, len(palette) - 1)
    t = (idx - lo)[..., None]

    pal = np.array(palette, dtype=float)
    rgb = (1 - t) * pal[lo] + t * pal[hi]
    return np.clip(rgb, 0, 1)


# ----------------------------
# Papercut-like fallback
# ----------------------------

def draw_papercut_like(ax, rng: KmerRNG, palette: List[Tuple[float, float, float]],
                       layers: int, band: float, alpha: float):
    for i in range(layers):
        t = np.linspace(0, 2 * math.pi, 900)
        a = 0.18 + 0.85 * (i / max(1, layers - 1))
        b = 0.22 + 0.80 * (i / max(1, layers - 1))
        wob = 0.02 + 0.06 * rng.uniform()

        x = a * np.cos(t) * (1.0 + wob * np.sin(3*t + rng.uniform(0, 2*math.pi)))
        y = b * np.sin(t) * (1.0 + wob * np.cos(4*t + rng.uniform(0, 2*math.pi)))

        x += rng.normal(0, 0.02)
        y += rng.normal(0, 0.02)

        ax.plot(x, y, linewidth=band * 120.0, alpha=alpha,
                color=palette[i % len(palette)], solid_capstyle="round")


# ----------------------------
# Marble-adjacent styles
# ----------------------------

def draw_marble(ax, rng: KmerRNG, palette, bands: int, lw_min: float, lw_max: float, alpha: float):
    f = make_scalar_field(rng, components=14, chaotic=0.25)
    X, Y, Z = scalar_grid(f, n=360, span=1.05)
    cs = ax.contour(X, Y, Z, levels=bands)
    for i, c in enumerate(cs.collections):
        c.set_color([palette[i % len(palette)]])
        c.set_alpha(alpha)
        c.set_linewidth(lw_min + (lw_max - lw_min) * (i / max(1, bands - 1)))
        c.set_capstyle("round")
        c.set_joinstyle("round")

def draw_lava(ax, rng: KmerRNG, palette, density: float, alpha: float, lw_min: float, lw_max: float):
    f = make_scalar_field(rng, components=16, chaotic=0.55)
    X, Y, Z = scalar_grid(f, n=360, span=1.05)
    bands = max(10, int(18 * density))
    cs = ax.contour(X, Y, Z, levels=bands)
    for i, c in enumerate(cs.collections):
        c.set_color([palette[i % len(palette)]])
        c.set_alpha(alpha)
        t = i / max(1, bands - 1)
        width = (lw_max * 1.8) * (0.35 + 1.65 * t)
        width *= (0.85 + 0.25 * rng.uniform())
        c.set_linewidth(max(lw_min * 1.2, width))
        c.set_capstyle("round")
        c.set_joinstyle("round")

def draw_tides(ax, rng: KmerRNG, palette, density: float, alpha: float, lw: float):
    f = make_scalar_field(rng, components=10, chaotic=0.10)
    X, Y, Z = scalar_grid(f, n=420, span=1.05)
    wx = rng.uniform(0.6, 1.6)
    wy = rng.uniform(0.6, 1.6)
    Z2 = Z + 0.35 * np.sin(wx * X + rng.uniform(0, 2*math.pi)) + 0.35 * np.cos(wy * Y + rng.uniform(0, 2*math.pi))
    levels = max(24, int(44 * density))
    draw_contours(ax, X, Y, Z2, palette, levels=levels, lw=lw, alpha=alpha)

def draw_quartz(ax, rng: KmerRNG, palette, density: float, alpha: float, lw: float):
    n = 420
    span = 1.05
    xs = np.linspace(-span, span, n)
    ys = np.linspace(-span, span, n)
    X, Y = np.meshgrid(xs, ys)

    planes = max(14, int(26 * density))
    Z = np.full_like(X, -1e9, dtype=float)
    for _ in range(planes):
        a = rng.uniform(-2.0, 2.0)
        b = rng.uniform(-2.0, 2.0)
        c = rng.uniform(-1.0, 1.0)
        Zi = a * X + b * Y + c
        Z = np.maximum(Z, Zi)

    Z = box_blur(Z, radius=6)
    levels = max(18, int(34 * density))
    draw_contours(ax, X, Y, Z, palette, levels=levels, lw=lw, alpha=alpha)

def draw_cells(ax, rng: KmerRNG, palette, density: float, alpha: float):
    n = 380
    span = 1.05
    xs = np.linspace(-span, span, n)
    ys = np.linspace(-span, span, n)
    X, Y = np.meshgrid(xs, ys)

    sites = max(26, int(60 * density))
    pts = np.array([[rng.uniform(-span, span), rng.uniform(-span, span)] for _ in range(sites)], dtype=float)

    dx = X[..., None] - pts[:, 0]
    dy = Y[..., None] - pts[:, 1]
    d2 = dx*dx + dy*dy

    d2_sorted = np.sort(d2, axis=2)
    f1 = np.sqrt(d2_sorted[..., 0])
    f2 = np.sqrt(d2_sorted[..., 1])

    Z = (f2 - f1)
    Z = box_blur(Z, radius=5)
    Z = np.tanh(3.2 * (Z - np.median(Z)))

    draw_filled_contours(ax, X, Y, Z, palette, levels=max(10, int(16 * density)), alpha=alpha)
    draw_contours(ax, X, Y, Z, palette, levels=max(10, int(18 * density)), lw=1.0, alpha=min(0.55, alpha))

def draw_nebula(ax, rng: KmerRNG, palette, density: float, alpha: float):
    f = make_scalar_field(rng, components=14, chaotic=0.35)
    X, Y, Z = scalar_grid(f, n=520, span=1.05)

    for _ in range(3):
        f2 = make_scalar_field(rng, components=10, chaotic=0.55)
        _, _, Z2 = scalar_grid(f2, n=520, span=1.05)
        Z = 0.75 * Z + 0.25 * Z2

    blur_r = int(10 + 10 * density)
    Z = box_blur(Z, radius=blur_r)
    Z = box_blur(Z, radius=max(2, blur_r // 2))

    rgb = palette_map(Z, palette)
    ax.imshow(rgb, extent=[-1.05, 1.05, -1.05, 1.05], origin="lower", alpha=alpha, interpolation="bilinear", zorder=1)


# ----------------------------
# Circular genome inspired styles
# ----------------------------

def draw_radial_rings(ax, rng: KmerRNG, palette, density: float, alpha: float, lw: float):
    # Use scalar field in polar coordinates so contours become rings
    n = 520
    span = 1.05
    xs = np.linspace(-span, span, n)
    ys = np.linspace(-span, span, n)
    X, Y = np.meshgrid(xs, ys)

    R = np.sqrt(X*X + Y*Y) + 1e-12
    T = np.arctan2(Y, X)

    # FASTA-driven ring harmonics
    a1 = rng.uniform(3.0, 9.0)
    a2 = rng.uniform(2.0, 7.0)
    a3 = rng.uniform(4.0, 11.0)
    w1 = rng.uniform(2.0, 7.0)
    w2 = rng.uniform(5.0, 13.0)

    Z = (
        np.sin(a1 * R + rng.uniform(0, 2*math.pi)) +
        0.65 * np.cos(a2 * R + 0.9*np.sin(w1*T + rng.uniform(0, 2*math.pi))) +
        0.35 * np.sin(a3 * R + 0.7*np.cos(w2*T + rng.uniform(0, 2*math.pi)))
    )

    # soften center a bit
    Z *= np.tanh(2.2 * (1.1 - R))

    levels = max(30, int(70 * density))
    draw_contours(ax, X, Y, Z, palette, levels=levels, lw=lw, alpha=alpha)

def draw_mandala(ax, rng: KmerRNG, palette, density: float, alpha: float):
    # Filled polar contours with k-fold radial symmetry
    n = 520
    span = 1.05
    xs = np.linspace(-span, span, n)
    ys = np.linspace(-span, span, n)
    X, Y = np.meshgrid(xs, ys)

    R = np.sqrt(X*X + Y*Y) + 1e-12
    T = np.arctan2(Y, X)

    kfold = rng.choice(9) + 5  # 5..13
    k2 = rng.choice(7) + 3     # 3..9

    Z = (
        np.cos((kfold) * T + rng.uniform(0, 2*math.pi)) * (1.0 - 0.75*R) +
        0.85*np.sin((k2) * T + rng.uniform(0, 2*math.pi)) * (0.3 + 0.7*np.cos(4.2*R + rng.uniform(0, 2*math.pi))) +
        0.70*np.cos((6.0 + rng.uniform(-1, 1)) * R + rng.uniform(0, 2*math.pi))
    )

    # radial envelope
    Z *= np.exp(-0.6 * (R * 2.2)**2)

    Z = box_blur(Z, radius=6)
    Z = np.tanh(2.4 * Z)

    # filled for that psychedelic poster look + edges
    draw_filled_contours(ax, X, Y, Z, palette, levels=max(12, int(22 * density)), alpha=min(0.95, alpha + 0.20))
    draw_contours(ax, X, Y, Z, palette, levels=max(14, int(28 * density)), lw=1.2, alpha=min(0.55, alpha))

def draw_plasmid(ax, rng: KmerRNG, palette, density: float, alpha: float, lw_min: float, lw_max: float):
    # Concentric circles and arcs with occasional "features"
    rings = max(10, int(26 * density))
    base_r0 = 0.10
    base_r1 = 1.02

    for i in range(rings):
        t = np.linspace(0, 2 * math.pi, 900)
        r = base_r0 + (base_r1 - base_r0) * (i / max(1, rings - 1))
        # wobble radius to look organic
        wob = (0.006 + 0.018 * rng.uniform()) * (1.0 - 0.65*(i / max(1, rings - 1)))
        r_t = r * (1.0 + wob * np.sin((3 + rng.choice(7)) * t + rng.uniform(0, 2*math.pi)))

        x = r_t * np.cos(t)
        y = r_t * np.sin(t)

        col = palette[i % len(palette)]
        lw = lw_min + (lw_max - lw_min) * (0.15 + 0.85 * (i / max(1, rings - 1)))
        lw *= (0.70 + 0.40 * rng.uniform())

        # draw as partial arc sometimes
        if rng.uniform() < 0.55:
            a0 = rng.uniform(0, 2*math.pi)
            a1 = a0 + rng.uniform(1.2, 5.4)
            mask = (t >= a0) & (t <= a1) if a0 < a1 else ((t >= a0) | (t <= (a1 % (2*math.pi))))
            ax.plot(x[mask], y[mask], linewidth=lw, alpha=alpha, color=col, solid_capstyle="round", zorder=2)
        else:
            ax.plot(x, y, linewidth=lw, alpha=alpha, color=col, solid_capstyle="round", zorder=2)

        # add feature "beads" on some rings
        if rng.uniform() < 0.40:
            m = max(8, int(18 * density))
            angs = np.sort(np.array([rng.uniform(0, 2*math.pi) for _ in range(m)]))
            ptsx = (r * np.cos(angs))
            ptsy = (r * np.sin(angs))
            s = (10 + 35 * (i / max(1, rings - 1))) * (0.6 + 0.7 * rng.uniform())
            ax.scatter(ptsx, ptsy, s=s, c=[col], alpha=min(0.85, alpha + 0.15), linewidths=0, zorder=3)

def draw_chromosome(ax, rng: KmerRNG, palette, density: float, alpha: float, lw_min: float, lw_max: float):
    # Ring-flow strokes: start points on annulus, move with tangential + curl noise
    strokes = max(1800, int(5200 * density))
    steps = max(160, int(280 * density))
    step_size = 0.010 + 0.006 * rng.uniform()

    # build a 2D curl-ish field and then add a tangential term
    field = make_flow_field(rng, components=18, gain=0.95 + 0.25*rng.uniform(), chaotic=0.25)

    for i in range(strokes):
        # start on an annulus
        r0 = rng.uniform(0.10, 1.02)
        th0 = rng.uniform(0, 2*math.pi)
        x = r0 * math.cos(th0)
        y = r0 * math.sin(th0)

        pts = np.zeros((steps, 2), dtype=float)
        m = 0
        for _ in range(steps):
            pts[m] = (x, y)
            m += 1

            vx, vy = field.vel(x, y)

            # add tangential "go around" motion (circular genome feel)
            r = math.sqrt(x*x + y*y) + 1e-12
            tx, ty = (-y / r, x / r)  # tangential unit vector
            tang = 0.55 + 0.55 * rng.uniform()
            vx = 0.55 * vx + tang * tx
            vy = 0.55 * vy + tang * ty

            # RK2
            xm = x + 0.5 * step_size * vx
            ym = y + 0.5 * step_size * vy
            vxm, vym = field.vel(xm, ym)
            rm = math.sqrt(xm*xm + ym*ym) + 1e-12
            tmx, tmy = (-ym / rm, xm / rm)
            vxm = 0.55 * vxm + tang * tmx
            vym = 0.55 * vym + tang * tmy

            x += step_size * vxm + rng.normal(0, 0.0006)
            y += step_size * vym + rng.normal(0, 0.0006)

            if abs(x) > 1.10 or abs(y) > 1.10:
                break

        pts = pts[:m]
        if pts.shape[0] < 12:
            continue

        pts = smooth_polyline(pts, k=9)
        col = palette[i % len(palette)]
        phase = i / max(10.0, rng.uniform(14, 44))
        lw = lw_min + (lw_max - lw_min) * (0.40 + 0.60 * math.sin(phase) * 0.5 + 0.30)
        ax.plot(pts[:, 0], pts[:, 1], linewidth=lw, alpha=alpha, color=col,
                solid_capstyle="round", solid_joinstyle="round", zorder=2)


# ----------------------------
# Registry
# ----------------------------

STYLES = [
    "papercut", "ribbons", "swirl",
    "fluid", "fluid_bold", "fluid_wispy",
    "turbulence", "streamlines", "bloom", "storm",
    "doodle", "ink",
    "topo", "marble",
    "lava", "nebula", "quartz", "tides", "cells",
    "scribble", "hatch", "stipple", "worms",

    # circular genome inspired
    "plasmid", "chromosome", "mandala", "radial_rings",
]

THEMES = [
    "vivid", "pastel", "ocean", "sunset",
    "forest", "ice", "earth", "berry", "neon",
    "mono", "ink", "fire", "aurora", "desert",
    "psy_acid", "psy_uv", "psy_candy", "psy_cyber", "psy_rainbow",
    "psy_lava", "psy_jungle", "psy_galaxy",
]

PSY_THEMES = [
    "psy_acid", "psy_uv", "psy_candy", "psy_cyber", "psy_rainbow",
    "psy_lava", "psy_jungle", "psy_galaxy",
]


# ----------------------------
# Rendering
# ----------------------------

def render_one(
    seq: str,
    out: str,
    style: str,
    theme: str,
    k: int,
    stride: int,
    user_seed: int,
    width: float,
    height: float,
    dpi: int,
    background: str,
    pad: float,
    density: float,
    alpha: float,
    lw_min: float,
    lw_max: float,
    fill: bool,
):
    # --fill: encourage coverage (less white)
    if fill:
        density = max(density, 1.25) * 1.15
        alpha = min(0.95, max(alpha, 0.32) * 1.10)
        pad = min(pad, 0.004)
        lw_min = max(lw_min, 1.2) * 1.10
        lw_max = max(lw_max, 6.0) * 1.15

    combo_seed = sha256_int(f"{style}|{theme}".encode("utf-8")) & 0xFFFFFFFFFFFFFFFF
    final_seed = int(user_seed) ^ int(combo_seed)

    rng = KmerRNG.from_sequence(seq, k=k, stride=stride, user_seed=final_seed)
    fig, ax = setup_figure(width, height, dpi, background)

    if theme in ("mono", "ink"):
        pal_n = 16
    elif style in ("topo", "marble", "lava", "tides", "quartz", "cells", "radial_rings"):
        pal_n = 28
    else:
        pal_n = 36

    palette = make_palette(rng, n=pal_n, theme=theme)

    # defaults for flow styles
    fill_box = 1.03
    field_components = 18
    field_gain = 1.05
    steps = int(280 * density)
    strokes = int(3200 * density)
    step_size = 0.017
    jitter = 0.0007
    chaotic = 0.0
    directional = 0.0
    mode = "line"
    dot_size = 4.0

    # dispatch non-flow styles
    if style in ("papercut", "ribbons", "swirl"):
        draw_papercut_like(ax, rng, palette, layers=int(60 * density), band=0.02, alpha=min(0.95, alpha + 0.40))
        finalize_and_save(fig, ax, out, pad)
        return

    if style == "topo":
        f = make_scalar_field(rng, components=18, chaotic=0.30)
        X, Y, Z = scalar_grid(f, n=420, span=1.05)
        draw_contours(ax, X, Y, Z, palette, levels=max(24, int(48 * density)), lw=max(0.6, lw_min), alpha=alpha)
        finalize_and_save(fig, ax, out, pad)
        return

    if style == "marble":
        draw_marble(ax, rng, palette, bands=max(12, int(22 * density)),
                    lw_min=max(0.8, lw_min), lw_max=max(2.5, lw_max), alpha=alpha)
        finalize_and_save(fig, ax, out, pad)
        return

    if style == "lava":
        draw_lava(ax, rng, palette, density=density, alpha=alpha, lw_min=lw_min, lw_max=lw_max)
        finalize_and_save(fig, ax, out, pad)
        return

    if style == "tides":
        draw_tides(ax, rng, palette, density=density, alpha=alpha, lw=max(0.6, lw_min))
        finalize_and_save(fig, ax, out, pad)
        return

    if style == "quartz":
        draw_quartz(ax, rng, palette, density=density, alpha=alpha, lw=max(0.6, lw_min))
        finalize_and_save(fig, ax, out, pad)
        return

    if style == "cells":
        draw_cells(ax, rng, palette, density=density, alpha=min(0.85, alpha + 0.20))
        finalize_and_save(fig, ax, out, pad)
        return

    if style == "nebula":
        draw_nebula(ax, rng, palette, density=density, alpha=min(0.95, alpha + 0.25))
        finalize_and_save(fig, ax, out, pad)
        return

    # circular styles
    if style == "radial_rings":
        draw_radial_rings(ax, rng, palette, density=density, alpha=alpha, lw=max(0.7, lw_min))
        finalize_and_save(fig, ax, out, pad)
        return

    if style == "mandala":
        draw_mandala(ax, rng, palette, density=density, alpha=alpha)
        finalize_and_save(fig, ax, out, pad)
        return

    if style == "plasmid":
        draw_plasmid(ax, rng, palette, density=density, alpha=alpha, lw_min=max(0.7, lw_min), lw_max=lw_max)
        finalize_and_save(fig, ax, out, pad)
        return

    if style == "chromosome":
        # looks best with slightly higher alpha
        draw_chromosome(ax, rng, palette, density=density, alpha=min(0.85, alpha + 0.10),
                        lw_min=max(0.7, lw_min), lw_max=lw_max)
        finalize_and_save(fig, ax, out, pad)
        return

    # flow style presets
    if style == "fluid":
        pass
    elif style == "fluid_bold":
        strokes = int(4200 * density)
        steps = int(300 * density)
        field_gain = 1.10
        step_size = 0.017
        jitter = 0.00075
    elif style == "fluid_wispy":
        strokes = int(5200 * density)
        steps = int(260 * density)
        field_gain = 0.92
        step_size = 0.016
        jitter = 0.00065
    elif style == "turbulence":
        strokes = int(3800 * density)
        steps = int(260 * density)
        field_components = 22
        field_gain = 1.20
        step_size = 0.020
        chaotic = 0.55
        jitter = 0.00085
    elif style == "streamlines":
        strokes = int(8200 * density)
        steps = int(220 * density)
        field_components = 16
        field_gain = 0.95
        step_size = 0.014
        jitter = 0.00055
    elif style == "bloom":
        strokes = int(5200 * density)
        steps = int(240 * density)
        field_components = 18
        field_gain = 1.05
        step_size = 0.016
        jitter = 0.00065
        fill_box = 0.55
    elif style == "storm":
        strokes = int(5200 * density)
        steps = int(240 * density)
        field_components = 20
        field_gain = 1.25
        step_size = 0.020
        chaotic = 0.35
        directional = 0.75
        jitter = 0.00080
    elif style == "ink":
        if theme not in ("mono", "ink"):
            palette = make_palette(rng, n=pal_n, theme="ink")
        strokes = int(12000 * density)
        steps = int(220 * density)
        field_components = 14
        field_gain = 0.90
        step_size = 0.013
        jitter = 0.00050
    elif style == "doodle":
        strokes = int(5200 * density)
        steps = int(180 * density)
        field_components = 12
        field_gain = 0.85
        step_size = 0.018
        jitter = 0.0012
        chaotic = 0.70
    elif style == "scribble":
        strokes = int(8800 * density)
        steps = int(140 * density)
        field_components = 10
        field_gain = 0.78
        step_size = 0.020
        jitter = 0.0015
        chaotic = 0.85
    elif style == "worms":
        strokes = int(9200 * density)
        steps = int(110 * density)
        field_components = 12
        field_gain = 0.82
        step_size = 0.015
        jitter = 0.0010
        chaotic = 0.55
    elif style == "hatch":
        strokes = int(9000 * density)
        steps = int(150 * density)
        field_components = 14
        field_gain = 0.92
        step_size = 0.013
        jitter = 0.00055
        chaotic = 0.25
        draw_flow_strokes(ax, rng, palette, strokes, steps, step_size, field_components, field_gain,
                          alpha=min(0.55, alpha), lw_min=max(0.6, lw_min * 0.55), lw_max=max(2.4, lw_max * 0.55),
                          fill_box=fill_box, chaotic=chaotic, jitter=jitter, directional=0.65)
        draw_flow_strokes(ax, rng, palette[::-1], strokes, steps, step_size, field_components, field_gain,
                          alpha=min(0.55, alpha), lw_min=max(0.6, lw_min * 0.55), lw_max=max(2.4, lw_max * 0.55),
                          fill_box=fill_box, chaotic=chaotic, jitter=jitter, directional=-0.65)
        finalize_and_save(fig, ax, out, pad)
        return
    elif style == "stipple":
        strokes = int(14000 * density)
        steps = int(90 * density)
        field_components = 10
        field_gain = 0.90
        step_size = 0.016
        jitter = 0.0010
        chaotic = 0.40
        mode = "dots"
        dot_size = max(2.0, 6.0 - 2.0 * density)
    else:
        raise ValueError(f"Unknown style: {style}")

    draw_flow_strokes(
        ax=ax,
        rng=rng,
        palette=palette,
        strokes=strokes,
        steps=steps,
        step_size=step_size,
        field_components=field_components,
        field_gain=field_gain,
        alpha=alpha,
        lw_min=lw_min,
        lw_max=lw_max,
        fill_box=fill_box,
        chaotic=chaotic,
        jitter=jitter,
        directional=directional,
        mode=mode,
        dot_size=dot_size,
    )

    finalize_and_save(fig, ax, out, pad)


# ----------------------------
# CLI utilities
# ----------------------------

def ensure_png(path: str) -> str:
    return path if path.lower().endswith(".png") else path + ".png"

def safe_mkdir(path: str):
    os.makedirs(path, exist_ok=True)

def parse_csv_list(s: str) -> List[str]:
    return [x.strip() for x in s.split(",") if x.strip()]

def resolve_styles(styles_arg: str) -> List[str]:
    if styles_arg.lower() in ("all", "*"):
        return list(STYLES)
    items = parse_csv_list(styles_arg)
    for it in items:
        if it not in STYLES:
            raise SystemExit(f"Unknown style in --styles: {it}\nAllowed: {', '.join(STYLES)}")
    return items

def resolve_themes(themes_arg: str) -> List[str]:
    t = themes_arg.lower()
    if t in ("all", "*"):
        return list(THEMES)
    if t in ("psy", "psychedelic"):
        return list(PSY_THEMES)
    items = parse_csv_list(themes_arg)
    for it in items:
        if it not in THEMES:
            raise SystemExit(f"Unknown theme in --themes: {it}\nAllowed: {', '.join(THEMES)}")
    return items


def main():
    p = argparse.ArgumentParser(description="FASTA-driven abstract painting generator (multi-style).")

    p.add_argument("--fasta", required=True, help="Input FASTA")
    p.add_argument("--out", help="Output PNG path (single-run mode).")
    p.add_argument("--outdir", default=".", help="Output directory (for --all mode, or if --out omitted).")
    p.add_argument("--prefix", default="painting", help="Prefix used for auto output naming.")
    p.add_argument("--sep", default="__", help="Separator in output filenames (default: __)")
    p.add_argument("--all", action="store_true",
                   help="Run selected styles × selected themes and write <prefix><sep><style><sep><theme>.png to --outdir")

    p.add_argument("--style", default="fluid", choices=STYLES, help="Painting style (single-run mode).")
    p.add_argument("--theme", default="ocean", choices=THEMES, help="Color theme (single-run mode).")

    p.add_argument("--styles", default="all",
                   help="Comma-separated styles for --all (or 'all'). Example: fluid,chromosome,marble")
    p.add_argument("--themes", default="all",
                   help="Comma-separated themes for --all (or 'all' or 'psy'). Example: vivid,psy_uv,psy_cyber")

    p.add_argument("--k", type=int, default=9, help="k-mer size")
    p.add_argument("--stride", type=int, default=7, help="k-mer stride")
    p.add_argument("--seed", type=int, default=0, help="User seed mixed in (still deterministic)")

    p.add_argument("--width", type=float, default=14.0, help="Figure width inches")
    p.add_argument("--height", type=float, default=6.0, help="Figure height inches")
    p.add_argument("--dpi", type=int, default=300, help="DPI")

    p.add_argument("--background", default="white", help="Background color")
    p.add_argument("--pad", type=float, default=0.006, help="Relative padding (smaller fills canvas more)")

    p.add_argument("--density", type=float, default=1.0,
                   help="Global density multiplier. 0.5=lighter, 1.5=heavier.")
    p.add_argument("--alpha", type=float, default=0.26, help="Global opacity")
    p.add_argument("--lw_min", type=float, default=1.2, help="Min linewidth (stroke styles)")
    p.add_argument("--lw_max", type=float, default=7.0, help="Max linewidth (stroke styles)")

    p.add_argument("--fill", action="store_true",
                   help="Boost coverage (less white): increases density/alpha, reduces pad, thickens lines slightly.")

    args = p.parse_args()

    seq = read_fasta(args.fasta)
    if len(seq) < 200:
        raise SystemExit(f"FASTA sequence too short after filtering (len={len(seq)}).")

    safe_mkdir(args.outdir)

    if args.all:
        styles = resolve_styles(args.styles)
        themes = resolve_themes(args.themes)

        total = 0
        for style in styles:
            for theme in themes:
                out_name = f"{args.prefix}{args.sep}{style}{args.sep}{theme}.png"
                out_path = os.path.join(args.outdir, out_name)
                render_one(
                    seq=seq,
                    out=out_path,
                    style=style,
                    theme=theme,
                    k=args.k,
                    stride=args.stride,
                    user_seed=args.seed,
                    width=args.width,
                    height=args.height,
                    dpi=args.dpi,
                    background=args.background,
                    pad=args.pad,
                    density=args.density,
                    alpha=args.alpha,
                    lw_min=args.lw_min,
                    lw_max=args.lw_max,
                    fill=args.fill,
                )
                total += 1
                print(f"[OK] {out_path}")
        print(f"[DONE] Wrote {total} images to: {args.outdir}")
        return

    if args.out:
        out_path = ensure_png(args.out)
    else:
        out_name = f"{args.prefix}{args.sep}{args.style}{args.sep}{args.theme}.png"
        out_path = os.path.join(args.outdir, out_name)

    render_one(
        seq=seq,
        out=out_path,
        style=args.style,
        theme=args.theme,
        k=args.k,
        stride=args.stride,
        user_seed=args.seed,
        width=args.width,
        height=args.height,
        dpi=args.dpi,
        background=args.background,
        pad=args.pad,
        density=args.density,
        alpha=args.alpha,
        lw_min=args.lw_min,
        lw_max=args.lw_max,
        fill=args.fill,
    )
    print(f"[OK] Wrote: {out_path}")

if __name__ == "__main__":
    main()
