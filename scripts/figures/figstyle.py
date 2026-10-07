# -*- coding: utf-8 -*-
"""Shared drawing style for method figures (palette, icons, units) -- copied from the driving-VLA
project's ibdstyle.py (Okabe-Ito colours, numbered steps), with DejaVu Sans and larger fonts for the web.
1 data unit = 0.01 in; figure width 630 = ACL text width; all fonts >= 6.5 pt."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle, Wedge, Polygon, PathPatch
from matplotlib.path import Path
from matplotlib.transforms import Affine2D
from PIL import Image

plt.rcParams.update({"font.family": "DejaVu Sans", "mathtext.fontset": "stix", "pdf.fonttype": 42,
                     "svg.fonttype": "none", "axes.unicode_minus": False})

BLUE, SKY, ORANGE, GREEN, VERM = "#0072B2", "#56B4E9", "#E69F00", "#009E73", "#D55E00"
PURPLE, GRAY, DGRAY = "#7B52AB", "#6E6E6E", "#3A3A3A"
LBLUE, LPURPLE, LGREEN, LORANGE = "#EAF2FA", "#EEE7F7", "#E3F3EC", "#FCEFD9"
PANEL_FC, PANEL_EC = "#F7F9FC", "#C3CEDD"
TOK_X, TOK_C, TOK_Q, TOK_P, TOK_STAR = "#BFD7EE", "#BFE3D3", "#E2E2E2", "white", "#D9C9EE"
FS_T, FS, FS_S = 10.0, 9.0, 8.0
PT = 100 / 72
SECT = ((40, 73), (73, 107), (107, 140))   # front-right, front, front-left


class Canvas:
    def __init__(self, W, H):
        self.W, self.H = W, H
        self.fig = plt.figure(figsize=(W / 100, H / 100), dpi=100)
        self.ax = self.fig.add_axes([0, 0, 1, 1])
        ax = self.ax
        ax.set_xlim(0, W); ax.set_ylim(0, H); ax.set_aspect("equal"); ax.axis("off")
        self.REN = self.fig.canvas.get_renderer()
        self.SPACE = {fs: self.width_of("a a", fs) - self.width_of("aa", fs) for fs in (FS, FS_S, 6.8)}

    # ------------------------------------------------------------ primitives
    def rbox(self, x, y, w, h, fc="white", ec=GRAY, lw=0.8, r=3, ls="-", z=1):
        p = FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}", fc=fc, ec=ec, lw=lw, ls=ls,
                           zorder=z)
        self.ax.add_patch(p)
        return p

    def rect(self, x, y, w, h, fc="white", ec=GRAY, lw=0.6, ls="-", z=4):
        self.ax.add_patch(Rectangle((x, y), w, h, fc=fc, ec=ec, lw=lw, ls=ls, zorder=z))

    def txt(self, x, y, s, fs=FS, color="black", ha="center", va="center", weight="normal", style="normal", z=5):
        return self.ax.text(x, y, s, fontsize=fs, color=color, ha=ha, va=va, weight=weight, style=style, zorder=z)

    def width_of(self, s, fs=FS, weight="normal"):
        t = self.ax.text(0, 0, s, fontsize=fs, weight=weight)
        w = t.get_window_extent(renderer=self.REN).width
        t.remove()
        return w

    def arrow(self, p, q, color=DGRAY, lw=0.9, ls="-", head=5.5, z=4):
        p, q = np.array(p, float), np.array(q, float)
        if ls == "-":
            self.ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=head, color=color, lw=lw,
                                              zorder=z, shrinkA=0, shrinkB=0))
            return
        d = (q - p) / np.linalg.norm(q - p)
        self.ax.plot([p[0], q[0] - d[0] * 3], [p[1], q[1] - d[1] * 3], color=color, lw=lw, ls=ls, zorder=z,
                     dash_capstyle="butt")
        self.ax.add_patch(FancyArrowPatch(q - d * 4.5, q, arrowstyle="-|>", mutation_scale=head, color=color, lw=lw,
                                          zorder=z, shrinkA=0, shrinkB=0))

    def polyarrow(self, pts, color=DGRAY, lw=0.9, ls="-", head=5.5, z=4):
        pts = [np.array(v, float) for v in pts]
        for a, b in zip(pts[:-2], pts[1:-1]):
            self.ax.plot([a[0], b[0]], [a[1], b[1]], color=color, lw=lw, ls=ls, zorder=z)
        self.arrow(pts[-2], pts[-1], color, lw, ls, head, z)

    def line(self, xs, ys, color=GRAY, lw=0.8, ls="-", z=3):
        self.ax.plot(xs, ys, color=color, lw=lw, ls=ls, zorder=z)

    # ------------------------------------------------------------ icons
    def snowflake(self, x, y, s=8, color=SKY, lw=0.9):
        for ang in (90, 30, 150):
            a = np.deg2rad(ang)
            dx, dy = np.cos(a) * s / 2, np.sin(a) * s / 2
            self.ax.plot([x - dx, x + dx], [y - dy, y + dy], color=color, lw=lw, solid_capstyle="round", zorder=6)
            for sgn in (1, -1):
                bx, by = x + sgn * dx * 0.55, y + sgn * dy * 0.55
                base = a if sgn == 1 else a + np.pi
                for da in (40, -40):
                    b = base + np.deg2rad(da)
                    self.ax.plot([bx, bx + np.cos(b) * s * 0.2], [by, by + np.sin(b) * s * 0.2], color=color,
                                 lw=lw * 0.8, solid_capstyle="round", zorder=6)

    @staticmethod
    def _flame_path(x, y, s):
        v = [(x, y - .5 * s),
             (x - .46 * s, y - .5 * s), (x - .42 * s, y + .02 * s), (x - .04 * s, y + .5 * s),
             (x + .06 * s, y + .22 * s), (x + .44 * s, y + .12 * s), (x + .38 * s, y - .18 * s),
             (x + .34 * s, y - .46 * s), (x + .12 * s, y - .5 * s), (x, y - .5 * s), (x, y - .5 * s)]
        c = [Path.MOVETO] + [Path.CURVE4] * 9 + [Path.CLOSEPOLY]
        return Path(v, c)

    def flame(self, x, y, s=8):
        self.ax.add_patch(PathPatch(self._flame_path(x, y, s), fc=VERM, ec="none", zorder=6))
        self.ax.add_patch(PathPatch(self._flame_path(x + .01 * s, y - .16 * s, s * .52), fc="#F5C242", ec="none",
                                    zorder=7))

    def badge(self, x, y, n, fc=DGRAY, r=5.2):
        """Numbered step marker."""
        from matplotlib.patches import Circle
        self.ax.add_patch(Circle((x, y), r, fc=fc, ec="white", lw=0.6, zorder=7))
        self.txt(x, y - 0.2, str(n), FS_S, "white", weight="bold", z=8)

    def mlp(self, x, y, w=30, h=13, fc="#E8C9C5", ec="#B07A73", label="MLP"):
        """Small trapezoid head (narrow at the top), as in the SimLingo architecture figure."""
        self.ax.add_patch(Polygon([(x, y), (x + w, y), (x + w - 5, y + h), (x + 5, y + h)], closed=True, fc=fc, ec=ec,
                                  lw=0.7, zorder=4))
        self.txt(x + w / 2, y + h / 2, label, FS_S, DGRAY)

    def pill(self, x_right, y, s, fc, color="white"):
        w = self.width_of(s, FS_S, "bold") + 6
        self.rbox(x_right - w, y - 5, w, 10, fc=fc, ec="none", r=5, z=4)
        self.txt(x_right - w / 2, y - 0.2, s, FS_S, color, weight="bold")

    def tokens(self, x, y, n, fc, ec, s=7.0, step=9.0, lw=0.6, z=4):
        for i in range(n):
            self.rect(x + i * step, y - s / 2, s, s, fc=fc, ec=ec, lw=lw, z=z)
        return x + (n - 1) * step + s

    def vtokens(self, x, y_top, n, fc, ec, s=7.0, step=9.0, lw=0.6, z=4):
        for i in range(n):
            self.rect(x - s / 2, y_top - i * step - s, s, s, fc=fc, ec=ec, lw=lw, z=z)
        return y_top - (n - 1) * step - s

    def waypoints(self, x, y, color, n=6, dx=6.0, curve=0.9):
        pts = [(x + dx * i, y + curve * i ** 1.5) for i in range(n)]
        self.ax.plot([p[0] for p in pts], [p[1] for p in pts], color="#9A9A9A", lw=0.6, zorder=3)
        for p in pts:
            self.ax.add_patch(plt.Circle(p, 1.6, fc=color, ec="white", lw=0.3, zorder=4))
        return pts

    def fan(self, cx, cy, rad, fills=None, ec="#9A9A9A", lw=0.5, mark=None, mark_color=PURPLE):
        fills = fills or {}
        for si, (t1, t2) in enumerate(SECT):
            for ri in range(3):
                self.ax.add_patch(Wedge((cx, cy), rad[ri + 1], t1, t2, width=rad[ri + 1] - rad[ri],
                                        fc=fills.get((si, ri), "white"), ec=ec, lw=lw, zorder=2))
        if mark is not None:
            si, ri = mark
            t1, t2 = SECT[si]
            self.ax.add_patch(Wedge((cx, cy), rad[ri + 1], t1, t2, width=rad[ri + 1] - rad[ri], fc="none",
                                    ec=mark_color, lw=1.1, ls=(0, (2.2, 1.4)), zorder=3))
        self.car(cx, cy - 0.5, rad[0] * 0.9, rad[0] * 1.5, fc=BLUE)

    def car(self, x, y, w=5, h=8.5, ang=0, fc="#8C8C8C", ec="none", ls="-", alpha=1, lw=0.8, z=5):
        p = FancyBboxPatch((-w / 2, -h / 2), w, h, boxstyle="round,pad=0,rounding_size=1.0", fc=fc, ec=ec, lw=lw,
                           ls=ls, alpha=alpha, zorder=z)
        p.set_transform(Affine2D().rotate_deg(ang).translate(x, y) + self.ax.transData)
        self.ax.add_patch(p)

    def image(self, path, crop, x, y, w, box=None, box_color=PURPLE):
        im = Image.open(path).convert("RGB")
        cx0, cy0, cx1, cy1 = crop
        sub = np.asarray(im.crop(crop))
        h = w * (cy1 - cy0) / (cx1 - cx0)
        self.ax.imshow(sub, extent=(x, x + w, y, y + h), zorder=3, interpolation="lanczos")
        self.rect(x, y, w, h, fc="none", ec="#8A8A8A", lw=0.5, z=4)
        if box is not None:
            bx0, by0, bx1, by1 = box
            X0 = x + (bx0 - cx0) / (cx1 - cx0) * w
            X1 = x + (min(bx1, cx1) - cx0) / (cx1 - cx0) * w
            Y0 = y + h - (by1 - cy0) / (cy1 - cy0) * h
            Y1 = y + h - (by0 - cy0) / (cy1 - cy0) * h
            self.ax.add_patch(Rectangle((X0, Y0), X1 - X0, Y1 - Y0, fc="none", ec=box_color, lw=1.1,
                                        ls=(0, (2.4, 1.4)), zorder=5))
        return h

    # ------------------------------------------------------------ legend
    def legend(self, items, y):
        """items: list of (kind, label, color). kinds: snow, flame, fwd, priv, loss, grad, zero"""
        def draw(kind, x):
            if kind == "snow":
                self.snowflake(x + 4, y, 7.5); return 8
            if kind == "flame":
                self.flame(x + 4, y, 8); return 8
            if kind == "fwd":
                self.arrow((x, y), (x + 18, y)); return 18
            if kind == "priv":
                self.arrow((x, y), (x + 18, y), PURPLE, ls=(0, (3, 1.6))); return 18
            if kind == "loss":
                self.arrow((x, y), (x + 18, y), GRAY, ls=(0, (1, 1.4))); return 18
            if kind == "grad":
                self.arrow((x + 18, y), (x, y), VERM, ls=(0, (4, 1.6)), lw=1.0); return 18
            if kind == "mask":
                self.rect(x, y - 4, 8, 8, fc="white", ec="#B0B0B0", lw=0.5, z=5); return 8
            if kind == "attn":
                self.rect(x, y - 4, 8, 8, fc="#D6D6D6", ec="#B0B0B0", lw=0.5, z=5); return 8
            if kind == "attnz":
                self.rect(x, y - 4, 8, 8, fc="#F2B37A", ec="#B0B0B0", lw=0.5, z=5); return 8
            if kind == "ill":
                from matplotlib.patches import Ellipse
                self.ax.add_patch(Ellipse((x + 7, y), 14, 7, fc=SKY, ec=BLUE, lw=0.6, alpha=0.45, zorder=5))
                return 14
            if kind == "pgrad":
                for i, cc in enumerate(("#F3F7FB", "#9DC1E4", "#2F78B7")):
                    self.rect(x + i * 5, y - 4, 5, 8, fc=cc, ec="#9A9A9A", lw=0.4, z=5)
                return 15
            return 0
        widths = [{"snow": 8, "flame": 8, "fwd": 18, "priv": 18, "loss": 18, "grad": 18, "mask": 8, "attn": 8,
                   "attnz": 8, "ill": 14, "pgrad": 15}[k] for k, _, _ in items]
        total = sum(widths) + sum(self.width_of(t, FS_S) for _, t, _ in items) + 3 * len(items) + 14 * (len(items) - 1)
        x = (self.W - total) / 2
        for (k, t, col), w0 in zip(items, widths):
            draw(k, x)
            self.txt(x + w0 + 3, y, t, FS_S, col, ha="left")
            x += w0 + 3 + self.width_of(t, FS_S) + 14

    def save(self, out):
        self.fig.savefig(out + ".pdf")
        self.fig.savefig(out + ".svg")
        self.fig.savefig(out + ".png", dpi=600)
