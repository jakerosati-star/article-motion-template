// Layout data exported from render_card.py by export_layout.py: every position, wrap and size matches the template.
export type Word = { text: string; x: number; y: number; ty: number };
export type Layout = {
  size: [number, number];
  art: [number, number, number, number];
  seam: "bottom" | "left";
  seam_w: number;
  accent: string;
  k: number;
  X: number;
  TW: number;
  tag: { text: string; x: number; y: number; w: number; h: number; size: number; tx: number; ty: number };
  head: { size: number; band_h: number; raise_: number; lh: number; words: Word[] };
  byline: { text: string; size: number; x: number; y: number } | null;
  avatars: { x: number; y: number; size: number }[];
  logo: { kind: "horizontal" | "stacked"; x: number; y: number; w: number; h: number; opacity: number } | null;
};
export const FORMATS = ["1x1", "4x5", "9x16", "16x9"] as const;
export type Format = (typeof FORMATS)[number];
