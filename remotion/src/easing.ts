import { Easing } from "remotion";

// The template's easing curves (render_card.py): eoq, eoc, eic, eob
export const eoq = Easing.out(Easing.poly(5));
export const eoc = Easing.out(Easing.cubic);
export const eic = Easing.in(Easing.cubic);
export const eob = Easing.out(Easing.back(2.2));
export const clamp = { extrapolateLeft: "clamp", extrapolateRight: "clamp" } as const;
