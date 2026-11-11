import { AbsoluteFill, Img, Interactive, interpolate, staticFile, useCurrentFrame, useVideoConfig } from "remotion";
import { clamp, eic, eob, eoc, eoq } from "./easing";
import { gothic } from "./fonts";
import type { Layout } from "./layout";

// The template's headline panel (render_card.py), timed for a 10s loop:
// headline rises word by word from frame 0, seam 0.25s, tag 0.35s, avatar 0.9s, name 1.05s; outro in the last ~1.5s.
export const Card: React.FC<{
  layout: Layout;
  avatars: string[];
  Art: React.FC<{ width: number; height: number }>;
}> = ({ layout: L, avatars, Art }) => {
  const frame = useCurrentFrame();
  const { fps, durationInFrames } = useVideoConfig();
  const t = frame / fps;
  const P = durationInFrames / fps;
  const E = P - 10; // outro anchored to the loop end
  const [ax, ay, aw, ah] = L.art;
  const { k } = L;
  const NW = L.head.words.length;

  const seamIn = interpolate(t, [0.25, 0.95], [0, 1], { ...clamp, easing: eoq });
  const seamOut = interpolate(t, [9 + E, 9.45 + E], [0, 1], { ...clamp, easing: eic });
  const tagOpen =
    interpolate(t, [0.35, 0.8], [0, 1], { ...clamp, easing: eoq }) *
    (1 - interpolate(t, [8.85 + E, 9.25 + E], [0, 1], { ...clamp, easing: eic }));
  const tagText = interpolate(t, [0.55, 0.95], [0, 1], { ...clamp, easing: eoc });
  const nameIn = interpolate(t, [1.05, 1.65], [0, 1], { ...clamp, easing: eoc });
  const nameOut = interpolate(t, [8.5 + E, 8.8 + E], [0, 1], clamp);

  return (
    <AbsoluteFill style={{ backgroundColor: "black" }}>
      <AbsoluteFill name="Art" style={{ left: ax, top: ay, width: aw, height: ah }}>
        <Art width={aw} height={ah} />
      </AbsoluteFill>

      <Interactive.Div
        name="Accent seam"
        style={
          L.seam === "bottom"
            ? {
                position: "absolute",
                backgroundColor: L.accent,
                top: ay + ah,
                height: L.seam_w + 1,
                left: ax + aw * seamOut,
                width: Math.max(0, aw * (seamIn - seamOut)),
              }
            : {
                position: "absolute",
                backgroundColor: L.accent,
                left: ax - L.seam_w,
                width: L.seam_w,
                top: ay + ah * seamOut,
                height: Math.max(0, ah * (seamIn - seamOut)),
              }
        }
      />

      <Interactive.Div
        name="Tag"
        style={{
          position: "absolute",
          left: L.tag.x,
          top: L.tag.y,
          height: L.tag.h + 1,
          width: L.tag.w * tagOpen,
          overflow: "hidden",
          backgroundColor: L.accent,
        }}
      >
        <div
          style={{
            ...gothic(700, 75),
            position: "absolute",
            color: "white",
            fontSize: L.tag.size,
            left: L.tag.tx,
            top: L.tag.ty + (1 - tagText) * 20 * k,
            opacity: tagText,
          }}
        >
          {L.tag.text}
        </div>
      </Interactive.Div>

      {L.head.words.map((w, j) => {
        // first in, last out: words rise from / sink into a mask; the first word is already moving on frame 0
        const q = interpolate(t + 1 / fps, [j * 0.04, j * 0.04 + 0.5], [0, 1], { ...clamp, easing: eoq });
        const outAt = P - 0.36 - (NW - 1) * 0.03 + (NW - 1 - j) * 0.03;
        const qo = interpolate(t, [outAt, outAt + 0.28], [0, 1], { ...clamp, easing: eic });
        return (
          <Interactive.Div
            key={j}
            name={`Headline: ${w.text}`}
            style={{
              position: "absolute",
              left: w.x,
              top: w.y,
              height: L.head.band_h,
              width: L.head.size * 10,
              clipPath: `inset(0px -${L.head.size}px 0px -${L.head.size}px)`,
              pointerEvents: "none",
            }}
          >
            <div
              style={{
                ...gothic(700, 75),
                position: "absolute",
                color: "white",
                fontSize: L.head.size,
                top: w.ty + (1 - q) * L.head.band_h + qo * L.head.band_h,
              }}
            >
              {w.text}
            </div>
          </Interactive.Div>
        );
      })}

      {L.avatars.map((a, n) => {
        const q = interpolate(t, [0.9 + n * 0.12, 1.4 + n * 0.12], [0, 1], clamp);
        const qo = interpolate(t, [8.45 + E - n * 0.06, 8.75 + E - n * 0.06], [0, 1], clamp);
        const sz = a.size * eob(q) * (1 - qo * qo);
        return (
          <Interactive.Div
            key={n}
            name={`Avatar ${n + 1}`}
            style={{
              position: "absolute",
              left: a.x + (a.size - sz) / 2,
              top: a.y + (a.size - sz) / 2,
              width: sz,
              height: sz,
              borderRadius: "50%",
              overflow: "hidden",
              opacity: Math.min(1, q * 2.5),
            }}
          >
            <Img src={avatars[n]} style={{ width: "100%", height: "100%", objectFit: "cover" }} />
            {/* white ring */}
            <div style={{ position: "absolute", inset: 0, borderRadius: "50%", boxShadow: `inset 0 0 0 ${(a.size / 200) * 7}px white` }} />
          </Interactive.Div>
        );
      })}

      {L.byline ? (
        <Interactive.Div
          name="Byline"
          style={{
            ...gothic(700, 90),
            position: "absolute",
            color: "rgb(235,235,235)",
            fontSize: L.byline.size,
            left: L.byline.x,
            top: L.byline.y + (1 - nameIn) * 18 * k + nameOut * 10 * k,
            opacity: nameIn * (1 - nameOut),
          }}
        >
          {L.byline.text}
        </Interactive.Div>
      ) : null}

      {L.logo ? (
        <Img
          name="Logo"
          src={staticFile(`brand/logo-${L.logo.kind}.png`)}
          style={{ position: "absolute", left: L.logo.x, top: L.logo.y, width: L.logo.w, height: L.logo.h, opacity: L.logo.opacity }}
        />
      ) : null}
    </AbsoluteFill>
  );
};
