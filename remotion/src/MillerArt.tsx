import { AbsoluteFill, Img, Interactive, interpolate, staticFile, useCurrentFrame, useVideoConfig } from "remotion";
import arm from "../public/miller-daddy/arm.json";

// Miller's extra arm pats Trump on the head: three pats right away, three more mid-loop.
// The hand rests on Trump's head on frame 0 and at the loop point, so the loop is seamless.
const PATS: [number, number][] = [
  [0.15, 3],
  [4.6, 3],
]; // [start (s), number of pats]
const PAT = 0.42; // seconds per pat

const patLift = (t: number) => {
  for (const [t0, n] of PATS) {
    const u = (t - t0) / PAT;
    if (u >= 0 && u < n) return Math.sin(Math.PI * (u - Math.floor(u))) ** 1.5; // 0 = on his head, 1 = lifted
  }
  return 0;
};

const ART_W = 1920;
const ART_H = 1080;

export const MillerArt: React.FC<{ width: number; height: number }> = ({ width, height }) => {
  const frame = useCurrentFrame();
  const { fps, durationInFrames } = useVideoConfig();
  const lift = patLift(frame / fps);
  const [sx, sy] = arm.shoulder;
  const [wx, wy] = arm.wrist;

  return (
    <AbsoluteFill style={{ width, height, overflow: "hidden" }}>
      <AbsoluteFill
        name="Art (1920x1080, fitted)"
        style={{ width: ART_W, height: ART_H, scale: String(width / ART_W), transformOrigin: "0 0" }}
      >
        <Interactive.Div
          name="Push-in"
          style={{
            position: "absolute",
            inset: 0,
            transformOrigin: "50% 45%",
            // slow push-in and back over the loop: 1 → 1.035 → 1
            scale: String(interpolate(Math.cos((2 * Math.PI * frame) / durationInFrames), [-1, 1], [1.035, 1])),
          }}
        >
          <Img src={staticFile("miller-daddy/art.jpg")} style={{ position: "absolute", width: ART_W, height: ART_H }} />
          <Interactive.Div
            name="Extra arm (pivots at the shoulder)"
            style={{
              position: "absolute",
              inset: 0,
              transformOrigin: `${sx}px ${sy}px`,
              rotate: `${interpolate(lift, [0, 1], [0, 0.8])}deg`,
            }}
          >
            <Img
              src={staticFile("miller-daddy/sleeve.png")}
              style={{ position: "absolute", left: arm.sleeve.x, top: arm.sleeve.y, width: arm.sleeve.w, height: arm.sleeve.h }}
            />
            <Interactive.Div
              name="Hand (pivots at the wrist)"
              style={{
                position: "absolute",
                inset: 0,
                transformOrigin: `${wx}px ${wy}px`,
                rotate: `${interpolate(lift, [0, 1], [0, 10])}deg`,
              }}
            >
              <Img
                src={staticFile("miller-daddy/hand.png")}
                style={{
                  position: "absolute",
                  left: arm.hand.x,
                  top: arm.hand.y,
                  width: arm.hand.w,
                  height: arm.hand.h,
                  // contact shadow on his hair, softer and lower as the hand lifts
                  filter: `drop-shadow(0px ${interpolate(lift, [0, 1], [5, 16])}px ${interpolate(lift, [0, 1], [5, 12])}px rgba(0,0,0,${interpolate(lift, [0, 1], [0.5, 0.3])}))`,
                }}
              />
            </Interactive.Div>
          </Interactive.Div>
        </Interactive.Div>
      </AbsoluteFill>
    </AbsoluteFill>
  );
};
