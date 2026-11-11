import { Composition, Folder, staticFile } from "remotion";
import { Card } from "./Card";
import { FORMATS, type Layout } from "./layout";
import { MillerArt } from "./MillerArt";
import millerLayout from "../public/miller-daddy/layout.json";
import "./fonts";

const MillerCard: React.FC<{ layout: Layout; avatars: string[] }> = (p) => <Card {...p} Art={MillerArt} />;

// One folder per article, one composition per format (1x1, 4x5, 9x16, 16x9), 10s seamless loops at 30fps.
// Layouts come from render_card.py: `python3 remotion/export_layout.py articles/<slug>/article.json`
export const RemotionRoot: React.FC = () => {
  return (
    <Folder name="miller-daddy">
      {FORMATS.map((f) => {
        const layout = (millerLayout as unknown as Record<string, Layout>)[f];
        return (
          <Composition
            key={f}
            id={`miller-daddy-${f}`}
            component={MillerCard}
            durationInFrames={300}
            fps={30}
            width={layout.size[0]}
            height={layout.size[1]}
            defaultProps={{ layout, avatars: [staticFile("miller-daddy/avatar-0.jpg")] }}
          />
        );
      })}
    </Folder>
  );
};
