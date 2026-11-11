import { loadFont } from "@remotion/fonts";
import { staticFile } from "remotion";

export const FONT = "Special Gothic";
loadFont({
  family: FONT,
  url: staticFile("brand/SpecialGothic-VariableFont_wdthwght.ttf"),
  weight: "400 700",
  display: "block",
});

// Variable axes per the template: headline / tag 700·75, byline 700·90
export const gothic = (wght: number, wdth: number): React.CSSProperties => ({
  fontFamily: FONT,
  fontWeight: wght,
  fontVariationSettings: `'wght' ${wght}, 'wdth' ${wdth}`,
  lineHeight: 1,
  whiteSpace: "pre",
});
