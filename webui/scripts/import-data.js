import { copyFileSync, mkdirSync, statSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));
const root = resolve(here, "..", "..");
const out = resolve(here, "..", "public", "data");

const files = [
  "opencs2_play_vectors.csv",
  "opencs2_play_edges.csv",
  "opencs2_play_summary.json",
];

mkdirSync(out, { recursive: true });

for (const file of files) {
  const src = resolve(root, "outputs", file);
  const dest = resolve(out, file);
  statSync(src);
  copyFileSync(src, dest);
  console.log(`${src} -> ${dest}`);
}
