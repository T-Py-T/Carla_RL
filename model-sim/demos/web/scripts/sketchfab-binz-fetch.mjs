#!/usr/bin/env node
/**
 * Attempt binz download + wine decrypt (fallback when SKETCHFAB_TOKEN unavailable).
 * Requires SF_Ripper tools at /tmp/sf-ripper (cloned separately).
 */
import fs from "node:fs";
import path from "node:path";
import { execFileSync } from "node:child_process";
import { fileURLToPath } from "node:url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const CATALOG = JSON.parse(fs.readFileSync(path.join(__dirname, "sketchfab-traffic-models.json"), "utf8"));
const SF_RIPPER = process.env.SF_RIPPER_DIR || "/tmp/sf-ripper";
const uid = process.argv[2] || CATALOG[0].uid;
const outDir = process.argv[3] || `/tmp/sf-binz/${uid}`;

fs.mkdirSync(outDir, { recursive: true });

const embed = await fetch(`https://sketchfab.com/models/${uid}/embed`).then((r) => r.text());
const marker =
  ' <div class="dom-data-container" style="display:none;" id="js-dom-data-prefetched-data"><!--';
const vi = JSON.parse(embed.split(marker)[1].split("-->")[0].replace(/&#34;/g, '"'))[`/i/models/${uid}`];
const key1 = vi.files[0].p[0].b;
fs.writeFileSync(path.join(outDir, "key.txt"), key1);

// Search embed scripts for key2 (SF_Ripper pattern + fallbacks)
const scripts = [...embed.matchAll(/<script src="([^"]+)"/g)].map((m) => m[1]);
let key2 = null;
for (const url of scripts) {
  if (!url.includes("sketchfab")) continue;
  const js = await fetch(url).then((r) => r.text());
  const patterns = [
    /pXZ0:\([^)]*\)=>\{[^}]*\};const n="([^"]+)"/,
    /a\.d\(t,\{k:\(\)=>n\}\);const n="([^"]+)"/,
    /k:\(\)=>n\}[^;]*;const n="([A-Za-z0-9+/=]{40,})"/,
  ];
  for (const pat of patterns) {
    const m = js.match(pat);
    if (m) {
      key2 = m[1];
      break;
    }
  }
  if (key2) break;
}
if (!key2) {
  console.error("key2 not found in viewer JS — binz decrypt blocked");
  process.exit(1);
}
fs.writeFileSync(path.join(outDir, "key2.txt"), key2);

const binzUrl = vi.files[0].osgjsUrl;
const binzPath = path.join(outDir, "file.binz");
const binz = await fetch(binzUrl, { headers: { Referer: "https://sketchfab.com/" } });
fs.writeFileSync(binzPath, Buffer.from(await binz.arrayBuffer()));
console.log("Downloaded", binzPath, fs.statSync(binzPath).size);

const decrypt = path.join(SF_RIPPER, "tools/binz/binzDecrypt.exe");
execFileSync("wine", [decrypt, "key.txt", "file.binz"], { cwd: outDir, stdio: "inherit" });
console.log("Output:", fs.readdirSync(outDir));
