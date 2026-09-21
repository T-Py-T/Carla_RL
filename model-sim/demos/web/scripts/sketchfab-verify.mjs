#!/usr/bin/env node
/**
 * Re-check Sketchfab shortlist license + download metadata (no auth required).
 * Usage: node scripts/sketchfab-verify.mjs
 */
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const CATALOG = JSON.parse(fs.readFileSync(path.join(__dirname, "sketchfab-traffic-models.json"), "utf8"));
const OPTIONAL = {
  id: "sketchfab-car-pack",
  uid: "20f9af9b8a404d5cb022ac6fe87f21f5",
  artist: "Comrade1280",
  modelName: "Generic passenger car pack",
  optional: true,
};

async function prefetch(uid) {
  const embed = await fetch(`https://sketchfab.com/models/${uid}/embed`).then((r) => r.text());
  const marker =
    ' <div class="dom-data-container" style="display:none;" id="js-dom-data-prefetched-data"><!--';
  const chunk = embed.split(marker)[1]?.split("-->")[0];
  if (!chunk) return null;
  const data = JSON.parse(chunk.replace(/&#34;/g, '"'));
  return data[`/i/models/${uid}`];
}

const all = [...CATALOG, OPTIONAL];
const rows = [];

for (const spec of all) {
  const api = await fetch(`https://api.sketchfab.com/v3/models/${spec.uid}`).then((r) => r.json());
  const pref = await prefetch(spec.uid);
  const lic = api.license || {};
  const ok = lic.slug === "by";
  rows.push({
    id: spec.id || spec.uid.slice(0, 8),
    name: api.name,
    artist: api.user?.displayName,
    license: lic.label,
    licenseSlug: lic.slug,
    licenseUrl: lic.url,
    downloadType: pref?.downloadType ?? api.downloadType,
    faceCount: api.faceCount,
    viewerUrl: api.viewerUrl,
    approved: ok,
    optional: !!spec.optional,
  });
}

console.log(JSON.stringify(rows, null, 2));
const rejected = rows.filter((r) => !r.approved);
if (rejected.length) {
  console.error("\nREJECTED (not CC-BY):", rejected.map((r) => r.id).join(", "));
  process.exit(1);
}
console.log(`\nAll ${rows.length} models pass CC-BY (slug=by). Vendor via SKETCHFAB_TOKEN + npm run fetch:sketchfab-traffic`);
