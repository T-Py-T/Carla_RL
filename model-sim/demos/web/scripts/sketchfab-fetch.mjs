#!/usr/bin/env node
/**
 * Fetch CC-BY Sketchfab traffic models via the official download API.
 * Requires SKETCHFAB_TOKEN (https://sketchfab.com/settings/password → API token).
 *
 * Usage:
 *   SKETCHFAB_TOKEN=... node scripts/sketchfab-fetch.mjs
 *   SKETCHFAB_TOKEN=... node scripts/sketchfab-fetch.mjs sketchfab-generic-sedan
 */
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { execFileSync } from "node:child_process";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const ROOT = path.resolve(__dirname, "..");
const PUBLIC = path.join(ROOT, "public", "models", "traffic");
const CATALOG = JSON.parse(fs.readFileSync(path.join(__dirname, "sketchfab-traffic-models.json"), "utf8"));

const token = process.env.SKETCHFAB_TOKEN?.trim();
if (!token) {
  console.error("SKETCHFAB_TOKEN is required. Create one at https://sketchfab.com/settings/password");
  process.exit(1);
}

const only = process.argv.slice(2);
const models = only.length ? CATALOG.filter((m) => only.includes(m.id) || only.includes(m.slug)) : CATALOG;
if (!models.length) {
  console.error("No matching models in sketchfab-traffic-models.json");
  process.exit(1);
}

async function api(pathname) {
  const res = await fetch(`https://api.sketchfab.com/v3${pathname}`, {
    headers: { Authorization: `Token ${token}` },
  });
  const body = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(`${pathname}: ${body.detail || res.statusText}`);
  return body;
}

function writeAttribution(dir, spec, modelUrl, licenseUrl) {
  const attribution = `# ${spec.modelName} (NPC traffic)

- **Original artist:** [${spec.artist}](${spec.artistUrl})
- **Original model:** [${spec.modelName}](${modelUrl}) on Sketchfab
- **License:** [CC BY 4.0](${licenseUrl}) (verified via Sketchfab API, ${new Date().toISOString().slice(0, 10)})
- **Bundled:** \`model.glb\` — Draco-compressed for web playback

This work is based on "${spec.modelName}" (${modelUrl}) by ${spec.artist} (${spec.artistUrl}) licensed under CC-BY-4.0 (${licenseUrl}).
`;
  const license = `This work is based on "${spec.modelName}" (${modelUrl}) by ${spec.artist} (${spec.artistUrl}) licensed under CC-BY-4.0 (${licenseUrl})
`;
  fs.writeFileSync(path.join(dir, "ATTRIBUTION.md"), attribution);
  fs.writeFileSync(path.join(dir, "LICENSE.txt"), license);
}

function run(cmd, args) {
  execFileSync(cmd, args, { stdio: "inherit" });
}

for (const spec of models) {
  console.log(`\n=== ${spec.id} (${spec.uid}) ===`);
  const meta = await api(`/models/${spec.uid}`);
  if (meta.license?.slug !== "by") {
    throw new Error(`${spec.id}: expected CC-BY, got ${meta.license?.label}`);
  }
  const dl = await api(`/models/${spec.uid}/download`);
  const glbEntry = dl.glb || dl.source || Object.values(dl).find((v) => v?.url?.endsWith?.(".glb"));
  const glbUrl = glbEntry?.url || dl.glb?.url;
  if (!glbUrl) throw new Error(`${spec.id}: no GLB URL in download response`);

  const outDir = path.join(PUBLIC, spec.slug);
  fs.mkdirSync(outDir, { recursive: true });
  const rawPath = path.join(outDir, "source.glb");
  const simplifiedPath = path.join(outDir, "simplified.glb");
  const finalPath = path.join(outDir, "model.glb");

  console.log("Downloading", glbUrl);
  const fileRes = await fetch(glbUrl);
  if (!fileRes.ok) throw new Error(`Download failed: ${fileRes.status}`);
  fs.writeFileSync(rawPath, Buffer.from(await fileRes.arrayBuffer()));

  const ratio = String(spec.simplifyRatio ?? 0.25);
  console.log("Simplifying ratio", ratio);
  run("npx", ["--yes", "@gltf-transform/cli", "simplify", rawPath, simplifiedPath, "--ratio", ratio, "--error", "0.001"]);
  console.log("Draco encoding");
  run("npx", ["--yes", "@gltf-transform/cli", "draco", simplifiedPath, finalPath]);

  fs.unlinkSync(rawPath);
  fs.unlinkSync(simplifiedPath);

  const modelUrl = meta.viewerUrl || `https://sketchfab.com/3d-models/_-${spec.uid}`;
  const licenseUrl = meta.license?.url || "https://creativecommons.org/licenses/by/4.0/";
  writeAttribution(outDir, spec, modelUrl, licenseUrl);

  const sizeKb = Math.round(fs.statSync(finalPath).size / 1024);
  console.log(`Wrote ${finalPath} (${sizeKb} KB)`);
}

function enableSketchfabFleet() {
  const assetsPath = path.join(ROOT, "src", "vendor", "jevpilot", "traffic-assets.ts");
  const worldPath = path.join(ROOT, "src", "playback", "world.ts");
  let assets = fs.readFileSync(assetsPath, "utf8");
  if (!assets.includes("...SKETCHFAB_TRAFFIC_MODELS")) {
    assets = assets.replace(
      "] as const;\n\n/** Sketchfab CC-BY shortlist",
      ",\n  ...SKETCHFAB_TRAFFIC_MODELS,\n] as const;\n\n/** Sketchfab CC-BY shortlist",
    );
    fs.writeFileSync(assetsPath, assets);
    console.log("Merged SKETCHFAB_TRAFFIC_MODELS into TRAFFIC_MODELS");
  }

  const mixedSpawns = `    const specs = [
      { type: "vehicle", x: -2.5, z: -13, speed: 0.09, modelId: "sketchfab-modern-sedan" },
      { type: "vehicle", x: 2.8, z: -20, speed: 0.07, modelId: "sketchfab-toyoace-van" },
      { type: "vehicle", x: 0.5, z: -30, speed: 0.08, modelId: "sketchfab-gmc-school-bus" },
      { type: "vehicle", x: -0.5, z: -42, speed: 0.1, modelId: "sketchfab-generic-sedan" },
      { type: "vehicle", x: 2.8, z: -54, speed: 0.075, modelId: "kenney-truck" },
      { type: "vehicle", x: -2.5, z: -66, speed: 0.085, modelId: "sketchfab-box-truck" },
      { type: "vehicle", x: 0.5, z: -78, speed: 0.07, modelId: "sketchfab-delivery-truck" },
      { type: "vehicle", x: -2.5, z: -90, speed: 0.065, modelId: "sketchfab-renault-master-van" },
      { type: "vehicle", x: 2.8, z: -102, speed: 0.06, modelId: "kenney-suv" },
      { type: "vehicle", x: -0.5, z: -114, speed: 0.055, modelId: "khronos-milk-truck" },`;

  let world = fs.readFileSync(worldPath, "utf8");
  world = world.replace(
    /const specs = \[[\s\S]*?\{ type: "pedestrian", x: -9, z: -22/,
    `${mixedSpawns}\n      { type: "pedestrian", x: -9, z: -22`,
  );
  world = world.replace(
    "// After `npm run fetch:sketchfab-traffic`, swap in SKETCHFAB_TRAFFIC_MODELS ids for mixed fleet.\n    ",
    "",
  );
  fs.writeFileSync(worldPath, world);
  console.log("Updated world.ts with Sketchfab + Kenney mixed spawns");
}

enableSketchfabFleet();
console.log("\nDone. Rebuild demo and recapture proof artifacts.");
