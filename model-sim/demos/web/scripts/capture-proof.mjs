#!/usr/bin/env node
/**
 * PR #114 proof capture — Playwright viewport / canvas, no Cursor IDE chrome.
 *
 * Motion frames are Playwright viewport screenshots (HTML HUD + canvas,
 * no Cursor chrome). capture-mode disables backdrop-filter so sequential
 * screenshots do not hang. JSON panel is hidden during the motion roll so
 * the lead vehicle stays visible; nav + bottom HUD stay on (speed / BRK).
 *
 * Encode: libx264 yuv420p +faststart, then `ffmpeg -i out.mp4 -f null -`
 * must succeed and start/mid/end SHA-256 hashes must be distinct.
 */
import crypto from "node:crypto";
import fs from "node:fs";
import http from "node:http";
import os from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { spawnSync } from "node:child_process";
import { chromium } from "playwright";
import { verifyArtifactDir, assertPngSignature } from "./verify-artifacts.mjs";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const webRoot = path.resolve(__dirname, "..");
const root = path.join(webRoot, "dist");
const outDir = process.argv[2] ?? path.resolve(webRoot, "../../../docs/pr-114-artifacts");
const PORT = process.env.CAPTURE_PORT || process.argv[3] || "";
const WIDTH = 1280;
const HEIGHT = 720;
const FRAMES = 48;
const FPS = 12;
/** Short warmup so the clip still contains the approach + scripted slow/stop. */
const WARMUP_START = 2;
const STILL_WARMUP = 36;
const PLAIN_WARMUP = 16;
const STEPS_PER_FRAME = 2;
const MIN_LEAD_GAP_M = 4;

function contentType(filePath) {
  if (filePath.endsWith(".css")) return "text/css";
  if (filePath.endsWith(".js")) return "text/javascript";
  if (filePath.endsWith(".html")) return "text/html";
  if (filePath.endsWith(".png")) return "image/png";
  if (filePath.endsWith(".glb")) return "model/gltf-binary";
  if (filePath.endsWith(".wasm")) return "application/wasm";
  return "application/octet-stream";
}

function startStaticServer() {
  return new Promise((resolve) => {
    const server = http.createServer((req, res) => {
      const urlPath = decodeURIComponent((req.url || "/").split("?")[0]);
      const rel = urlPath === "/" ? "/index.html" : urlPath;
      const filePath = path.join(root, rel);
      if (!filePath.startsWith(root) || !fs.existsSync(filePath)) {
        res.writeHead(404);
        res.end("not found");
        return;
      }
      res.writeHead(200, { "Content-Type": contentType(filePath) });
      fs.createReadStream(filePath).pipe(res);
    });
    server.listen(0, "127.0.0.1", () => resolve(server));
  });
}

function sha256(buf) {
  return crypto.createHash("sha256").update(buf).digest("hex");
}

function dataUrlToBuffer(dataUrl) {
  const comma = dataUrl.indexOf(",");
  if (comma < 0) throw new Error("invalid data URL");
  return Buffer.from(dataUrl.slice(comma + 1), "base64");
}

function writePng(filePath, buf) {
  fs.writeFileSync(filePath, buf);
  assertPngSignature(filePath);
}

async function warmupPage(page, url, steps) {
  await page.goto(url, { waitUntil: "domcontentloaded", timeout: 45000 });
  await page.waitForFunction(() => typeof window.stepSimulation === "function");
  return page.evaluate(async (n) => {
    await window.waitForDemoReady();
    for (let i = 0; i < n; i++) window.stepSimulation();
    return window.demoMotionSample();
  }, steps);
}

function encodeMp4(framesDir, mp4) {
  const r = spawnSync(
    "ffmpeg",
    [
      "-y",
      "-framerate",
      String(FPS),
      "-i",
      path.join(framesDir, "frame_%04d.png"),
      "-c:v",
      "libx264",
      "-preset",
      "medium",
      "-crf",
      "18",
      "-pix_fmt",
      "yuv420p",
      "-movflags",
      "+faststart",
      "-vf",
      "scale=trunc(iw/2)*2:trunc(ih/2)*2",
      mp4,
    ],
    { encoding: "utf8" },
  );
  if (r.status !== 0) {
    throw new Error(`ffmpeg MP4 encode failed:\n${r.stderr}`);
  }
  process.stdout.write(r.stderr || "");
}

function tryEncodeGif(framesDir, gif) {
  const r = spawnSync(
    "ffmpeg",
    [
      "-y",
      "-framerate",
      String(FPS),
      "-i",
      path.join(framesDir, "frame_%04d.png"),
      "-vf",
      "fps=10,scale=960:-1:flags=lanczos,split[s0][s1];[s0]palettegen=stats_mode=diff[p];[s1][p]paletteuse=dither=bayer:bayer_scale=3",
      "-loop",
      "0",
      gif,
    ],
    { encoding: "utf8" },
  );
  if (r.status !== 0) {
    console.warn("GIF encode failed; shipping MP4 + strip only.\n", r.stderr);
    if (fs.existsSync(gif)) fs.unlinkSync(gif);
    return false;
  }
  return true;
}

function makeStrip(t0, tmid, tend, stripPath) {
  const r = spawnSync(
    "ffmpeg",
    [
      "-y",
      "-i",
      t0,
      "-i",
      tmid,
      "-i",
      tend,
      "-filter_complex",
      "[0:v][1:v][2:v]hstack=inputs=3",
      "-frames:v",
      "1",
      stripPath,
    ],
    { encoding: "utf8" },
  );
  if (r.status !== 0) throw new Error(`strip encode failed:\n${r.stderr}`);
  assertPngSignature(stripPath);
}

async function main() {
  let server = null;
  let port = PORT;
  if (!port) {
    if (!fs.existsSync(root)) {
      const build = spawnSync("npm", ["run", "build"], { cwd: webRoot, stdio: "inherit" });
      if (build.status !== 0) throw new Error("vite build failed");
    }
    // Headless Chromium rejects the bundled Draco WASM; serve an uncompressed Model Y.
    const srcGlb = path.join(webRoot, "public/models/model-y/model-y.glb");
    const distGlb = path.join(root, "models/model-y/model-y.glb");
    const undraco = spawnSync("node", [path.join(webRoot, "scripts/undraco-modely.mjs"), srcGlb, distGlb], {
      encoding: "utf8",
    });
    if (undraco.status !== 0) {
      throw new Error(`Model Y undraco failed:\n${undraco.stderr || undraco.stdout}`);
    }
    process.stdout.write(undraco.stdout || "");
    server = await startStaticServer();
    port = server.address().port;
  }

  fs.mkdirSync(outDir, { recursive: true });
  const framesDir = path.join(os.tmpdir(), "pr114-threejs-frames");
  fs.rmSync(framesDir, { recursive: true, force: true });
  fs.mkdirSync(framesDir, { recursive: true });

  const browser = await chromium.launch({
    headless: true,
    args: [
      "--no-sandbox",
      "--disable-setuid-sandbox",
      "--use-gl=angle",
      "--use-angle=swiftshader",
      "--enable-unsafe-swiftshader",
      "--ignore-gpu-blocklist",
    ],
  });
  const ctx = await browser.newContext({
    viewport: { width: WIDTH, height: HEIGHT },
    deviceScaleFactor: 1,
  });
  // Model Y ego GLB (not procedural). Generic Kenney/procedural NPCs are OK.
  const q = `w=${WIDTH}&h=${HEIGHT}&proceduralTraffic=1&capture=1`;
  const host = `http://127.0.0.1:${port}/index.html?${q}`;

  try {
    console.log("before_plain_ego (viewport still)...");
    const pageBefore = await ctx.newPage();
    const plainMotion = await warmupPage(pageBefore, `${host}&plain=1`, PLAIN_WARMUP);
    const beforeBuf = await pageBefore.screenshot({ type: "png", timeout: 30000, animations: "disabled" });
    writePng(path.join(outDir, "before_plain_ego.png"), beforeBuf);
    console.log(`  plain egoZ=${plainMotion.egoZ.toFixed(2)} egoModel=${plainMotion.egoModel}`);
    if (plainMotion.egoModel !== "tesla-model-y") {
      throw new Error(`Ego must be Tesla Model Y, got ${plainMotion.egoModel}`);
    }
    await pageBefore.close();

    console.log("after_fsd_overlay (viewport still, HUD + overlay)...");
    const pageStill = await ctx.newPage();
    const stillMotion = await warmupPage(pageStill, host, STILL_WARMUP);
    const afterBuf = await pageStill.screenshot({ type: "png", timeout: 30000, animations: "disabled" });
    writePng(path.join(outDir, "after_fsd_overlay.png"), afterBuf);
    console.log(
      `  fsd egoZ=${stillMotion.egoZ.toFixed(2)} tracks=${stillMotion.tracks} ` +
        `speed=${stillMotion.speedKmh?.toFixed?.(1) ?? "?"} action=${stillMotion.action} ` +
        `leadGap=${Number.isFinite(stillMotion.leadGap) ? stillMotion.leadGap.toFixed(2) : "inf"}`,
    );
    if (stillMotion.egoModel !== "tesla-model-y") {
      throw new Error(`Ego must be Tesla Model Y, got ${stillMotion.egoModel}`);
    }
    if (stillMotion.action !== 3) {
      throw new Error(`after still must show BRAKE (action=3), got ${stillMotion.action}`);
    }
    await pageStill.close();

    console.log(`rollout ${FRAMES} viewport frames (HUD visible, no Cursor chrome)...`);
    const page = await ctx.newPage();
    const first = await warmupPage(page, host, WARMUP_START);
    if (first.egoModel !== "tesla-model-y") {
      throw new Error(`Motion clip ego must be Tesla Model Y, got ${first.egoModel}`);
    }
    // Keep nav + bottom HUD (speed / Slow for hazard / BRK); hide the JSON panel so the lead stays in view.
    await page.evaluate(() => {
      const json = document.getElementById("json-dialog");
      if (json) json.style.display = "none";
    });
    const hashes = new Set();
    let lastZ = first.egoZ;
    let firstZ = first.egoZ;
    let minLeadGap = Number.isFinite(first.leadGap) ? first.leadGap : Infinity;
    let maxSpeed = first.speedKmh ?? 0;
    let minSpeed = first.speedKmh ?? 0;
    let sawBrake = first.action === 3;

    for (let i = 0; i < FRAMES; i++) {
      const motion = await page.evaluate((steps) => {
        for (let s = 0; s < steps; s++) window.stepSimulation();
        return window.demoMotionSample();
      }, STEPS_PER_FRAME);
      const buf = await page.screenshot({ type: "png", timeout: 20000, animations: "disabled" });
      const framePath = path.join(framesDir, `frame_${String(i).padStart(4, "0")}.png`);
      writePng(framePath, buf);
      hashes.add(sha256(buf));
      lastZ = motion.egoZ;
      if (i === 0) firstZ = motion.egoZ;
      if (Number.isFinite(motion.leadGap)) minLeadGap = Math.min(minLeadGap, motion.leadGap);
      if (Number.isFinite(motion.speedKmh)) {
        maxSpeed = Math.max(maxSpeed, motion.speedKmh);
        minSpeed = Math.min(minSpeed, motion.speedKmh);
      }
      if (motion.action === 3) sawBrake = true;
      if (i % 12 === 0) {
        console.log(
          `  frame ${i}: egoZ=${motion.egoZ.toFixed(2)} speed=${motion.speedKmh?.toFixed?.(1)} ` +
            `action=${motion.action} leadGap=${Number.isFinite(motion.leadGap) ? motion.leadGap.toFixed(2) : "inf"} ` +
            `unique=${hashes.size}`,
        );
      }
    }
    await page.close();

    console.log(
      `egoZ ${firstZ.toFixed(2)} → ${lastZ.toFixed(2)}, unique viewport hashes ${hashes.size}/${FRAMES}, ` +
        `speed ${maxSpeed.toFixed(1)}→${minSpeed.toFixed(1)}, minLeadGap=${minLeadGap.toFixed(2)}, brake=${sawBrake}`,
    );
    if (hashes.size < 16) throw new Error(`Too few unique frames: ${hashes.size}`);
    if (Math.abs(lastZ - firstZ) < 1.5) throw new Error("Insufficient ego motion");
    if (minLeadGap < MIN_LEAD_GAP_M) {
      throw new Error(`Drive-through: min leadGap ${minLeadGap.toFixed(2)} m < ${MIN_LEAD_GAP_M}`);
    }
    if (!sawBrake) throw new Error("Clip never selected BRAKE / Slow for hazard (action=3)");
    if (maxSpeed - minSpeed < 6) {
      throw new Error(`Speed did not drop enough for a visible slow (${maxSpeed.toFixed(1)} → ${minSpeed.toFixed(1)})`);
    }

    const t0 = path.join(outDir, "motion_t0.png");
    const tmid = path.join(outDir, "motion_tmid.png");
    const tend = path.join(outDir, "motion_tend.png");
    fs.copyFileSync(path.join(framesDir, "frame_0000.png"), t0);
    fs.copyFileSync(path.join(framesDir, `frame_${String(Math.floor((FRAMES - 1) / 2)).padStart(4, "0")}.png`), tmid);
    fs.copyFileSync(path.join(framesDir, `frame_${String(FRAMES - 1).padStart(4, "0")}.png`), tend);
    makeStrip(t0, tmid, tend, path.join(outDir, "motion_strip.png"));

    const mp4 = path.join(outDir, "fsd_town_playback_demo.mp4");
    console.log("encode MP4 libx264 yuv420p +faststart...");
    encodeMp4(framesDir, mp4);

    const gif = path.join(outDir, "fsd_town_playback_demo.gif");
    console.log("encode GIF (palette)...");
    tryEncodeGif(framesDir, gif);

    console.log("verify artifacts...");
    const result = verifyArtifactDir(outDir);
    console.log(`  MP4 ${result.motion.frames} frames, 3 distinct hashes`);
    for (const h of result.motion.hashes) console.log(`    n=${h.n} ${h.sha256}`);
    console.log(`  GIF usable: ${result.gifOk}`);

    for (const name of fs.readdirSync(outDir)) {
      const p = path.join(outDir, name);
      if (fs.statSync(p).isFile()) console.log(`  ${name}: ${fs.statSync(p).size} bytes`);
    }
  } finally {
    await browser.close();
    server?.close();
  }
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
