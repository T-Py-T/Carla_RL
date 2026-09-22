#!/usr/bin/env node
/**
 * PR proof capture — Playwright viewport screenshots against Vite/static dist.
 * Uses a fresh page per animation frame (Playwright hangs on 2nd screenshot otherwise).
 * No Cursor IDE chrome.
 */
import crypto from "node:crypto";
import fs from "node:fs";
import http from "node:http";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { spawnSync } from "node:child_process";
import { chromium } from "playwright";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const webRoot = path.resolve(__dirname, "..");
const root = path.join(webRoot, "dist");
const outDir = process.argv[2] ?? path.resolve(webRoot, "../../docs/pr-114-artifacts");
const PORT = process.env.CAPTURE_PORT || process.argv[3] || "";
const WIDTH = 1280;
const HEIGHT = 720;
const FRAMES = 48;
const FPS = 12;
const WARMUP_START = 22;
const WARMUP_PER_FRAME = 2;

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

function md5(buf) {
  return crypto.createHash("md5").update(buf).digest("hex");
}

async function warmupPage(page, url, steps) {
  await page.goto(url, { waitUntil: "domcontentloaded" });
  await page.waitForFunction(() => typeof window.stepSimulation === "function");
  return page.evaluate(async (n) => {
    await window.waitForDemoReady();
    for (let i = 0; i < n; i++) window.stepSimulation();
    return window.demoMotionSample();
  }, steps);
}

async function main() {
  let server = null;
  let port = PORT;
  if (!port) {
    if (!fs.existsSync(root)) spawnSync("npm", ["run", "build"], { cwd: webRoot, stdio: "inherit" });
    server = await startStaticServer();
    port = server.address().port;
  }

  fs.mkdirSync(outDir, { recursive: true });
  const framesDir = path.join(outDir, "threejs_frames");
  fs.rmSync(framesDir, { recursive: true, force: true });
  fs.mkdirSync(framesDir, { recursive: true });

  const browser = await chromium.launch({ headless: true });
  const ctx = await browser.newContext({ viewport: { width: WIDTH, height: HEIGHT } });
  const q = `w=${WIDTH}&h=${HEIGHT}&proceduralTraffic=1&procedural=1&capture=1`;
  const host = `http://127.0.0.1:${port}/index.html?${q}`;

  try {
    console.log("before_plain_ego...");
    const pageBefore = await ctx.newPage();
    const plainMotion = await warmupPage(pageBefore, `${host}&plain=1`, 18);
    await pageBefore.screenshot({ path: path.join(outDir, "before_plain_ego.png"), type: "png", timeout: 30000 });
    console.log(`  plain egoZ=${plainMotion.egoZ.toFixed(2)}`);
    await pageBefore.close();

    console.log("after_fsd_overlay still...");
    const pageStill = await ctx.newPage();
    const stillMotion = await warmupPage(pageStill, host, 44);
    await pageStill.screenshot({ path: path.join(outDir, "after_fsd_overlay.png"), type: "png", timeout: 30000 });
    console.log(`  fsd egoZ=${stillMotion.egoZ.toFixed(2)} tracks=${stillMotion.tracks}`);
    await pageStill.close();

    console.log(`rollout ${FRAMES} frames (fresh page each)...`);
    const hashes = new Set();
    let firstZ = 0;
    let lastZ = 0;

    for (let i = 0; i < FRAMES; i++) {
      const steps = WARMUP_START + i * WARMUP_PER_FRAME;
      const page = await ctx.newPage();
      const motion = await warmupPage(page, host, steps);
      const buf = await page.screenshot({ type: "png", timeout: 30000 });
      fs.writeFileSync(path.join(framesDir, `frame_${String(i).padStart(4, "0")}.png`), buf);
      hashes.add(md5(buf));
      if (i === 0) firstZ = motion.egoZ;
      lastZ = motion.egoZ;
      if (i % 12 === 0) {
        console.log(`  frame ${i}: steps=${steps} egoZ=${motion.egoZ.toFixed(2)} unique=${hashes.size}`);
      }
      await page.close();
    }

    console.log(`egoZ ${firstZ.toFixed(2)} → ${lastZ.toFixed(2)}, unique hashes ${hashes.size}/${FRAMES}`);
    if (hashes.size < 10) throw new Error(`Too few unique frames: ${hashes.size}`);
    if (Math.abs(lastZ - firstZ) < 2) throw new Error("Insufficient ego motion");

    const mp4 = path.join(outDir, "fsd_town_playback_demo.mp4");
    spawnSync(
      "ffmpeg",
      [
        "-y",
        "-framerate",
        String(FPS),
        "-i",
        path.join(framesDir, "frame_%04d.png"),
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        mp4,
      ],
      { stdio: "inherit" },
    );

    const gif = path.join(outDir, "fsd_town_playback_demo.gif");
    spawnSync(
      "ffmpeg",
      [
        "-y",
        "-framerate",
        String(FPS),
        "-i",
        path.join(framesDir, "frame_%04d.png"),
        "-vf",
        "fps=10,scale=960:-1:flags=lanczos,split[s0][s1];[s0]palettegen=stats_mode=diff[p];[s1][p]paletteuse",
        "-loop",
        "0",
        gif,
      ],
      { stdio: "inherit" },
    );

    for (const name of [
      "before_plain_ego.png",
      "after_fsd_overlay.png",
      "fsd_town_playback_demo.gif",
      "fsd_town_playback_demo.mp4",
    ]) {
      const p = path.join(outDir, name);
      console.log(`  ${name}: ${fs.statSync(p).size} bytes`);
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
