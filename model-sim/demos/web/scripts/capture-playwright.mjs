#!/usr/bin/env node
/** Playwright-based proof capture (works with GPU/software GL in this env). */
import fs from "node:fs";
import http from "node:http";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { spawnSync } from "node:child_process";
import { chromium } from "playwright";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const webRoot = path.resolve(__dirname, "..");
const root = path.join(webRoot, "dist");
const outDir = process.argv[2] ?? "/workspace/docs/pr-114-artifacts";

if (!fs.existsSync(root)) {
  spawnSync("npm", ["run", "build"], { cwd: webRoot, stdio: "inherit" });
}

function contentType(filePath) {
  if (filePath.endsWith(".css")) return "text/css";
  if (filePath.endsWith(".js")) return "text/javascript";
  if (filePath.endsWith(".html")) return "text/html";
  if (filePath.endsWith(".png")) return "image/png";
  if (filePath.endsWith(".glb")) return "model/gltf-binary";
  if (filePath.endsWith(".wasm")) return "application/wasm";
  if (filePath.endsWith(".jpg")) return "image/jpeg";
  return "application/octet-stream";
}

function startServer() {
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

async function main() {
  fs.mkdirSync(outDir, { recursive: true });
  const framesDir = path.join(outDir, "threejs_frames");
  fs.mkdirSync(framesDir, { recursive: true });

  const server = await startServer();
  const port = server.address().port;

  const browser = await chromium.launch({ headless: true });
  const page = await browser.newPage({ viewport: { width: 1280, height: 720 } });

  const url = `http://127.0.0.1:${port}/index.html?w=1280&h=720&proceduralTraffic=1&procedural=1`;
  await page.goto(url, { waitUntil: "load", timeout: 45000 });
  await page.waitForFunction(() => typeof window.stepSimulation === "function");
  await page.evaluate(() => window.waitForDemoReady());

  for (let i = 0; i < 20; i++) await page.evaluate(() => window.stepSimulation());

  await page.screenshot({ path: path.join(outDir, "after_fsd_overlay.png"), type: "png" });

  const frameCount = 60;
  for (let i = 0; i < frameCount; i++) {
    await page.evaluate(() => window.stepSimulation());
    await page.screenshot({
      path: path.join(framesDir, `frame_${String(i).padStart(4, "0")}.png`),
      type: "png",
    });
  }

  await browser.close();
  server.close();

  const mp4 = path.join(outDir, "fsd_town_playback_demo.mp4");
  spawnSync(
    "ffmpeg",
    ["-y", "-framerate", "12", "-i", path.join(framesDir, "frame_%04d.png"), "-c:v", "libx264", "-pix_fmt", "yuv420p", mp4],
    { stdio: "inherit" },
  );

  const gif = path.join(outDir, "fsd_town_playback_demo.gif");
  spawnSync(
    "ffmpeg",
    ["-y", "-framerate", "12", "-i", path.join(framesDir, "frame_%04d.png"), "-loop", "0", gif],
    { stdio: "inherit" },
  );

  console.log("Artifacts:", outDir);
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
