#!/usr/bin/env node
import fs from "node:fs";
import http from "node:http";
import path from "node:path";
import { fileURLToPath } from "node:url";
import puppeteer from "puppeteer";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const root = path.resolve(__dirname, "..");

function parseArgs() {
  const args = process.argv.slice(2);
  const out = {
    outputDir: "/opt/cursor/artifacts/carla-fsd-p0",
    frames: 120,
    fps: 12,
    width: 960,
    height: 540,
    video: "",
  };
  for (let i = 0; i < args.length; i += 2) {
    const key = args[i];
    const value = args[i + 1];
    if (key === "--output-dir") out.outputDir = value;
    if (key === "--frames") out.frames = Number(value);
    if (key === "--fps") out.fps = Number(value);
    if (key === "--width") out.width = Number(value);
    if (key === "--height") out.height = Number(value);
    if (key === "--video") out.video = value;
  }
  return out;
}

function contentType(filePath) {
  if (filePath.endsWith(".css")) return "text/css";
  if (filePath.endsWith(".js")) return "text/javascript";
  if (filePath.endsWith(".html")) return "text/html";
  if (filePath.endsWith(".png")) return "image/png";
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
    server.listen(0, "127.0.0.1", () => {
      const { port } = server.address();
      resolve({ server, port });
    });
  });
}

async function main() {
  const args = parseArgs();
  const framesDir = path.join(args.outputDir, "threejs_frames");
  fs.mkdirSync(framesDir, { recursive: true });

  const { server, port } = await startStaticServer();
  const browser = await puppeteer.launch({
    headless: true,
    args: ["--no-sandbox", "--disable-setuid-sandbox", "--use-gl=angle", "--use-angle=swiftshader"],
  });
  try {
    const page = await browser.newPage();
    await page.setViewport({ width: args.width, height: args.height, deviceScaleFactor: 1 });
    await page.goto(`http://127.0.0.1:${port}/index.html?w=${args.width}&h=${args.height}`, {
      waitUntil: "networkidle0",
    });
    await page.waitForFunction(() => typeof window.stepSimulation === "function");

    for (let i = 0; i < args.frames; i += 1) {
      await page.evaluate(() => window.stepSimulation());
      const framePath = path.join(framesDir, `frame_${String(i).padStart(4, "0")}.png`);
      await page.screenshot({ path: framePath, type: "png" });
    }

    if (args.video) {
      const { spawnSync } = await import("node:child_process");
      const inputPattern = path.join(framesDir, "frame_%04d.png");
      const ffmpeg = spawnSync(
        "ffmpeg",
        ["-y", "-framerate", String(args.fps), "-i", inputPattern, "-c:v", "libx264", "-pix_fmt", "yuv420p", args.video],
        { stdio: "inherit" },
      );
      if (ffmpeg.status !== 0) {
        console.warn("ffmpeg unavailable; leaving PNG frame sequence only.");
      }
    }
  } finally {
    await browser.close();
    server.close();
  }

  console.log(`Captured ${args.frames} Three.js frames to ${framesDir}`);
}

main().catch((error) => {
  console.error(error);
  process.exit(1);
});
