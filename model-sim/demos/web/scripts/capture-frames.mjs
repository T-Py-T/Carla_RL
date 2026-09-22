#!/usr/bin/env node
/** Capture frame burst from running Vite dev server (port 5173). */
import fs from "node:fs";
import path from "node:path";
import { spawnSync } from "node:child_process";
import { chromium } from "playwright";

const outDir = process.argv[2] ?? "/workspace/docs/pr-114-artifacts";
const port = process.argv[3] ?? "5173";

async function main() {
  fs.mkdirSync(outDir, { recursive: true });
  const framesDir = path.join(outDir, "threejs_frames");
  fs.mkdirSync(framesDir, { recursive: true });

  const browser = await chromium.launch({ headless: true, channel: "chrome" });
  const page = await browser.newPage({ viewport: { width: 1280, height: 720 } });

  const url = `http://127.0.0.1:${port}/index.html?w=1280&h=720&proceduralTraffic=1&procedural=1`;
  console.log("Loading", url);
  await page.goto(url, { waitUntil: "load", timeout: 30000 });
  await page.waitForFunction(() => typeof window.stepSimulation === "function", { timeout: 15000 });
  await page.evaluate(() => window.waitForDemoReady());

  for (let i = 0; i < 24; i++) await page.evaluate(() => window.stepSimulation());
  await page.screenshot({ path: path.join(outDir, "after_fsd_overlay.png"), type: "png" });
  console.log("Saved still");

  const frameCount = 48;
  for (let i = 0; i < frameCount; i++) {
    await page.evaluate(() => window.stepSimulation());
    await page.screenshot({
      path: path.join(framesDir, `frame_${String(i).padStart(4, "0")}.png`),
      type: "png",
    });
    if (i % 12 === 0) console.log(`frame ${i}/${frameCount}`);
  }

  await browser.close();

  spawnSync(
    "ffmpeg",
    ["-y", "-framerate", "12", "-i", path.join(framesDir, "frame_%04d.png"), "-c:v", "libx264", "-pix_fmt", "yuv420p", path.join(outDir, "fsd_town_playback_demo.mp4")],
    { stdio: "inherit" },
  );
  spawnSync(
    "ffmpeg",
    ["-y", "-framerate", "12", "-i", path.join(framesDir, "frame_%04d.png"), "-loop", "0", path.join(outDir, "fsd_town_playback_demo.gif")],
    { stdio: "inherit" },
  );
  console.log("Done:", outDir);
}

main().catch((e) => { console.error(e); process.exit(1); });
