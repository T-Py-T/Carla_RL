#!/usr/bin/env node
/**
 * Verify PR #114 motion media is actually playable and not EOL-corrupted.
 *
 * Checks:
 *  - PNG signature includes 0x0D (stripped when `* text eol=lf` is applied)
 *  - ffmpeg decode of MP4 reports zero errors and the expected frame count
 *  - start / mid / end extracted frames have three distinct SHA-256 hashes
 */
import crypto from "node:crypto";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { spawnSync } from "node:child_process";
import { fileURLToPath } from "node:url";

const PNG_SIG = Buffer.from([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]);
const GIF_SIG = Buffer.from("GIF89a");

function sha256(buf) {
  return crypto.createHash("sha256").update(buf).digest("hex");
}

function run(cmd, args, opts = {}) {
  const r = spawnSync(cmd, args, { encoding: "utf8", ...opts });
  return r;
}

export function assertPngSignature(filePath) {
  const head = fs.readFileSync(filePath).subarray(0, 8);
  if (!head.equals(PNG_SIG)) {
    throw new Error(
      `${filePath}: PNG signature corrupted (got ${head.toString("hex")}, want ${PNG_SIG.toString("hex")}). ` +
        "Git text/eol conversion likely stripped 0x0D.",
    );
  }
}

export function assertGifSignature(filePath) {
  const head = fs.readFileSync(filePath).subarray(0, 6);
  if (!head.equals(GIF_SIG)) {
    throw new Error(`${filePath}: not a GIF89a (${head.toString("hex")})`);
  }
}

export function verifyMp4Motion(mp4Path, { minFrames = 24 } = {}) {
  if (!fs.existsSync(mp4Path)) throw new Error(`Missing MP4: ${mp4Path}`);

  const decode = run("ffmpeg", ["-v", "error", "-i", mp4Path, "-f", "null", "-"]);
  const err = `${decode.stderr || ""}${decode.stdout || ""}`.trim();
  if (decode.status !== 0 || err) {
    throw new Error(`MP4 decode failed (exit ${decode.status}):\n${err}`);
  }

  const probe = run("ffprobe", [
    "-v",
    "error",
    "-count_frames",
    "-select_streams",
    "v:0",
    "-show_entries",
    "stream=nb_read_frames,nb_frames,width,height,codec_name,pix_fmt",
    "-of",
    "json",
    mp4Path,
  ]);
  if (probe.status !== 0) throw new Error(`ffprobe failed: ${probe.stderr}`);
  const info = JSON.parse(probe.stdout);
  const stream = info.streams?.[0] ?? {};
  const frames = Number(stream.nb_read_frames || stream.nb_frames || 0);
  if (frames < minFrames) {
    throw new Error(`MP4 has only ${frames} frames (need >= ${minFrames})`);
  }
  if (stream.pix_fmt && stream.pix_fmt !== "yuv420p") {
    throw new Error(`MP4 pix_fmt is ${stream.pix_fmt}, expected yuv420p`);
  }

  const tmp = fs.mkdtempSync(path.join(os.tmpdir(), "pr114-verify-"));
  const idxs = [0, Math.floor((frames - 1) / 2), frames - 1];
  const hashes = [];
  for (const [i, n] of idxs.entries()) {
    const out = path.join(tmp, `frame_${i}.png`);
    const ext = run("ffmpeg", [
      "-y",
      "-i",
      mp4Path,
      "-vf",
      `select=eq(n\\,${n})`,
      "-frames:v",
      "1",
      "-update",
      "1",
      out,
    ]);
    if (ext.status !== 0 || !fs.existsSync(out)) {
      throw new Error(`Failed to extract frame n=${n}: ${ext.stderr}`);
    }
    assertPngSignature(out);
    hashes.push({ n, sha256: sha256(fs.readFileSync(out)), path: out });
  }

  const unique = new Set(hashes.map((h) => h.sha256));
  if (unique.size !== 3) {
    throw new Error(
      `Motion proof failed: start/mid/end hashes not distinct (${unique.size}/3). ` +
        hashes.map((h) => `n=${h.n} ${h.sha256}`).join(" ; "),
    );
  }

  return { frames, hashes, stream };
}

export function verifyArtifactDir(outDir) {
  const requiredPng = ["before_plain_ego.png", "after_fsd_overlay.png", "motion_strip.png"];
  for (const name of requiredPng) {
    const p = path.join(outDir, name);
    if (!fs.existsSync(p)) throw new Error(`Missing ${p}`);
    assertPngSignature(p);
  }
  for (const name of ["motion_t0.png", "motion_tmid.png", "motion_tend.png"]) {
    const p = path.join(outDir, name);
    if (!fs.existsSync(p)) throw new Error(`Missing ${p}`);
    assertPngSignature(p);
  }

  const mp4 = path.join(outDir, "fsd_town_playback_demo.mp4");
  const motion = verifyMp4Motion(mp4);

  const gif = path.join(outDir, "fsd_town_playback_demo.gif");
  let gifOk = false;
  if (fs.existsSync(gif) && fs.statSync(gif).size > 0) {
    assertGifSignature(gif);
    const g = run("ffprobe", [
      "-v",
      "error",
      "-count_frames",
      "-select_streams",
      "v:0",
      "-show_entries",
      "stream=nb_read_frames,nb_frames",
      "-of",
      "csv=p=0",
      gif,
    ]);
    const n = Number(String(g.stdout).split(",")[0] || 0);
    gifOk = g.status === 0 && n >= 8;
    if (!gifOk) {
      console.warn(`GIF present but weak (${g.stdout.trim()} frames); MP4 + strip are the motion proof.`);
    }
  }

  return { motion, gifOk };
}

if (process.argv[1] === fileURLToPath(import.meta.url)) {
  const outDir = process.argv[2] ?? path.resolve(path.dirname(fileURLToPath(import.meta.url)), "../../../../docs/pr-115-artifacts");
  const result = verifyArtifactDir(outDir);
  console.log("MP4 frames:", result.motion.frames);
  console.log("pix_fmt:", result.motion.stream.pix_fmt, "codec:", result.motion.stream.codec_name);
  for (const h of result.motion.hashes) {
    console.log(`  n=${h.n} sha256=${h.sha256}`);
  }
  console.log("GIF usable:", result.gifOk);
  console.log("OK");
}
