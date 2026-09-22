#!/usr/bin/env node
import http from "node:http";
import fs from "node:fs";
import path from "node:path";

const outDir = process.argv[2] ?? "/workspace/docs/pr-114-artifacts";
const framesDir = path.join(outDir, "threejs_frames");
fs.mkdirSync(framesDir, { recursive: true });

let count = 0;

const server = http.createServer((req, res) => {
  res.setHeader("Access-Control-Allow-Origin", "*");
  res.setHeader("Access-Control-Allow-Methods", "POST, OPTIONS");
  if (req.method === "OPTIONS") {
    res.writeHead(204);
    res.end();
    return;
  }
  let body = "";
  req.on("data", (c) => (body += c));
  req.on("end", () => {
    const base64 = body.replace(/^data:image\/\w+;base64,/, "");
    let file;
    if (req.url === "/still") {
      file = path.join(outDir, "after_fsd_overlay.png");
    } else if (req.url === "/before") {
      file = path.join(outDir, "before_plain_ego.png");
    } else if (req.url === "/frame") {
      file = path.join(framesDir, `frame_${String(count).padStart(4, "0")}.png`);
      count += 1;
    } else {
      res.writeHead(404);
      res.end();
      return;
    }
    fs.writeFileSync(file, Buffer.from(base64, "base64"));
    res.writeHead(200);
    res.end("ok");
  });
});

server.listen(9999, "127.0.0.1", () => console.log("receiver :9999 →", outDir));
