/**
 * Decode the truncated/Draco Model Y GLB to an uncompressed GLB so
 * headless Chromium (Playwright) can load Tesla Model Y without WASM Draco.
 */
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import draco3d from "draco3dgltf";

const input = process.argv[2];
const output = process.argv[3];
if (!input || !output) {
  console.error("usage: undraco-modely.mjs <in.glb> <out.glb>");
  process.exit(1);
}

function readGlb(filePath) {
  const buf = fs.readFileSync(filePath);
  const jsonLen = buf.readUInt32LE(12);
  const json = JSON.parse(buf.subarray(20, 20 + jsonLen).toString("utf8"));
  let binStart = 20 + jsonLen;
  if (binStart % 4) binStart += 4 - (binStart % 4);
  const bin = buf.subarray(binStart + 8);
  return { json, bin };
}

function writeGlb(json, bin) {
  const jsonBuf = Buffer.from(JSON.stringify(json));
  const jsonPad = (4 - (jsonBuf.length % 4)) % 4;
  const jsonChunk = jsonBuf.length + jsonPad;
  const binPad = (4 - (bin.length % 4)) % 4;
  const binChunk = bin.length + binPad;
  const total = 12 + 8 + jsonChunk + 8 + binChunk;
  const out = Buffer.alloc(total);
  out.write("glTF", 0);
  out.writeUInt32LE(2, 4);
  out.writeUInt32LE(total, 8);
  out.writeUInt32LE(jsonChunk, 12);
  out.write("JSON", 16);
  jsonBuf.copy(out, 20);
  out.fill(0x20, 20 + jsonBuf.length, 20 + jsonChunk);
  const binHeader = 20 + jsonChunk;
  out.writeUInt32LE(binChunk, binHeader);
  out.write("BIN\0", binHeader + 4);
  bin.copy(out, binHeader + 8);
  return out;
}

let decoderModule = await draco3d.createDecoderModule();
let decoder = new decoderModule.Decoder();

async function resetDecoder() {
  decoderModule = await draco3d.createDecoderModule();
  decoder = new decoderModule.Decoder();
}

function decodeMesh(bytes) {
  const buffer = new decoderModule.DecoderBuffer();
  const copy = Uint8Array.from(bytes);
  buffer.Init(new Int8Array(copy.buffer, copy.byteOffset, copy.byteLength), copy.byteLength);
  const geomType = decoder.GetEncodedGeometryType(buffer);
  if (geomType !== decoderModule.TRIANGULAR_MESH) {
    decoderModule.destroy(buffer);
    throw new Error(`unsupported draco geometry ${geomType}`);
  }
  const mesh = new decoderModule.Mesh();
  const status = decoder.DecodeBufferToMesh(buffer, mesh);
  if (!status.ok()) {
    const msg = status.error_msg();
    decoderModule.destroy(mesh);
    decoderModule.destroy(buffer);
    throw new Error(msg);
  }
  const numFaces = mesh.num_faces();
  const numPoints = mesh.num_points();
  const indices = new Uint32Array(numFaces * 3);
  const ia = new decoderModule.DracoInt32Array();
  for (let f = 0; f < numFaces; f++) {
    decoder.GetFaceFromMesh(mesh, f, ia);
    indices[f * 3] = ia.GetValue(0);
    indices[f * 3 + 1] = ia.GetValue(1);
    indices[f * 3 + 2] = ia.GetValue(2);
  }
  decoderModule.destroy(ia);

  const attrById = {};
  const attrCount = mesh.num_attributes();
  for (let a = 0; a < attrCount; a++) {
    const attr = decoder.GetAttribute(mesh, a);
    const uniqueId = attr.unique_id();
    const components = attr.num_components();
    const arr = new decoderModule.DracoFloat32Array();
    decoder.GetAttributeFloatForAllPoints(mesh, attr, arr);
    const out = new Float32Array(numPoints * components);
    for (let i = 0; i < out.length; i++) out[i] = arr.GetValue(i);
    decoderModule.destroy(arr);
    attrById[uniqueId] = { floats: out, components };
  }

  decoderModule.destroy(buffer);
  decoderModule.destroy(mesh);
  return { indices, attrById, numPoints };
}

const { json, bin } = readGlb(input);
const newBinParts = [];
let newOffset = 0;
const newAccessors = [];
const newBufferViews = [];

const pushBytes = (u8, target) => {
  const pad = (4 - (u8.byteLength % 4)) % 4;
  newBinParts.push(Buffer.from(u8.buffer, u8.byteOffset, u8.byteLength));
  if (pad) newBinParts.push(Buffer.alloc(pad));
  const viewIndex = newBufferViews.length;
  newBufferViews.push({
    buffer: 0,
    byteOffset: newOffset,
    byteLength: u8.byteLength,
    target,
  });
  newOffset += u8.byteLength + pad;
  return viewIndex;
};

const pushAccessor = (accessor, viewIndex) => {
  const next = { ...accessor, bufferView: viewIndex };
  delete next.byteOffset;
  const idx = newAccessors.length;
  newAccessors.push(next);
  return idx;
};

let decoded = 0;
for (const mesh of json.meshes) {
  const kept = [];
  for (const prim of mesh.primitives) {
    const draco = prim.extensions?.KHR_draco_mesh_compression;
    if (!draco) {
      kept.push(prim);
      continue;
    }
    const view = json.bufferViews[draco.bufferView];
    const end = Math.min(bin.length, view.byteOffset + view.byteLength);
    const bytes = bin.subarray(view.byteOffset, end);
    let decodedMesh;
    try {
      decodedMesh = decodeMesh(bytes);
    } catch (err) {
      console.warn(`skip ${mesh.name ?? "mesh"}: ${err.message}`);
      await resetDecoder();
      continue;
    }
    kept.push(prim);
    const { indices, attrById, numPoints } = decodedMesh;

    const indexView = pushBytes(new Uint8Array(indices.buffer), 34963);
    prim.indices = pushAccessor(
      { ...json.accessors[prim.indices], count: indices.length, componentType: 5125, type: "SCALAR" },
      indexView,
    );

    for (const [name, uniqueId] of Object.entries(draco.attributes)) {
      const oldAcc = json.accessors[prim.attributes[name]];
      const decoded = attrById[uniqueId];
      if (!decoded) throw new Error(`missing draco attr ${name} id=${uniqueId}`);
      const viewIndex = pushBytes(new Uint8Array(decoded.floats.buffer), 34962);
      prim.attributes[name] = pushAccessor({ ...oldAcc, count: numPoints }, viewIndex);
    }
    delete prim.extensions;
    decoded += 1;
  }
  mesh.primitives = kept;
}

json.accessors = newAccessors;
json.bufferViews = newBufferViews;
json.buffers = [{ byteLength: newOffset }];
delete json.extensionsRequired;
json.extensionsUsed = (json.extensionsUsed || []).filter((e) => e !== "KHR_draco_mesh_compression");
if (!json.extensionsUsed.length) delete json.extensionsUsed;

const newBin = Buffer.concat(newBinParts);
fs.mkdirSync(path.dirname(output), { recursive: true });
fs.writeFileSync(output, writeGlb(json, newBin));
console.log(`decoded ${decoded} meshes → ${output} (${newBin.length} bin bytes)`);
