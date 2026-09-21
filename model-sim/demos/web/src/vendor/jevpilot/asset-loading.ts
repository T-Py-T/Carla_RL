/** Vendored from standardagents/jevpilot — src/asset-loading.js */
import { LoadingManager } from "three";

export const assetManager = new LoadingManager();
let idle = true;
const waiting = new Set<() => void>();

assetManager.onStart = () => {
  idle = false;
};
assetManager.onLoad = () => {
  idle = true;
  for (const resolve of waiting) resolve();
  waiting.clear();
};

export function assetsReady(): Promise<void> {
  return idle ? Promise.resolve() : new Promise((resolve) => waiting.add(resolve));
}
