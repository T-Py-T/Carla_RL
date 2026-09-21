import "./styles/jevpilot.css";
import "./styles/playback.css";
import { JevTownDemo } from "./playback/demo";

declare global {
  interface Window {
    JevTownDemo: typeof JevTownDemo;
    demo?: JevTownDemo;
    demoReady?: Promise<void>;
    createDemo: (params?: Record<string, unknown>) => JevTownDemo;
    stepSimulation: () => boolean;
    renderPlainFrame: () => Promise<boolean>;
    waitForDemoReady: () => Promise<void>;
  }
}

window.JevTownDemo = JevTownDemo;

window.createDemo = (params = {}) => {
  if (window.demo?.renderer) window.demo.renderer.dispose();
  if (window.demo?.perception) window.demo.perception.dispose();
  const query = new URLSearchParams(location.search);
  window.demo = new JevTownDemo({
    width: Number(params.width ?? query.get("w") ?? 1280),
    height: Number(params.height ?? query.get("h") ?? 720),
    plain: Boolean(params.plain ?? query.get("plain") === "1"),
    procedural: Boolean(params.procedural ?? query.get("procedural") === "1"),
    proceduralTraffic: Boolean(params.proceduralTraffic ?? query.get("proceduralTraffic") === "1"),
  });
  window.demoReady = window.demo.ready;
  return window.demo;
};

window.stepSimulation = () => {
  if (!window.demo) window.createDemo();
  window.demo!.step();
  return true;
};

window.renderPlainFrame = async () => {
  window.createDemo({ plain: true });
  await window.demo!.ready;
  for (let i = 0; i < 28; i++) window.demo!.step();
  return true;
};

window.waitForDemoReady = () => {
  if (!window.demo) window.createDemo();
  return window.demo!.ready ?? Promise.resolve();
};
