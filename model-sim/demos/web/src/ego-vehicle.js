import * as THREE from "https://cdn.jsdelivr.net/npm/three@0.172.0/build/three.module.js";

function addWheels(group, track, wheelbase, trimMat, rimMat) {
  for (const sx of [-track / 2, track / 2]) {
    for (const z of [wheelbase * 0.34, -wheelbase * 0.34]) {
      const tire = new THREE.Mesh(
        new THREE.CylinderGeometry(0.33, 0.33, 0.24, 20),
        trimMat,
      );
      tire.rotation.z = Math.PI / 2;
      tire.position.set(sx, 0.33, z);
      group.add(tire);

      const rim = new THREE.Mesh(
        new THREE.CylinderGeometry(0.18, 0.18, 0.26, 12),
        rimMat,
      );
      rim.rotation.z = Math.PI / 2;
      rim.position.set(sx, 0.33, z);
      group.add(rim);
    }
  }
}

function buildSedan(options) {
  const {
    bodyColor = "#eef1f5",
    accentColor = "#111318",
    glassColor = "#1a2533",
    scale = 1,
    width = 1.92,
    height = 0.62,
    length = 4.35,
  } = options;

  const group = new THREE.Group();
  const bodyMat = new THREE.MeshStandardMaterial({
    color: bodyColor,
    metalness: 0.42,
    roughness: 0.24,
  });
  const glassMat = new THREE.MeshStandardMaterial({
    color: glassColor,
    metalness: 0.85,
    roughness: 0.08,
    transparent: true,
    opacity: 0.78,
  });
  const trimMat = new THREE.MeshStandardMaterial({ color: accentColor, roughness: 0.55 });
  const rimMat = new THREE.MeshStandardMaterial({ color: "#b8bec8", metalness: 0.7, roughness: 0.25 });
  const chromeMat = new THREE.MeshStandardMaterial({ color: "#d8dde4", metalness: 0.9, roughness: 0.15 });

  const s = scale;
  const w = width * s;
  const h = height * s;
  const l = length * s;

  const lower = new THREE.Mesh(new THREE.BoxGeometry(w, h * 0.72, l * 0.88), bodyMat);
  lower.position.y = 0.52 * s;
  group.add(lower);

  const belt = new THREE.Mesh(new THREE.BoxGeometry(w * 0.94, h * 0.18, l * 0.72), bodyMat);
  belt.position.set(0, 0.78 * s, -l * 0.04);
  group.add(belt);

  const hood = new THREE.Mesh(new THREE.BoxGeometry(w * 0.88, h * 0.28, l * 0.28), bodyMat);
  hood.position.set(0, 0.66 * s, l * 0.28);
  hood.rotation.x = -0.08;
  group.add(hood);

  const nose = new THREE.Mesh(new THREE.BoxGeometry(w * 0.82, h * 0.22, l * 0.12), bodyMat);
  nose.position.set(0, 0.58 * s, l * 0.44);
  group.add(nose);

  const cabin = new THREE.Mesh(new THREE.BoxGeometry(w * 0.84, h * 0.62, l * 0.42), bodyMat);
  cabin.position.set(0, 0.98 * s, -l * 0.04);
  group.add(cabin);

  const roof = new THREE.Mesh(
    new THREE.BoxGeometry(w * 0.72, h * 0.18, l * 0.34),
    bodyMat,
  );
  roof.position.set(0, 1.18 * s, -l * 0.06);
  group.add(roof);

  const windshield = new THREE.Mesh(new THREE.BoxGeometry(w * 0.76, h * 0.42, l * 0.06), glassMat);
  windshield.position.set(0, 1.0 * s, l * 0.12);
  windshield.rotation.x = -0.42;
  group.add(windshield);

  const rearGlass = new THREE.Mesh(new THREE.BoxGeometry(w * 0.72, h * 0.34, l * 0.05), glassMat);
  rearGlass.position.set(0, 1.02 * s, -l * 0.24);
  rearGlass.rotation.x = 0.34;
  group.add(rearGlass);

  for (const sx of [-1, 1]) {
    const sideGlass = new THREE.Mesh(new THREE.BoxGeometry(0.04, h * 0.34, l * 0.28), glassMat);
    sideGlass.position.set(sx * w * 0.42, 0.98 * s, -l * 0.04);
    group.add(sideGlass);

    const mirror = new THREE.Mesh(new THREE.BoxGeometry(0.08, 0.05, 0.12), trimMat);
    mirror.position.set(sx * w * 0.52, 0.96 * s, l * 0.08);
    group.add(mirror);

    const skirt = new THREE.Mesh(new THREE.BoxGeometry(0.06, h * 0.12, l * 0.62), trimMat);
    skirt.position.set(sx * w * 0.48, 0.38 * s, -l * 0.02);
    group.add(skirt);
  }

  const trunk = new THREE.Mesh(new THREE.BoxGeometry(w * 0.86, h * 0.24, l * 0.18), bodyMat);
  trunk.position.set(0, 0.72 * s, -l * 0.36);
  group.add(trunk);

  const bumperFront = new THREE.Mesh(new THREE.BoxGeometry(w * 0.9, h * 0.12, l * 0.06), trimMat);
  bumperFront.position.set(0, 0.4 * s, l * 0.46);
  group.add(bumperFront);

  const bumperRear = new THREE.Mesh(new THREE.BoxGeometry(w * 0.9, h * 0.12, l * 0.06), trimMat);
  bumperRear.position.set(0, 0.42 * s, -l * 0.46);
  group.add(bumperRear);

  for (const sx of [-0.62, 0.62]) {
    const headlight = new THREE.Mesh(
      new THREE.BoxGeometry(0.22, 0.1, 0.08),
      new THREE.MeshStandardMaterial({
        color: "#f5f8ff",
        emissive: "#aaccff",
        emissiveIntensity: 0.55,
        metalness: 0.2,
        roughness: 0.1,
      }),
    );
    headlight.position.set(sx * w * 0.5, 0.56 * s, l * 0.47);
    group.add(headlight);
  }

  const taillightMat = new THREE.MeshStandardMaterial({
    color: "#ff3344",
    emissive: "#aa1122",
    emissiveIntensity: 0.65,
  });
  const taillight = new THREE.Mesh(new THREE.BoxGeometry(w * 0.78, h * 0.1, l * 0.04), taillightMat);
  taillight.position.set(0, 0.68 * s, -l * 0.47);
  group.add(taillight);

  const grille = new THREE.Mesh(new THREE.BoxGeometry(w * 0.34, h * 0.14, l * 0.03), chromeMat);
  grille.position.set(0, 0.5 * s, l * 0.485);
  group.add(grille);

  addWheels(group, w * 0.98, l, trimMat, rimMat);

  group.userData.width = w;
  group.userData.depth = l;
  return group;
}

export function createEgoVehicle() {
  return buildSedan({
    bodyColor: "#eef1f5",
    accentColor: "#111318",
    glassColor: "#1a2533",
    scale: 1,
    width: 1.92,
    height: 0.62,
    length: 4.35,
  });
}

export function createNpcVehicle(color = "#5a7a96") {
  const group = buildSedan({
    bodyColor: color,
    accentColor: "#2a3038",
    glassColor: "#243040",
    scale: 0.92,
    width: 1.75,
    height: 0.58,
    length: 3.85,
  });
  group.userData.kind = "vehicle";
  group.userData.label = "Vehicle";
  return group;
}

export function createPedestrian(shirt = "#c27d55") {
  const group = new THREE.Group();
  const torso = new THREE.Mesh(
    new THREE.CapsuleGeometry(0.18, 0.45, 6, 10),
    new THREE.MeshStandardMaterial({ color: shirt }),
  );
  torso.position.y = 0.95;
  group.add(torso);
  const head = new THREE.Mesh(
    new THREE.SphereGeometry(0.16, 10, 10),
    new THREE.MeshStandardMaterial({ color: "#dec060" }),
  );
  head.position.y = 1.45;
  group.add(head);
  group.userData.kind = "pedestrian";
  group.userData.label = "Pedestrian";
  return group;
}
