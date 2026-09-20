import * as THREE from "https://cdn.jsdelivr.net/npm/three@0.172.0/build/three.module.js";

export function createEgoVehicle() {
  const group = new THREE.Group();
  const bodyMat = new THREE.MeshStandardMaterial({
    color: "#eef1f5",
    metalness: 0.35,
    roughness: 0.28,
  });
  const glassMat = new THREE.MeshStandardMaterial({
    color: "#1a2533",
    metalness: 0.8,
    roughness: 0.1,
    transparent: true,
    opacity: 0.82,
  });
  const trimMat = new THREE.MeshStandardMaterial({ color: "#111318", roughness: 0.5 });

  const body = new THREE.Mesh(new THREE.BoxGeometry(1.92, 0.62, 4.35), bodyMat);
  body.position.y = 0.58;
  group.add(body);

  const cabin = new THREE.Mesh(new THREE.BoxGeometry(1.62, 0.52, 2.1), glassMat);
  cabin.position.set(0, 0.98, -0.15);
  group.add(cabin);

  const nose = new THREE.Mesh(new THREE.BoxGeometry(1.75, 0.35, 1.05), bodyMat);
  nose.position.set(0, 0.52, 2.05);
  group.add(nose);

  for (const sx of [-0.96, 0.96]) {
    const wheel = new THREE.Mesh(
      new THREE.CylinderGeometry(0.34, 0.34, 0.22, 18),
      trimMat,
    );
    wheel.rotation.z = Math.PI / 2;
    wheel.position.set(sx, 0.34, 1.25);
    group.add(wheel);
    const wheel2 = wheel.clone();
    wheel2.position.z = -1.35;
    group.add(wheel2);
  }

  const taillight = new THREE.Mesh(
    new THREE.BoxGeometry(1.5, 0.08, 0.05),
    new THREE.MeshStandardMaterial({
      color: "#ff3344",
      emissive: "#aa1122",
      emissiveIntensity: 0.6,
    }),
  );
  taillight.position.set(0, 0.72, -2.18);
  group.add(taillight);

  group.userData.width = 1.92;
  group.userData.depth = 4.35;
  return group;
}

export function createNpcVehicle(color = "#5a7a96") {
  const group = new THREE.Group();
  const body = new THREE.Mesh(
    new THREE.BoxGeometry(1.75, 0.58, 3.8),
    new THREE.MeshStandardMaterial({ color, metalness: 0.25, roughness: 0.45 }),
  );
  body.position.y = 0.52;
  group.add(body);
  const cabin = new THREE.Mesh(
    new THREE.BoxGeometry(1.45, 0.45, 1.7),
    new THREE.MeshStandardMaterial({ color: "#243040", roughness: 0.2 }),
  );
  cabin.position.set(0, 0.92, -0.1);
  group.add(cabin);
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
