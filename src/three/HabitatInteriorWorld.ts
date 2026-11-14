import * as THREE from 'three';
import {
  InteriorInteractableId,
  INTERIOR_INTERACTABLES,
} from '../game/interiorCollision';
import { MissionResources } from '../game/simulation';
import { createDustParticleSpriteTexture } from './marsTextures';

export interface InteriorUpdateInput {
  resources: MissionResources;
  emergencyActive: boolean;
  moxieCrisisActive: boolean;
  skyColor: THREE.Color;
  zenithColor: THREE.Color;
  nightFactor: number;
}

function createDeckTileTexture(): THREE.CanvasTexture {
  const canvas = document.createElement('canvas');
  canvas.width = 512;
  canvas.height = 512;
  const ctx = canvas.getContext('2d')!;

  ctx.fillStyle = '#1E2532';
  ctx.fillRect(0, 0, 512, 512);

  // Brushed metallic panel grid
  ctx.strokeStyle = '#334155';
  ctx.lineWidth = 4;
  for (let i = 0; i <= 512; i += 128) {
    ctx.beginPath();
    ctx.moveTo(i, 0);
    ctx.lineTo(i, 512);
    ctx.stroke();

    ctx.beginPath();
    ctx.moveTo(0, i);
    ctx.lineTo(512, i);
    ctx.stroke();
  }

  // Subtle corner bolts & anti-slip micro-dots
  ctx.fillStyle = '#475569';
  for (let x = 0; x < 512; x += 128) {
    for (let y = 0; y < 512; y += 128) {
      ctx.fillRect(x + 8, y + 8, 5, 5);
      ctx.fillRect(x + 115, y + 8, 5, 5);
      ctx.fillRect(x + 8, y + 115, 5, 5);
      ctx.fillRect(x + 115, y + 115, 5, 5);
    }
  }

  ctx.strokeStyle = 'rgba(56, 189, 248, 0.14)';
  ctx.lineWidth = 2;
  ctx.strokeRect(20, 20, 472, 472);

  const tex = new THREE.CanvasTexture(canvas);
  tex.wrapS = THREE.RepeatWrapping;
  tex.wrapT = THREE.RepeatWrapping;
  tex.repeat.set(8, 8);
  tex.colorSpace = THREE.SRGBColorSpace;
  return tex;
}

function createConsoleScreenTexture(title: string, subtitle: string, accentHex: string): THREE.CanvasTexture {
  const canvas = document.createElement('canvas');
  canvas.width = 512;
  canvas.height = 256;
  const ctx = canvas.getContext('2d')!;

  ctx.fillStyle = '#070D19';
  ctx.fillRect(0, 0, 512, 256);

  ctx.strokeStyle = accentHex;
  ctx.lineWidth = 4;
  ctx.strokeRect(8, 8, 496, 240);

  ctx.fillStyle = accentHex;
  ctx.font = 'bold 24px monospace';
  ctx.fillText(title, 24, 48);

  ctx.fillStyle = '#CBD5E1';
  ctx.font = '16px monospace';
  ctx.fillText(subtitle, 24, 78);

  // Telemetry wavebars & radar ring
  ctx.strokeStyle = 'rgba(34, 211, 238, 0.45)';
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.arc(415, 145, 58, 0, Math.PI * 2);
  ctx.stroke();
  ctx.beginPath();
  ctx.arc(415, 145, 32, 0, Math.PI * 2);
  ctx.stroke();

  ctx.fillStyle = '#22D3EE';
  for (let i = 0; i < 8; i++) {
    const h = 24 + ((i * 29) % 78);
    ctx.fillRect(28 + i * 34, 210 - h, 22, h);
  }

  const tex = new THREE.CanvasTexture(canvas);
  tex.colorSpace = THREE.SRGBColorSpace;
  return tex;
}

function createSignTexture(line1: string, line2: string, borderColor: string): THREE.CanvasTexture {
  const canvas = document.createElement('canvas');
  canvas.width = 512;
  canvas.height = 128;
  const ctx = canvas.getContext('2d')!;

  ctx.fillStyle = '#0B1220';
  ctx.fillRect(0, 0, 512, 128);
  ctx.strokeStyle = borderColor;
  ctx.lineWidth = 6;
  ctx.strokeRect(6, 6, 500, 116);

  ctx.fillStyle = '#F8FAFC';
  ctx.textAlign = 'center';
  ctx.font = 'bold 34px sans-serif';
  ctx.fillText(line1, 256, 56);

  ctx.fillStyle = borderColor;
  ctx.font = 'bold 22px monospace';
  ctx.fillText(line2, 256, 94);

  const tex = new THREE.CanvasTexture(canvas);
  tex.colorSpace = THREE.SRGBColorSpace;
  return tex;
}

export class HabitatInteriorWorld {
  public readonly scene: THREE.Scene;
  public readonly interactableColliders: THREE.Mesh[] = [];

  private disposables: Array<{ dispose: () => void }> = [];
  private isSoftwareGl: boolean;

  // Dynamic lights & backdrop synced with Mars atmosphere / crises
  private exteriorSkyMat!: THREE.ShaderMaterial;
  private exteriorHorizonColor = new THREE.Color('#D97B54');
  private exteriorZenithColor = new THREE.Color('#B55A48');
  private moxieCrisisLight!: THREE.PointLight;
  private moxieBeaconMat!: THREE.MeshStandardMaterial;
  private moxieCoreMat!: THREE.MeshStandardMaterial;
  private moxieImpeller!: THREE.Group;
  private moxieValveWheel!: THREE.Group;
  private moxieValveSpinBoost = 0;

  // Holographic Mars Table
  private holoSpinGroup!: THREE.Group;
  private holoRingMesh!: THREE.Mesh;

  // Crewmates (Commander Maya & Flight Engineer)
  private crewmates: Array<{
    root: THREE.Group;
    head: THREE.Group;
    leftArm: THREE.Group;
    rightArm: THREE.Group;
    phase: number;
  }> = [];

  // Water-Wall tanks & Bio-Dome racks + mist
  private waterTankMats: THREE.MeshStandardMaterial[] = [];
  private waterTankBubbles: THREE.Mesh[] = [];
  private batteryLedMats: THREE.MeshStandardMaterial[] = [];
  private biodomeCropClusters: THREE.Group[] = [];
  private biodomeMistPoints!: THREE.Points;
  private biodomeMistPositions!: Float32Array;
  private biodomeUvLight!: THREE.PointLight;
  private biodomePulseBoost = 0;

  // Highlight ring for looked-at / nearby interactable
  private promptHighlightRing!: THREE.Mesh;
  private promptHighlightMat!: THREE.MeshBasicMaterial;

  private totalTime = 0;

  constructor(envMap: THREE.Texture | null, isSoftwareGl: boolean) {
    this.isSoftwareGl = isSoftwareGl;
    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color('#B85C43');
    if (envMap) {
      this.scene.environment = envMap;
      this.scene.environmentIntensity = 0.72;
    }

    this.buildInteriorLighting();
    this.buildExteriorPanoramaBackdrop();
    this.buildZoneACommandDome();
    this.buildAirlockCorridors();
    this.buildZoneBMoxieBay();
    this.buildZoneCHydroponicsBiodome();
    this.buildInteractableColliders();
  }

  private track<T extends { dispose: () => void }>(item: T): T {
    this.disposables.push(item);
    return item;
  }

  private buildInteriorLighting(): void {
    // Bright balanced indoor aerospace illumination across all 3 zones
    const hemi = new THREE.HemisphereLight('#E2EEFF', '#283044', this.isSoftwareGl ? 1.32 : 1.12);
    this.scene.add(hemi);

    const sunThroughWindows = new THREE.DirectionalLight('#FFF1DC', 1.15);
    sunThroughWindows.position.set(16, 18, 12);
    this.scene.add(sunThroughWindows);

    if (!this.isSoftwareGl) {
      const zoneADomeLight = new THREE.PointLight('#F8FAFC', 2.0, 24, 1.1);
      zoneADomeLight.position.set(0, 6.2, 0);
      this.scene.add(zoneADomeLight);

      const westAirlockLight = new THREE.PointLight('#BAE6FD', 1.4, 11, 1.2);
      westAirlockLight.position.set(-11.5, 2.6, 0);
      this.scene.add(westAirlockLight);

      const eastAirlockLight = new THREE.PointLight('#D8B4FE', 1.4, 11, 1.2);
      eastAirlockLight.position.set(11.5, 2.6, 0);
      this.scene.add(eastAirlockLight);

      const zoneBCyan = new THREE.PointLight('#38BDF8', 1.8, 18, 1.15);
      zoneBCyan.position.set(-20.0, 4.4, 0);
      this.scene.add(zoneBCyan);
    }

    const holoGlow = new THREE.PointLight('#22D3EE', 1.9, 12, 1.35);
    holoGlow.position.set(0, 2.2, 0);
    this.scene.add(holoGlow);

    // Zone B flashing red/amber crisis emergency light
    this.moxieCrisisLight = new THREE.PointLight('#FF2A2A', 0, 20, 1.2);
    this.moxieCrisisLight.position.set(-20.2, 3.8, -1.0);
    this.scene.add(this.moxieCrisisLight);

    // Zone C Hydroponics Bio-Dome vibrant Magenta/UV grow light
    this.biodomeUvLight = new THREE.PointLight('#E879F9', 3.4, 20, 1.1);
    this.biodomeUvLight.position.set(20.0, 4.5, 0);
    this.scene.add(this.biodomeUvLight);
  }

  private buildExteriorPanoramaBackdrop(): void {
    // Sky dome visible through Zone A panoramic windows & Zone C Bio-Dome glass roof
    const skyGeo = this.track(new THREE.SphereGeometry(120, 32, 18));
    this.exteriorSkyMat = this.track(
      new THREE.ShaderMaterial({
        side: THREE.BackSide,
        depthWrite: false,
        uniforms: {
          uHorizonColor: { value: this.exteriorHorizonColor },
          uZenithColor: { value: this.exteriorZenithColor },
        },
        vertexShader: `
          varying vec3 vDir;
          void main() {
            vDir = normalize(position);
            gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
          }
        `,
        fragmentShader: `
          uniform vec3 uHorizonColor;
          uniform vec3 uZenithColor;
          varying vec3 vDir;
          void main() {
            float h = clamp(vDir.y, 0.0, 1.0);
            vec3 col = mix(uHorizonColor, uZenithColor, pow(h, 0.55));
            gl_FragColor = vec4(col, 1.0);
          }
        `,
      })
    );
    const skyMesh = new THREE.Mesh(skyGeo, this.exteriorSkyMat);
    this.scene.add(skyMesh);

    // Exterior red Martian terrain & Jezero crater rim outside the windows (r = 32..95)
    const groundRingGeo = this.track(new THREE.RingGeometry(27, 108, 72, 4));
    groundRingGeo.rotateX(-Math.PI / 2);
    const gPos = groundRingGeo.attributes.position;
    for (let i = 0; i < gPos.count; i++) {
      const x = gPos.getX(i);
      const z = gPos.getZ(i);
      const r = Math.hypot(x, z);
      const ang = Math.atan2(z, x);
      const rim =
        r > 58
          ? ((r - 58) / 50) * (5.5 + 3.2 * Math.sin(ang * 5.0) + 1.8 * Math.cos(ang * 11.0))
          : Math.sin(x * 0.14) * Math.cos(z * 0.14) * 0.45;
      gPos.setY(i, -0.45 + rim);
    }
    groundRingGeo.computeVertexNormals();
    const marsGroundMat = this.track(
      new THREE.MeshStandardMaterial({
        color: '#A9492C',
        roughness: 0.94,
        metalness: 0.02,
      })
    );
    const outerTerrain = new THREE.Mesh(groundRingGeo, marsGroundMat);
    this.scene.add(outerTerrain);
  }

  private buildZoneACommandDome(): void {
    const deckTex = this.track(createDeckTileTexture());

    const floorMat = this.track(
      new THREE.MeshStandardMaterial({
        map: deckTex,
        color: '#CBD5E1',
        roughness: 0.48,
        metalness: 0.32,
      })
    );
    const upperDeckMat = this.track(
      new THREE.MeshStandardMaterial({
        color: '#334155',
        roughness: 0.42,
        metalness: 0.45,
      })
    );
    const hullWhite = this.track(
      new THREE.MeshStandardMaterial({
        color: '#E2E8F0',
        roughness: 0.36,
        metalness: 0.28,
      })
    );
    const darkTitanium = this.track(
      new THREE.MeshStandardMaterial({
        color: '#1E293B',
        roughness: 0.35,
        metalness: 0.78,
      })
    );
    const cyanTrimMat = this.track(
      new THREE.MeshStandardMaterial({
        color: '#38BDF8',
        emissive: '#0284C7',
        emissiveIntensity: 1.15,
        roughness: 0.22,
      })
    );

    // Main circular deck (r = 9.0)
    const mainFloor = new THREE.Mesh(
      this.track(new THREE.CylinderGeometry(9.0, 9.0, 0.2, 48)),
      floorMat
    );
    mainFloor.position.y = -0.1;
    this.scene.add(mainFloor);

    // Raised outer gallery ring (r = 6.5..9.0, y = 0.0..0.12) with step offset >= 0.02
    const outerRing = new THREE.Mesh(
      this.track(new THREE.RingGeometry(6.55, 8.95, 48)),
      upperDeckMat
    );
    outerRing.rotation.x = -Math.PI / 2;
    outerRing.position.y = 0.03;
    this.scene.add(outerRing);

    // Glowing cyan central holo floor ring
    const holoFloorRing = new THREE.Mesh(
      this.track(new THREE.RingGeometry(1.98, 2.12, 40)),
      cyanTrimMat
    );
    holoFloorRing.rotation.x = -Math.PI / 2;
    holoFloorRing.position.y = 0.02;
    this.scene.add(holoFloorRing);

    // Lower segmented knee wall (skip West & East corridor openings around angle PI and 0)
    const segCount = 24;
    const wallSegGeo = this.track(new THREE.BoxGeometry(2.45, 1.38, 0.36));
    const pillarGeo = this.track(new THREE.BoxGeometry(0.28, 3.45, 0.38));
    const windowGlassMat = this.track(
      new THREE.MeshStandardMaterial({
        color: '#BAE6FD',
        transparent: true,
        opacity: 0.14,
        roughness: 0.12,
        metalness: 0.1,
        depthWrite: false,
      })
    );
    const windowPaneGeo = this.track(new THREE.BoxGeometry(2.35, 1.98, 0.08));

    for (let i = 0; i < segCount; i++) {
      const ang = (i / segCount) * Math.PI * 2;
      const wx = Math.cos(ang) * 8.75;
      const wz = Math.sin(ang) * 8.75;

      // Keep West (x < -7.5, |z| < 2.2) and East (x > 7.5, |z| < 2.2) corridors open!
      const isWestPortal = wx < -6.5 && Math.abs(wz) < 2.25;
      const isEastPortal = wx > 6.5 && Math.abs(wz) < 2.25;
      if (isWestPortal || isEastPortal) continue;

      const wall = new THREE.Mesh(wallSegGeo, hullWhite);
      wall.position.set(wx, 0.69, wz);
      wall.rotation.y = -ang + Math.PI / 2;
      this.scene.add(wall);

      const pillar = new THREE.Mesh(pillarGeo, darkTitanium);
      pillar.position.set(Math.cos(ang + 0.13) * 8.78, 1.72, Math.sin(ang + 0.13) * 8.78);
      pillar.rotation.y = -(ang + 0.13) + Math.PI / 2;
      this.scene.add(pillar);

      // Panoramic curved window panes (leaving upper Martian vista clearly visible)
      const pane = new THREE.Mesh(windowPaneGeo, windowGlassMat);
      pane.position.set(wx, 2.36, wz);
      pane.rotation.y = -ang + Math.PI / 2;
      this.scene.add(pane);
    }

    // Upper fascia ring above panoramic windows
    const fasciaRing = new THREE.Mesh(
      this.track(new THREE.TorusGeometry(8.8, 0.24, 10, 48)),
      hullWhite
    );
    fasciaRing.rotation.x = Math.PI / 2;
    fasciaRing.position.y = 3.45;
    this.scene.add(fasciaRing);

    // Ribbed geodesic ceiling dome overhead (y = 3.45 up to y = 8.9)
    const domeCeilingShellMat = this.track(
      new THREE.MeshStandardMaterial({
        color: '#E2E8F0',
        roughness: 0.42,
        metalness: 0.22,
        side: THREE.BackSide,
        transparent: true,
        opacity: 0.34,
      })
    );
    const domeShell = new THREE.Mesh(
      this.track(new THREE.SphereGeometry(8.85, 32, 14, 0, Math.PI * 2, 0, Math.PI * 0.5)),
      domeCeilingShellMat
    );
    domeShell.position.y = 3.45;
    domeShell.scale.set(1, 0.62, 1);
    this.scene.add(domeShell);

    for (let i = 0; i < 6; i++) {
      const rib = new THREE.Mesh(
        this.track(new THREE.TorusGeometry(8.75, 0.11, 8, 36, Math.PI)),
        darkTitanium
      );
      rib.position.y = 3.45;
      rib.scale.set(1, 0.62, 1);
      rib.rotation.y = (i / 6) * Math.PI;
      this.scene.add(rib);
    }

    // Glowing overhead LED halo ring in Command Dome
    const ceilingRing = new THREE.Mesh(
      this.track(new THREE.TorusGeometry(5.35, 0.09, 8, 36)),
      cyanTrimMat
    );
    ceilingRing.rotation.x = Math.PI / 2;
    ceilingRing.position.y = 7.75;
    this.scene.add(ceilingRing);

    // Directional Bay Signage over West (MOXIE) and East (Bio-Dome) portals + South Airlock Exit
    const westSignMat = this.track(
      new THREE.MeshBasicMaterial({
        map: this.track(createSignTexture('ZONE B: MOXIE BAY', '<- LIFE SUPPORT & WATER-WALL', '#22D3EE')),
      })
    );
    const westSign = new THREE.Mesh(this.track(new THREE.PlaneGeometry(2.8, 0.7)), westSignMat);
    westSign.position.set(-8.15, 3.05, 0);
    westSign.rotation.y = Math.PI / 2;
    this.scene.add(westSign);

    const eastSignMat = this.track(
      new THREE.MeshBasicMaterial({
        map: this.track(createSignTexture('ZONE C: BIO-DOME', 'HYDROPONICS & CROPS ->', '#E879F9')),
      })
    );
    const eastSign = new THREE.Mesh(this.track(new THREE.PlaneGeometry(2.8, 0.7)), eastSignMat);
    eastSign.position.set(8.15, 3.05, 0);
    eastSign.rotation.y = -Math.PI / 2;
    this.scene.add(eastSign);

    // South Airlock Exit Hatch (z = +8.35)
    const airlockFrame = new THREE.Mesh(
      this.track(new THREE.BoxGeometry(2.3, 2.75, 0.42)),
      darkTitanium
    );
    airlockFrame.position.set(0, 1.38, 8.45);
    this.scene.add(airlockFrame);

    const airlockDoor = new THREE.Mesh(
      this.track(new THREE.BoxGeometry(1.65, 2.35, 0.48)),
      hullWhite
    );
    airlockDoor.position.set(0, 1.22, 8.45);
    this.scene.add(airlockDoor);

    const exitSignMat = this.track(
      new THREE.MeshBasicMaterial({
        map: this.track(createSignTexture('MAIN AIRLOCK', '[E] EXIT TO EXTERIOR', '#F59E0B')),
      })
    );
    const exitSign = new THREE.Mesh(this.track(new THREE.PlaneGeometry(2.2, 0.56)), exitSignMat);
    exitSign.position.set(0, 2.85, 8.2);
    exitSign.rotation.y = Math.PI;
    this.scene.add(exitSign);

    // ------------------------------------------------------------------
    // CENTRAL HOLOGRAPHIC MARS MAP TABLE (0, 0, 0)
    // ------------------------------------------------------------------
    const tableGroup = new THREE.Group();
    const pedestal = new THREE.Mesh(
      this.track(new THREE.CylinderGeometry(1.38, 1.56, 0.96, 16)),
      hullWhite
    );
    pedestal.position.y = 0.48;
    tableGroup.add(pedestal);

    const rimTable = new THREE.Mesh(
      this.track(new THREE.TorusGeometry(1.42, 0.08, 10, 32)),
      darkTitanium
    );
    rimTable.rotation.x = Math.PI / 2;
    rimTable.position.y = 0.98;
    tableGroup.add(rimTable);

    const emitterGlass = new THREE.Mesh(
      this.track(new THREE.CylinderGeometry(1.26, 1.26, 0.04, 28)),
      cyanTrimMat
    );
    emitterGlass.position.y = 0.99;
    tableGroup.add(emitterGlass);

    // Rotating 3D Holographic Jezero Crater terrain relief
    this.holoSpinGroup = new THREE.Group();
    this.holoSpinGroup.position.y = 1.36;

    const holoTerrainGeo = this.track(new THREE.CylinderGeometry(1.08, 1.12, 0.24, 28, 4));
    const hPos = holoTerrainGeo.attributes.position;
    for (let i = 0; i < hPos.count; i++) {
      const vx = hPos.getX(i);
      const vy = hPos.getY(i);
      const vz = hPos.getZ(i);
      if (vy > 0.05) {
        const r = Math.hypot(vx, vz);
        const rimBoost = r > 0.75 ? (r - 0.75) * 0.65 : 0;
        const deltaRidge = Math.sin(vx * 4.2) * Math.cos(vz * 4.2) * 0.08;
        hPos.setY(i, vy + rimBoost + deltaRidge);
      }
    }
    holoTerrainGeo.computeVertexNormals();
    const holoMat = this.track(
      new THREE.MeshBasicMaterial({
        color: '#22D3EE',
        wireframe: true,
        transparent: true,
        opacity: 0.72,
      })
    );
    const holoMapMesh = new THREE.Mesh(holoTerrainGeo, holoMat);
    this.holoSpinGroup.add(holoMapMesh);

    const holoOutpostPin = new THREE.Mesh(
      this.track(new THREE.OctahedronGeometry(0.12, 0)),
      this.track(
        new THREE.MeshBasicMaterial({ color: '#F59E0B', transparent: true, opacity: 0.95 })
      )
    );
    holoOutpostPin.position.y = 0.36;
    this.holoSpinGroup.add(holoOutpostPin);

    this.holoRingMesh = new THREE.Mesh(
      this.track(new THREE.TorusGeometry(1.22, 0.025, 8, 36)),
      this.track(new THREE.MeshBasicMaterial({ color: '#38BDF8', transparent: true, opacity: 0.65 }))
    );
    this.holoRingMesh.rotation.x = Math.PI / 2;
    this.holoSpinGroup.add(this.holoRingMesh);

    tableGroup.add(this.holoSpinGroup);
    this.scene.add(tableGroup);

    // ------------------------------------------------------------------
    // ROVER CONTROL TERMINAL DESK (0, 0, -5.8)
    // ------------------------------------------------------------------
    const roverDeskGroup = new THREE.Group();
    roverDeskGroup.position.set(0, 0, -5.8);

    const deskBase = new THREE.Mesh(
      this.track(new THREE.BoxGeometry(2.15, 1.02, 0.92)),
      hullWhite
    );
    deskBase.position.y = 0.51;
    roverDeskGroup.add(deskBase);

    const deskTop = new THREE.Mesh(
      this.track(new THREE.BoxGeometry(2.26, 0.08, 1.02)),
      darkTitanium
    );
    deskTop.position.y = 1.04;
    roverDeskGroup.add(deskTop);

    const screenStand = new THREE.Mesh(
      this.track(new THREE.BoxGeometry(1.85, 1.05, 0.12)),
      darkTitanium
    );
    screenStand.position.set(0, 1.62, -0.22);
    screenStand.rotation.x = 0.14;
    roverDeskGroup.add(screenStand);

    const roverScreenMat = this.track(
      new THREE.MeshBasicMaterial({
        map: this.track(
          createConsoleScreenTexture('ROVER TERMINAL // FHAZ', 'PRESS [E] TO DEPLOY & VIEW CAM', '#22D3EE')
        ),
      })
    );
    const roverScreen = new THREE.Mesh(
      this.track(new THREE.PlaneGeometry(1.72, 0.92)),
      roverScreenMat
    );
    roverScreen.position.set(0, 1.62, -0.15);
    roverScreen.rotation.x = 0.14;
    roverDeskGroup.add(roverScreen);

    this.scene.add(roverDeskGroup);

    // ------------------------------------------------------------------
    // 2 MISSION CONSOLES + 2 ANIMATED LOW-POLY ASTRONAUT CREWMATES
    // ------------------------------------------------------------------
    const buildCrewStation = (
      x: number,
      z: number,
      rotY: number,
      stripeHex: string,
      screenTitle: string,
      phase: number
    ) => {
      const consoleGroup = new THREE.Group();
      consoleGroup.position.set(x, 0, z);
      consoleGroup.rotation.y = rotY;

      const cBody = new THREE.Mesh(
        this.track(new THREE.BoxGeometry(2.0, 0.95, 0.78)),
        darkTitanium
      );
      cBody.position.y = 0.48;
      consoleGroup.add(cBody);

      const cScreen = new THREE.Mesh(
        this.track(new THREE.PlaneGeometry(1.65, 0.72)),
        this.track(
          new THREE.MeshBasicMaterial({
            map: this.track(createConsoleScreenTexture(screenTitle, 'TELEMETRY LINK ACTIVE', stripeHex)),
          })
        )
      );
      cScreen.position.set(0, 1.38, -0.15);
      cScreen.rotation.x = 0.22;
      consoleGroup.add(cScreen);
      this.scene.add(consoleGroup);

      // Low-poly astronaut crewmate standing behind/at console
      const suitMat = hullWhite;
      const stripeMat = this.track(
        new THREE.MeshStandardMaterial({
          color: stripeHex,
          emissive: stripeHex,
          emissiveIntensity: 0.25,
          roughness: 0.35,
        })
      );
      const visorMat = this.track(
        new THREE.MeshStandardMaterial({
          color: '#F59E0B',
          roughness: 0.12,
          metalness: 0.92,
        })
      );

      const crewRoot = new THREE.Group();
      // Stand slightly toward center looking at console
      const dirX = Math.sin(rotY);
      const dirZ = Math.cos(rotY);
      crewRoot.position.set(x + dirX * 0.95, 0, z + dirZ * 0.95);
      crewRoot.rotation.y = rotY + Math.PI; // Face the console screen

      // Boots + Legs
      const legGeo = this.track(new THREE.CylinderGeometry(0.1, 0.12, 0.76, 8));
      const leftLeg = new THREE.Mesh(legGeo, suitMat);
      leftLeg.position.set(-0.15, 0.38, 0);
      const rightLeg = new THREE.Mesh(legGeo, suitMat);
      rightLeg.position.set(0.15, 0.38, 0);
      crewRoot.add(leftLeg, rightLeg);

      // Torso + chest unit + PLSS backpack
      const torso = new THREE.Mesh(
        this.track(new THREE.BoxGeometry(0.46, 0.56, 0.28)),
        suitMat
      );
      torso.position.y = 1.04;
      crewRoot.add(torso);

      const chestBand = new THREE.Mesh(
        this.track(new THREE.BoxGeometry(0.48, 0.14, 0.3)),
        stripeMat
      );
      chestBand.position.y = 1.12;
      crewRoot.add(chestBand);

      const backpack = new THREE.Mesh(
        this.track(new THREE.BoxGeometry(0.36, 0.46, 0.18)),
        darkTitanium
      );
      backpack.position.set(0, 1.06, -0.2);
      crewRoot.add(backpack);

      // Articulated Head + Gold Visor
      const headGroup = new THREE.Group();
      headGroup.position.set(0, 1.48, 0);
      const helmet = new THREE.Mesh(
        this.track(new THREE.SphereGeometry(0.2, 12, 10)),
        suitMat
      );
      headGroup.add(helmet);
      const visor = new THREE.Mesh(
        this.track(new THREE.SphereGeometry(0.175, 10, 8, 0, Math.PI, 0.2, 1.1)),
        visorMat
      );
      visor.position.set(0, 0.02, 0.055);
      headGroup.add(visor);
      crewRoot.add(headGroup);

      // Articulated Left & Right Arms reaching toward console
      const armGeo = this.track(new THREE.CylinderGeometry(0.075, 0.08, 0.48, 8));
      armGeo.translate(0, -0.2, 0);

      const leftArm = new THREE.Group();
      leftArm.position.set(-0.3, 1.24, 0.04);
      leftArm.rotation.x = -0.92;
      leftArm.add(new THREE.Mesh(armGeo, suitMat));
      crewRoot.add(leftArm);

      const rightArm = new THREE.Group();
      rightArm.position.set(0.3, 1.24, 0.04);
      rightArm.rotation.x = -0.92;
      rightArm.add(new THREE.Mesh(armGeo, suitMat));
      crewRoot.add(rightArm);

      this.scene.add(crewRoot);
      this.crewmates.push({ root: crewRoot, head: headGroup, leftArm, rightArm, phase });
    };

    buildCrewStation(-4.35, -4.15, 0.52, '#F59E0B', 'CMDR MAYA // HAB OPS', 0.0);
    buildCrewStation(4.35, -4.15, -0.52, '#22D3EE', 'FLIGHT ENG // ISRU', 2.2);
  }

  private buildAirlockCorridors(): void {
    const floorMat = this.track(
      new THREE.MeshStandardMaterial({ color: '#334155', roughness: 0.5, metalness: 0.55 })
    );
    const hullMat = this.track(
      new THREE.MeshStandardMaterial({ color: '#CBD5E1', roughness: 0.38, metalness: 0.3 })
    );
    const darkRibMat = this.track(
      new THREE.MeshStandardMaterial({ color: '#1E293B', roughness: 0.35, metalness: 0.8 })
    );
    const guideMat = this.track(
      new THREE.MeshStandardMaterial({
        color: '#22D3EE',
        emissive: '#06B6D4',
        emissiveIntensity: 1.0,
      })
    );

    const buildCorridor = (startX: number, endX: number, accentHex: string) => {
      const length = Math.abs(endX - startX);
      const midX = (startX + endX) * 0.5;
      const accentMat = this.track(
        new THREE.MeshStandardMaterial({
          color: accentHex,
          emissive: accentHex,
          emissiveIntensity: 0.9,
        })
      );

      // Floor & Ceiling slabs (clean height separation >= 0.03)
      const floor = new THREE.Mesh(
        this.track(new THREE.BoxGeometry(length, 0.18, 3.6)),
        floorMat
      );
      floor.position.set(midX, -0.09, 0);
      this.scene.add(floor);

      const guide = new THREE.Mesh(
        this.track(new THREE.BoxGeometry(length - 0.6, 0.03, 0.24)),
        guideMat
      );
      guide.position.set(midX, 0.02, 0);
      this.scene.add(guide);

      const ceiling = new THREE.Mesh(
        this.track(new THREE.BoxGeometry(length, 0.22, 3.6)),
        hullMat
      );
      ceiling.position.set(midX, 3.25, 0);
      this.scene.add(ceiling);

      // Side walls
      const wallGeo = this.track(new THREE.BoxGeometry(length, 3.25, 0.22));
      const northWall = new THREE.Mesh(wallGeo, hullMat);
      northWall.position.set(midX, 1.62, -1.75);
      const southWall = new THREE.Mesh(wallGeo, hullMat);
      southWall.position.set(midX, 1.62, 1.75);
      this.scene.add(northWall, southWall);

      // Pressurized bulkhead arch ribs
      const ribCount = 5;
      for (let i = 0; i < ribCount; i++) {
        const rx = startX + ((endX - startX) * (i + 0.5)) / ribCount;
        const leftPost = new THREE.Mesh(
          this.track(new THREE.BoxGeometry(0.24, 3.15, 0.22)),
          darkRibMat
        );
        leftPost.position.set(rx, 1.58, -1.56);
        const rightPost = new THREE.Mesh(
          this.track(new THREE.BoxGeometry(0.24, 3.15, 0.22)),
          darkRibMat
        );
        rightPost.position.set(rx, 1.58, 1.56);
        const topBeam = new THREE.Mesh(
          this.track(new THREE.BoxGeometry(0.24, 0.24, 3.34)),
          darkRibMat
        );
        topBeam.position.set(rx, 3.04, 0);
        const lightStrip = new THREE.Mesh(
          this.track(new THREE.BoxGeometry(0.14, 0.07, 2.2)),
          accentMat
        );
        lightStrip.position.set(rx, 2.9, 0);
        this.scene.add(leftPost, rightPost, topBeam, lightStrip);
      }
    };

    buildCorridor(-15.8, -7.8, '#22D3EE');
    buildCorridor(7.8, 15.8, '#E879F9');
  }

  private buildZoneBMoxieBay(): void {
    const cx = -20.0;
    const deckTex = this.track(createDeckTileTexture());
    const floorMat = this.track(
      new THREE.MeshStandardMaterial({
        map: deckTex,
        color: '#94A3B8',
        roughness: 0.45,
        metalness: 0.4,
      })
    );
    const hullWhite = this.track(
      new THREE.MeshStandardMaterial({ color: '#E2E8F0', roughness: 0.38, metalness: 0.3 })
    );
    const darkMetal = this.track(
      new THREE.MeshStandardMaterial({ color: '#1E293B', roughness: 0.32, metalness: 0.82 })
    );

    // Bay floor
    const floor = new THREE.Mesh(
      this.track(new THREE.CylinderGeometry(5.8, 5.8, 0.2, 32)),
      floorMat
    );
    floor.position.set(cx, -0.1, 0);
    this.scene.add(floor);

    // Bay perimeter walls (skip East corridor portal opening at x > -15.4, |z| < 1.9)
    const wallCount = 16;
    const wallGeo = this.track(new THREE.BoxGeometry(2.35, 3.6, 0.3));
    for (let i = 0; i < wallCount; i++) {
      const ang = (i / wallCount) * Math.PI * 2;
      const wx = cx + Math.cos(ang) * 5.65;
      const wz = Math.sin(ang) * 5.65;
      if (wx > cx + 4.3 && Math.abs(wz) < 2.05) continue;
      const wall = new THREE.Mesh(wallGeo, hullWhite);
      wall.position.set(wx, 1.8, wz);
      wall.rotation.y = -ang + Math.PI / 2;
      this.scene.add(wall);
    }

    // Domed industrial ceiling cap
    const ceiling = new THREE.Mesh(
      this.track(new THREE.CylinderGeometry(5.2, 5.8, 1.4, 24)),
      darkMetal
    );
    ceiling.position.set(cx, 4.3, 0);
    this.scene.add(ceiling);

    // ------------------------------------------------------------------
    // PHYSICAL MOXIE OXYGEN GENERATOR MACHINE (-21.6, 0, -1.5)
    // ------------------------------------------------------------------
    const moxieGroup = new THREE.Group();
    moxieGroup.position.set(-21.6, 0, -1.5);
    moxieGroup.rotation.y = 0.45;

    const moxieSkid = new THREE.Mesh(
      this.track(new THREE.BoxGeometry(2.3, 0.24, 1.7)),
      darkMetal
    );
    moxieSkid.position.y = 0.12;
    moxieGroup.add(moxieSkid);

    const moxieCabinet = new THREE.Mesh(
      this.track(new THREE.BoxGeometry(1.9, 1.95, 1.35)),
      hullWhite
    );
    moxieCabinet.position.y = 1.2;
    moxieGroup.add(moxieCabinet);

    this.moxieCoreMat = this.track(
      new THREE.MeshStandardMaterial({
        color: '#22D3EE',
        emissive: '#06B6D4',
        emissiveIntensity: 1.2,
        roughness: 0.2,
      })
    );
    const electrolysisCore = new THREE.Mesh(
      this.track(new THREE.CylinderGeometry(0.36, 0.36, 1.15, 14)),
      this.moxieCoreMat
    );
    electrolysisCore.position.set(-0.35, 1.28, 0.62);
    moxieGroup.add(electrolysisCore);

    // Spinning CO2 Intake Impeller on front-top
    this.moxieImpeller = new THREE.Group();
    this.moxieImpeller.position.set(0.45, 2.25, 0);
    const bladeGeo = this.track(new THREE.BoxGeometry(0.75, 0.06, 0.14));
    const bladeMat = this.track(
      new THREE.MeshStandardMaterial({ color: '#F59E0B', roughness: 0.3, metalness: 0.6 })
    );
    const b1 = new THREE.Mesh(bladeGeo, bladeMat);
    const b2 = new THREE.Mesh(bladeGeo, bladeMat);
    b2.rotation.y = Math.PI / 2;
    this.moxieImpeller.add(b1, b2);
    moxieGroup.add(this.moxieImpeller);

    // Red/Amber Manual Intake Valve Wheel (spins on [E] manual repair)
    this.moxieValveWheel = new THREE.Group();
    this.moxieValveWheel.position.set(0.42, 1.28, 0.72);
    const valveRim = new THREE.Mesh(
      this.track(new THREE.TorusGeometry(0.24, 0.045, 8, 18)),
      this.track(
        new THREE.MeshStandardMaterial({
          color: '#EF4444',
          emissive: '#B91C1C',
          emissiveIntensity: 0.65,
          roughness: 0.3,
        })
      )
    );
    const valveSpoke = new THREE.Mesh(
      this.track(new THREE.BoxGeometry(0.44, 0.05, 0.05)),
      darkMetal
    );
    const valveSpoke2 = new THREE.Mesh(
      this.track(new THREE.BoxGeometry(0.05, 0.44, 0.05)),
      darkMetal
    );
    this.moxieValveWheel.add(valveRim, valveSpoke, valveSpoke2);
    moxieGroup.add(this.moxieValveWheel);

    // Emergency alarm beacon dome atop MOXIE unit
    this.moxieBeaconMat = this.track(
      new THREE.MeshStandardMaterial({
        color: '#22D3EE',
        emissive: '#06B6D4',
        emissiveIntensity: 0.8,
        roughness: 0.18,
      })
    );
    const alarmBeacon = new THREE.Mesh(
      this.track(new THREE.SphereGeometry(0.18, 12, 10)),
      this.moxieBeaconMat
    );
    alarmBeacon.position.set(-0.62, 2.34, 0.35);
    moxieGroup.add(alarmBeacon);

    const moxieLabel = new THREE.Mesh(
      this.track(new THREE.PlaneGeometry(1.3, 0.42)),
      this.track(
        new THREE.MeshBasicMaterial({
          map: this.track(createSignTexture('MOXIE O2 UNIT', '[E] REPAIR VALVE', '#22D3EE')),
        })
      )
    );
    moxieLabel.position.set(0, 2.65, 0.45);
    moxieGroup.add(moxieLabel);

    this.scene.add(moxieGroup);

    // ------------------------------------------------------------------
    // GLOWING CYAN WATER-WALL SHIELDING TANKS
    // ------------------------------------------------------------------
    const tankPositions: Array<[number, number]> = [
      [-18.5, -4.0],
      [-22.2, 2.3],
    ];
    tankPositions.forEach(([tx, tz], idx) => {
      const tankGroup = new THREE.Group();
      tankGroup.position.set(tx, 0, tz);

      const capGeo = this.track(new THREE.CylinderGeometry(0.68, 0.72, 0.25, 16));
      const baseCap = new THREE.Mesh(capGeo, darkMetal);
      baseCap.position.y = 0.125;
      const topCap = new THREE.Mesh(capGeo, darkMetal);
      topCap.position.y = 2.68;
      tankGroup.add(baseCap, topCap);

      const waterMat = this.track(
        new THREE.MeshStandardMaterial({
          color: '#22D3EE',
          emissive: '#0284C7',
          emissiveIntensity: 0.95,
          transparent: true,
          opacity: 0.78,
          roughness: 0.14,
          metalness: 0.2,
        })
      );
      this.waterTankMats.push(waterMat);

      const col = new THREE.Mesh(
        this.track(new THREE.CylinderGeometry(0.58, 0.58, 2.35, 18)),
        waterMat
      );
      col.position.y = 1.4;
      tankGroup.add(col);

      // Rising tank bubbles
      const bMat = this.track(
        new THREE.MeshBasicMaterial({ color: '#E0F2FE', transparent: true, opacity: 0.85 })
      );
      for (let b = 0; b < 4; b++) {
        const bubble = new THREE.Mesh(this.track(new THREE.SphereGeometry(0.08, 8, 8)), bMat);
        bubble.position.set(
          Math.cos(b * 1.7 + idx) * 0.24,
          0.45 + b * 0.52,
          Math.sin(b * 1.7 + idx) * 0.24
        );
        bubble.userData.baseY = 0.45;
        this.waterTankBubbles.push(bubble);
        tankGroup.add(bubble);
      }

      this.scene.add(tankGroup);
    });

    // ------------------------------------------------------------------
    // BATTERY STORAGE RACKS (-17.6, 0, 3.6)
    // ------------------------------------------------------------------
    const rackGroup = new THREE.Group();
    rackGroup.position.set(-17.6, 0, 3.6);
    rackGroup.rotation.y = -0.55;
    const cabinet = new THREE.Mesh(
      this.track(new THREE.BoxGeometry(1.5, 2.15, 0.85)),
      darkMetal
    );
    cabinet.position.y = 1.08;
    rackGroup.add(cabinet);

    for (let i = 0; i < 4; i++) {
      const ledMat = this.track(
        new THREE.MeshStandardMaterial({
          color: '#10B981',
          emissive: '#10B981',
          emissiveIntensity: 1.1,
        })
      );
      this.batteryLedMats.push(ledMat);
      const ledBar = new THREE.Mesh(
        this.track(new THREE.BoxGeometry(1.15, 0.14, 0.06)),
        ledMat
      );
      ledBar.position.set(0, 0.48 + i * 0.42, 0.43);
      rackGroup.add(ledBar);
    }
    this.scene.add(rackGroup);
  }

  private buildZoneCHydroponicsBiodome(): void {
    const cx = 20.0;
    const deckTex = this.track(createDeckTileTexture());
    const floorMat = this.track(
      new THREE.MeshStandardMaterial({
        map: deckTex,
        color: '#94A3B8',
        roughness: 0.52,
        metalness: 0.28,
      })
    );
    const hullWhite = this.track(
      new THREE.MeshStandardMaterial({ color: '#E2E8F0', roughness: 0.36, metalness: 0.26 })
    );
    const darkMetal = this.track(
      new THREE.MeshStandardMaterial({ color: '#1E293B', roughness: 0.38, metalness: 0.78 })
    );
    const uvStripMat = this.track(
      new THREE.MeshStandardMaterial({
        color: '#F0ABFC',
        emissive: '#D946EF',
        emissiveIntensity: 1.6,
      })
    );

    const floor = new THREE.Mesh(
      this.track(new THREE.CylinderGeometry(6.0, 6.0, 0.2, 32)),
      floorMat
    );
    floor.position.set(cx, -0.1, 0);
    this.scene.add(floor);

    // Lower knee wall around Bio-Dome (skip West corridor opening)
    const wallCount = 16;
    const wallGeo = this.track(new THREE.BoxGeometry(2.4, 1.35, 0.3));
    for (let i = 0; i < wallCount; i++) {
      const ang = (i / wallCount) * Math.PI * 2;
      const wx = cx + Math.cos(ang) * 5.85;
      const wz = Math.sin(ang) * 5.85;
      if (wx < cx - 4.4 && Math.abs(wz) < 2.05) continue;
      const wall = new THREE.Mesh(wallGeo, hullWhite);
      wall.position.set(wx, 0.68, wz);
      wall.rotation.y = -ang + Math.PI / 2;
      this.scene.add(wall);
    }

    // Ribbed Glass Greenhouse Dome Overhead
    const bioGlassMat = this.track(
      new THREE.MeshStandardMaterial({
        color: '#F5D0FE',
        transparent: true,
        opacity: 0.15,
        roughness: 0.14,
        metalness: 0.08,
        side: THREE.BackSide,
        depthWrite: false,
      })
    );
    const domeGlass = new THREE.Mesh(
      this.track(new THREE.SphereGeometry(5.95, 24, 14, 0, Math.PI * 2, 0, Math.PI * 0.48)),
      bioGlassMat
    );
    domeGlass.position.set(cx, 1.25, 0);
    this.scene.add(domeGlass);

    for (let i = 0; i < 4; i++) {
      const rib = new THREE.Mesh(
        this.track(new THREE.TorusGeometry(5.9, 0.08, 8, 28, Math.PI)),
        hullWhite
      );
      rib.position.set(cx, 1.25, 0);
      rib.rotation.y = (i / 4) * Math.PI;
      this.scene.add(rib);
    }

    // ------------------------------------------------------------------
    // VERTICAL HYDROPONICS PLANT RACKS UNDER MAGENTA/UV GROW LIGHTS
    // ------------------------------------------------------------------
    const leafMat = this.track(
      new THREE.MeshStandardMaterial({
        color: '#22C55E',
        emissive: '#10B981',
        emissiveIntensity: 0.45,
        roughness: 0.44,
      })
    );
    const soilMat = this.track(
      new THREE.MeshStandardMaterial({ color: '#1F2937', roughness: 0.9 })
    );

    const rackConfigs: Array<{ x: number; z: number; rotY: number }> = [
      { x: 20.0, z: -2.6, rotY: 0 },
      { x: 22.8, z: 0.0, rotY: Math.PI / 2 },
      { x: 20.0, z: 2.6, rotY: 0 },
    ];

    rackConfigs.forEach((rc) => {
      const rack = new THREE.Group();
      rack.position.set(rc.x, 0, rc.z);
      rack.rotation.y = rc.rotY;

      // 4 vertical uprights
      const postGeo = this.track(new THREE.BoxGeometry(0.1, 2.55, 0.1));
      for (const px of [-1.0, 1.0]) {
        for (const pz of [-0.42, 0.42]) {
          const post = new THREE.Mesh(postGeo, darkMetal);
          post.position.set(px, 1.28, pz);
          rack.add(post);
        }
      }

      // 3 stacked growing shelves
      for (let shelf = 0; shelf < 3; shelf++) {
        const sy = 0.42 + shelf * 0.82;
        const tray = new THREE.Mesh(
          this.track(new THREE.BoxGeometry(2.05, 0.1, 0.88)),
          soilMat
        );
        tray.position.y = sy;
        rack.add(tray);

        const uvBar = new THREE.Mesh(
          this.track(new THREE.BoxGeometry(1.85, 0.05, 0.12)),
          uvStripMat
        );
        uvBar.position.set(0, sy + 0.62, 0);
        rack.add(uvBar);

        const cropsRow = new THREE.Group();
        cropsRow.position.y = sy + 0.06;
        for (let c = 0; c < 5; c++) {
          const cxOffset = -0.74 + c * 0.37;
          const plantCluster = new THREE.Mesh(
            this.track(new THREE.DodecahedronGeometry(0.16, 1)),
            leafMat
          );
          plantCluster.position.set(cxOffset, 0.15, (c % 2 === 0 ? 1 : -1) * 0.12);
          plantCluster.scale.set(1.15, 0.9, 1.15);
          cropsRow.add(plantCluster);
        }
        this.biodomeCropClusters.push(cropsRow);
        rack.add(cropsRow);
      }

      this.scene.add(rack);
    });

    // Mist Particle System in Zone C
    const mistSprite = this.track(createDustParticleSpriteTexture());
    const mistCount = 72;
    this.biodomeMistPositions = new Float32Array(mistCount * 3);
    for (let i = 0; i < mistCount; i++) {
      const ang = (i / mistCount) * Math.PI * 2;
      const rad = 0.8 + ((i * 1.7) % 3.8);
      this.biodomeMistPositions[i * 3] = cx + Math.cos(ang) * rad;
      this.biodomeMistPositions[i * 3 + 1] = 0.35 + ((i * 0.41) % 2.4);
      this.biodomeMistPositions[i * 3 + 2] = Math.sin(ang) * rad;
    }
    const mistGeo = this.track(new THREE.BufferGeometry());
    mistGeo.setAttribute('position', new THREE.BufferAttribute(this.biodomeMistPositions, 3));
    const mistMat = this.track(
      new THREE.PointsMaterial({
        map: mistSprite,
        color: '#F5D0FE',
        size: 0.42,
        transparent: true,
        opacity: 0.45,
        depthWrite: false,
      })
    );
    this.biodomeMistPoints = new THREE.Points(mistGeo, mistMat);
    this.scene.add(this.biodomeMistPoints);
  }

  private buildInteractableColliders(): void {
    const colliderMat = this.track(new THREE.MeshBasicMaterial({ visible: false }));
    for (const item of INTERIOR_INTERACTABLES) {
      const hit = new THREE.Mesh(
        this.track(new THREE.CylinderGeometry(1.35, 1.35, 2.6, 10)),
        colliderMat
      );
      hit.position.set(item.x, 1.3, item.z);
      hit.layers.set(1);
      hit.userData.interactableId = item.id;
      this.interactableColliders.push(hit);
      this.scene.add(hit);
    }

    this.promptHighlightMat = this.track(
      new THREE.MeshBasicMaterial({
        color: '#22D3EE',
        side: THREE.DoubleSide,
        transparent: true,
        opacity: 0.72,
        depthWrite: false,
      })
    );
    const ringGeo = this.track(new THREE.RingGeometry(1.1, 1.32, 28));
    ringGeo.rotateX(-Math.PI / 2);
    this.promptHighlightRing = new THREE.Mesh(ringGeo, this.promptHighlightMat);
    this.promptHighlightRing.visible = false;
    this.scene.add(this.promptHighlightRing);
  }

  public triggerStationFeedback(id: InteriorInteractableId): void {
    if (id === 'moxie_unit') {
      this.moxieValveSpinBoost = 2.4;
    } else if (id === 'crop_racks') {
      this.biodomePulseBoost = 2.4;
    }
  }

  public setHighlightedInteractable(id: InteriorInteractableId | null): void {
    if (!id) {
      this.promptHighlightRing.visible = false;
      return;
    }
    const target = INTERIOR_INTERACTABLES.find((x) => x.id === id);
    if (!target) {
      this.promptHighlightRing.visible = false;
      return;
    }
    this.promptHighlightRing.position.set(target.x, 0.04, target.z);
    this.promptHighlightRing.visible = true;
  }

  public update(dt: number, input: InteriorUpdateInput): void {
    this.totalTime += dt;

    // Sync exterior sky panorama visible through panoramic windows
    this.exteriorHorizonColor.copy(input.skyColor);
    this.exteriorZenithColor.copy(input.zenithColor);
    if (this.scene.background instanceof THREE.Color) {
      this.scene.background.copy(input.skyColor);
    }

    // Rotate central Holographic Mars Map
    if (this.holoSpinGroup) {
      this.holoSpinGroup.rotation.y += dt * 0.42;
      this.holoRingMesh.rotation.z += dt * 0.65;
    }

    // Animate Crewmates (subtle breathing, console typing, head scan)
    for (const c of this.crewmates) {
      const t = this.totalTime + c.phase;
      c.head.rotation.y = Math.sin(t * 0.9) * 0.24;
      c.head.rotation.x = Math.sin(t * 1.4) * 0.07;
      c.leftArm.rotation.x = -0.92 + Math.sin(t * 7.5) * 0.09;
      c.rightArm.rotation.x = -0.92 + Math.cos(t * 8.2) * 0.09;
    }

    // Zone B MOXIE Impeller, Manual Valve Spin, Water-Wall Tank Bubbles & Crisis Flash
    if (this.moxieImpeller) {
      this.moxieImpeller.rotation.y += dt * (input.moxieCrisisActive ? 1.4 : 4.8);
    }
    if (this.moxieValveSpinBoost > 0) {
      this.moxieValveSpinBoost = Math.max(0, this.moxieValveSpinBoost - dt);
      this.moxieValveWheel.rotation.z += dt * 9.5;
    }

    for (let i = 0; i < this.waterTankBubbles.length; i++) {
      const b = this.waterTankBubbles[i];
      b.position.y += dt * 0.55;
      if (b.position.y > 2.45) b.position.y = 0.42;
    }

    // Emergency red/amber flashing point-light in Zone B during active crisis
    if (input.moxieCrisisActive || input.emergencyActive) {
      const flash = 0.5 + 0.5 * Math.sin(this.totalTime * 6.4);
      this.moxieCrisisLight.color.setHex(flash > 0.5 ? 0xff2a2a : 0xff8c00);
      this.moxieCrisisLight.intensity = 1.6 + flash * 4.4;
      this.moxieBeaconMat.color.setHex(0xff2a2a);
      this.moxieBeaconMat.emissive.setHex(0xff2a2a);
      this.moxieBeaconMat.emissiveIntensity = 1.0 + flash * 2.4;
    } else {
      this.moxieCrisisLight.intensity = THREE.MathUtils.lerp(
        this.moxieCrisisLight.intensity,
        0,
        Math.min(1, dt * 6)
      );
      this.moxieBeaconMat.color.setHex(0x22d3ee);
      this.moxieBeaconMat.emissive.setHex(0x06b6d4);
      this.moxieBeaconMat.emissiveIntensity = 0.9;
    }

    // Battery LED bars reflect current Power kWh
    const powerRatio = THREE.MathUtils.clamp(input.resources.powerKwh / 120, 0, 1);
    for (let i = 0; i < this.batteryLedMats.length; i++) {
      const lit = powerRatio >= (i + 0.5) / this.batteryLedMats.length;
      const mat = this.batteryLedMats[i];
      if (lit) {
        mat.color.setHex(powerRatio < 0.25 ? 0xf59e0b : 0x10b981);
        mat.emissive.setHex(powerRatio < 0.25 ? 0xd97706 : 0x10b981);
        mat.emissiveIntensity = 1.15;
      } else {
        mat.color.setHex(0x334155);
        mat.emissive.setHex(0x000000);
        mat.emissiveIntensity = 0.1;
      }
    }

    // Zone C Bio-Dome crop growth scaling + UV pulse + drifting mist particles
    if (this.biodomePulseBoost > 0) {
      this.biodomePulseBoost = Math.max(0, this.biodomePulseBoost - dt);
    }
    const cropScale = THREE.MathUtils.clamp(0.75 + (input.resources.cropGrowth - 1) * 0.14, 0.72, 1.45);
    for (let i = 0; i < this.biodomeCropClusters.length; i++) {
      this.biodomeCropClusters[i].scale.set(1, cropScale, 1);
    }
    this.biodomeUvLight.intensity = 3.2 + (this.biodomePulseBoost > 0 ? 1.8 : 0) + Math.sin(this.totalTime * 2.2) * 0.25;

    for (let i = 0; i < this.biodomeMistPositions.length; i += 3) {
      this.biodomeMistPositions[i + 1] += dt * (this.biodomePulseBoost > 0 ? 0.75 : 0.22);
      if (this.biodomeMistPositions[i + 1] > 2.75) {
        this.biodomeMistPositions[i + 1] = 0.3;
      }
    }
    this.biodomeMistPoints.geometry.attributes.position.needsUpdate = true;

    if (this.promptHighlightRing.visible) {
      const p = 1 + Math.sin(this.totalTime * 5.2) * 0.07;
      this.promptHighlightRing.scale.set(p, 1, p);
    }

    void this.isSoftwareGl;
  }

  public dispose(): void {
    for (const item of this.disposables) {
      item.dispose();
    }
    this.disposables = [];
  }
}
