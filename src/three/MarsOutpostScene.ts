import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import { RoomEnvironment } from 'three/examples/jsm/environments/RoomEnvironment.js';
import { ModuleId, SolHazardType } from '../data/campaignArc';
import {
  computeTauFogDensity,
  MissionResources,
  SolAtmosphereTargets,
} from '../game/simulation';
import {
  createContactShadowTexture,
  createDustOverlayTexture,
  createDustParticleSpriteTexture,
  createHabitatHullTexture,
  createMarsRegolithTextures,
  createRoverTrackTexture,
  createScorchDecalTexture,
  createSolarCellTexture,
} from './marsTextures';

import { DonkiSeverityLevel } from '../utils/nasaApi';

export type EmergencyMode = 'none' | 'dust' | 'flare';
export type ActiveSystemGlow = Record<ModuleId, number>;

export interface DonkiSceneTelemetry {
  level: DonkiSeverityLevel;
  intensity: number;
  cmeActive: boolean;
  shieldActive: boolean;
}

export interface AtmosphereTimelineInput {
  atmosphere: SolAtmosphereTargets;
  hazardType: SolHazardType;
  activeSystemGlow: ActiveSystemGlow;
  speedMultiplier: number;
  donki?: DonkiSceneTelemetry;
}

export interface ProjectedModuleAnchor {
  id: ModuleId;
  x: number;
  y: number;
  visible: boolean;
}

export interface MarsSceneCallbacks {
  onModuleClick: (moduleId: ModuleId) => void;
  onHoverModule: (moduleId: ModuleId | null) => void;
  onAnchorsUpdate: (anchors: ProjectedModuleAnchor[]) => void;
  onWebglError?: (reason: string) => void;
}

interface ModulePulse {
  moduleId: ModuleId;
  startTimeMs: number;
  duration: number;
}

interface NightCycleAnim {
  active: boolean;
  startTimeMs: number;
  duration: number;
  hazardType: SolHazardType;
  shieldBlockedFlare: boolean;
  completed: boolean;
  onComplete: () => void;
}

const MARTIAN_HORIZON_HEX = 0xd97b54;

function detectSoftwareWebGL(): boolean {
  try {
    const probe = document.createElement('canvas');
    const gl = probe.getContext('webgl2') || probe.getContext('webgl');
    if (!gl) return false;
    const dbg = gl.getExtension('WEBGL_debug_renderer_info');
    const renderer = dbg ? String(gl.getParameter(dbg.UNMASKED_RENDERER_WEBGL)) : '';
    gl.getExtension('WEBGL_lose_context')?.loseContext();
    return /SwiftShader|llvmpipe|softpipe|Software|Basic Render/i.test(renderer);
  } catch {
    return false;
  }
}

export function groundHeightAt(x: number, z: number): number {
  const r = Math.hypot(x, z);
  const angle = Math.atan2(z, x);
  const dunes =
    Math.sin(x * 0.21) * Math.cos(z * 0.17) * 0.16 +
    Math.sin((x - z) * 0.12) * 0.09;
  // Flat central plateau for base foundations, gently rolling crater basin out toward distant rim
  const plateauWeight = THREE.MathUtils.smoothstep(r, 12.0, 28.0);
  const gentleSlope =
    plateauWeight *
    (0.32 + Math.min(2.1, Math.max(0, r - 18) * 0.022) * (0.85 + 0.25 * Math.sin(angle * 4.0 + 0.6)));
  const detail = plateauWeight * dunes + (1 - plateauWeight) * (dunes * 0.1);
  return gentleSlope + detail;
}

function createAngularRockGeometry(seed: number, radius: number, detail = 1): THREE.DodecahedronGeometry {
  const geo = new THREE.DodecahedronGeometry(radius, detail);
  const pos = geo.attributes.position;
  const v = new THREE.Vector3();

  for (let i = 0; i < pos.count; i++) {
    v.fromBufferAttribute(pos, i);
    const dirHash =
      Math.sin(v.x * 12.9898 + v.y * 78.233 + v.z * 37.719 + seed * 19.3) * 43758.5453;
    const n = dirHash - Math.floor(dirHash);
    // Quantize slightly for flat sedimentary/volcanic fracture planes
    const facetScale = 0.76 + Math.floor(n * 5) * 0.11;
    v.multiplyScalar(facetScale);
    if (v.y > 0) v.y *= 0.82;
    pos.setXYZ(i, v.x, v.y, v.z);
  }
  geo.computeVertexNormals();
  return geo;
}

export class MarsOutpostScene {
  private container: HTMLElement;
  private callbacks: MarsSceneCallbacks;
  private isSoftwareGl: boolean;

  private renderer!: THREE.WebGLRenderer;
  private scene!: THREE.Scene;
  private camera!: THREE.PerspectiveCamera;
  private controls!: OrbitControls;
  private pmremGenerator: THREE.PMREMGenerator | null = null;
  private envRenderTarget: THREE.WebGLRenderTarget | null = null;

  private sunLight!: THREE.DirectionalLight;
  private rimLight!: THREE.DirectionalLight;
  private hemiLight!: THREE.HemisphereLight;
  private shelterInteriorLight!: THREE.PointLight;
  private moxieLight!: THREE.PointLight;
  private biodomeLight!: THREE.PointLight;

  private skyDomeMat: THREE.ShaderMaterial | null = null;
  private skyColor = new THREE.Color(MARTIAN_HORIZON_HEX);
  private daySkyColor = new THREE.Color(MARTIAN_HORIZON_HEX);
  private dustAlertSkyColor = new THREE.Color('#A64B28');
  private flareAlertSkyColor = new THREE.Color('#661C33');
  private mClassHorizonColor = new THREE.Color('#8E476A');
  private mClassZenithColor = new THREE.Color('#2B1742');
  private xClassHorizonColor = new THREE.Color('#6E192B');
  private xClassZenithColor = new THREE.Color('#190612');
  private sunsetSkyColor = new THREE.Color('#8C3828');
  private nightSkyColor = new THREE.Color('#090F22');
  private flareSkyColor = new THREE.Color('#4C1D3D');

  private calmSunColor = new THREE.Color('#FFF0D4');
  private alertSunColor = new THREE.Color('#FF8C61');
  private xClassSunColor = new THREE.Color('#FF384C');
  private calmHemiSky = new THREE.Color('#F3B496');
  private calmHemiGround = new THREE.Color('#3A1912');
  private xClassHemiSky = new THREE.Color('#DC2626');
  private xClassHemiGround = new THREE.Color('#4A0B14');
  private calmWindowColor = new THREE.Color('#FEF08A');
  private calmWindowEmissive = new THREE.Color('#F59E0B');
  private calmHullColor = new THREE.Color('#E2E8F0');
  private alertHullTint = new THREE.Color('#FCA5A5');
  private alertRedColor = new THREE.Color('#FF2A2A');
  private alertAmberColor = new THREE.Color('#FF7A00');
  private calmMoxieColor = new THREE.Color('#22D3EE');
  private calmMoxieEmissive = new THREE.Color('#06B6D4');
  private calmBioColor = new THREE.Color('#A7F3D0');
  private calmBioEmissive = new THREE.Color('#10B981');
  private calmLedColor = new THREE.Color('#E879F9');
  private calmLedEmissive = new THREE.Color('#D946EF');
  private calmBeaconColor = new THREE.Color('#38BDF8');
  private calmBeaconEmissive = new THREE.Color('#0EA5E9');
  private blackColor = new THREE.Color('#000000');
  private scratchAlertColor = new THREE.Color();
  private scratchTargetZenith = new THREE.Color('#B55A48');
  private zenithColor = new THREE.Color('#B55A48');
  private zenithDayColor = new THREE.Color('#B55A48');
  private zenithNightColor = new THREE.Color('#050814');

  private fog!: THREE.FogExp2;
  private starsPoints!: THREE.Points;
  private starsMat!: THREE.PointsMaterial;
  private auroraGroup!: THREE.Group;
  private auroraMats: THREE.ShaderMaterial[] = [];

  // Dynamic module references
  private solarTiltGroups: THREE.Group[] = [];
  private solarCellMat!: THREE.MeshPhysicalMaterial;
  private solarBatteryMat!: THREE.MeshStandardMaterial;
  private solarDustMat!: THREE.MeshStandardMaterial;
  private habitatDustMat!: THREE.MeshStandardMaterial;
  private solarWipers: Array<{ mesh: THREE.Mesh; baseX: number }> = [];

  private moxieFan!: THREE.Group;
  private moxieTankMat!: THREE.MeshStandardMaterial;
  private moxieBubbles: THREE.Mesh[] = [];

  private waterShieldMesh!: THREE.Mesh;
  private waterShieldMat!: THREE.MeshPhysicalMaterial;
  private waterShieldHexMesh!: THREE.Mesh;
  private waterShieldHexMat!: THREE.ShaderMaterial;
  private shelterHullMat!: THREE.MeshStandardMaterial;
  private shelterWindowMat!: THREE.MeshStandardMaterial;
  private shelterBeaconMat!: THREE.MeshStandardMaterial;
  private commsBeaconMat!: THREE.MeshStandardMaterial;

  private biodomeCrops: THREE.Group[] = [];
  private bioLeafMat!: THREE.MeshStandardMaterial;
  private biodomeGlassMat!: THREE.MeshPhysicalMaterial;
  private bioLedRingMat!: THREE.MeshStandardMaterial;

  private targetRing!: THREE.Mesh;
  private hoverRing!: THREE.Mesh;

  // Particle effects
  private dustStormPoints!: THREE.Points;
  private dustStormMat!: THREE.PointsMaterial;
  private dustPositions!: Float32Array;

  private flareBoltsGroup!: THREE.Group;
  private flareBolts: Array<{
    mesh: THREE.Mesh;
    mat: THREE.MeshBasicMaterial;
    vel: THREE.Vector3;
    speed: number;
    angle: number;
    radius: number;
    deflected: boolean;
    skyStreak: boolean;
  }> = [];

  private anchorWorldMap: Record<ModuleId, THREE.Vector3> = {
    solar: new THREE.Vector3(-7.2, 3.05, 2.2),
    moxie: new THREE.Vector3(-4.0, 3.65, -4.4),
    shelter: new THREE.Vector3(0.9, 4.45, 1.0),
    biodome: new THREE.Vector3(7.1, 3.35, 2.0),
  };

  private moduleWorldCenters: Record<ModuleId, THREE.Vector3> = {
    solar: new THREE.Vector3(-6.8, 0.2, 2.2),
    moxie: new THREE.Vector3(-3.7, 0.2, -4.4),
    shelter: new THREE.Vector3(0.9, 0.2, 1.0),
    biodome: new THREE.Vector3(6.8, 0.2, 2.0),
  };

  private hitColliders: THREE.Mesh[] = [];
  private raycaster = new THREE.Raycaster();
  private pointerNdc = new THREE.Vector2();
  private projScratch = new THREE.Vector3();
  private lastAnchors: ProjectedModuleAnchor[] = [];

  private pointerDownPos = { x: 0, y: 0 };
  private hoveredModule: ModuleId | null = null;
  private highlightedTarget: ModuleId | 'any' = 'moxie';

  private resources: MissionResources = {
    powerKwh: 88,
    lifeSupportPct: 80,
    crewHealthPct: 88,
    panelDustPct: 16,
    waterWallCm: 16,
    cropGrowth: 2,
    sciencePoints: 0,
  };
  private currentTau = 0.38;
  private emergencyMode: EmergencyMode = 'none';
  private emergencyBlend = 0;

  private targetAtmosphere: SolAtmosphereTargets = {
    solProgress: 0,
    crisisIntensity: 0.12,
    effectiveTau: 0.38,
    fogDensity: 0.015,
    sunElevation: 0.98,
    nightFactor: 0,
  };
  private smoothedSolProgress = 0;
  private smoothedCrisisIntensity = 0.12;
  private smoothedEffectiveTau = 0.38;
  private smoothedSunElevation = 0.98;
  private smoothedNightFactor = 0;
  private targetHazardType: SolHazardType = 'none';
  private simSpeedMultiplier = 1;
  private activeSystemGlow: ActiveSystemGlow = {
    solar: 0.35,
    moxie: 0.35,
    shelter: 0.25,
    biodome: 0.3,
  };
  private smoothedSystemGlow: ActiveSystemGlow = {
    solar: 0.35,
    moxie: 0.35,
    shelter: 0.25,
    biodome: 0.3,
  };
  private moduleBoostTimers: Record<ModuleId, number> = {
    solar: 0,
    moxie: 0,
    shelter: 0,
    biodome: 0,
  };
  private targetSunPos = new THREE.Vector3(22, 19, 14);
  private targetSkyColor = new THREE.Color(MARTIAN_HORIZON_HEX);
  private targetDayBaseColor = new THREE.Color(MARTIAN_HORIZON_HEX);
  private targetSunColor = new THREE.Color('#FFF0D4');

  private donkiTelemetry: DonkiSceneTelemetry = {
    level: 'quiet',
    intensity: 0.15,
    cmeActive: false,
    shieldActive: false,
  };
  private smoothedDonkiM = 0;
  private smoothedDonkiX = 0;
  private smoothedShieldHex = 0;
  private shieldBoostUntilMs = 0;
  private shieldImpactPulse = 0;
  private streakNormScratch = new THREE.Vector3();
  private streakUpAxis = new THREE.Vector3(0, 1, 0);

  private activePulse: ModulePulse | null = null;
  private nightAnim: NightCycleAnim = {
    active: false,
    startTimeMs: 0,
    duration: 3.0,
    hazardType: 'none',
    shieldBlockedFlare: false,
    completed: false,
    onComplete: () => {},
  };

  private disposables: Array<{ dispose: () => void }> = [];
  private rafId: number | null = null;
  private swTimerId: number | null = null;
  private destroyed = false;
  private lastFrameTime = performance.now();
  private totalTime = 0;

  constructor(container: HTMLElement, callbacks: MarsSceneCallbacks) {
    this.container = container;
    this.callbacks = callbacks;
    this.isSoftwareGl = detectSoftwareWebGL();
    this.raycaster.layers.set(1);

    this.initScene();
    this.bindEvents();
    this.startLoop();
  }

  private track<T extends { dispose: () => void }>(item: T): T {
    this.disposables.push(item);
    return item;
  }

  private initScene(): void {
    const width = Math.max(this.container.clientWidth || window.innerWidth, 320);
    const height = Math.max(this.container.clientHeight || window.innerHeight, 320);

    try {
      this.renderer = new THREE.WebGLRenderer({
        antialias: !this.isSoftwareGl,
        powerPreference: 'high-performance',
        alpha: false,
      });
    } catch (err) {
      this.callbacks.onWebglError?.(
        err instanceof Error ? err.message : 'Unable to initialize 3D WebGL viewport.'
      );
      return;
    }

    this.renderer.setPixelRatio(this.isSoftwareGl ? 1 : Math.min(window.devicePixelRatio || 1, 2));
    this.renderer.setSize(width, height);
    this.renderer.outputColorSpace = THREE.SRGBColorSpace;
    this.renderer.toneMapping = THREE.ACESFilmicToneMapping;
    this.renderer.toneMappingExposure = 1.06;
    if (!this.isSoftwareGl) {
      this.renderer.shadowMap.enabled = true;
      this.renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    } else {
      this.renderer.shadowMap.enabled = false;
    }

    this.renderer.domElement.style.display = 'block';
    this.renderer.domElement.style.width = '100%';
    this.renderer.domElement.style.height = '100%';
    this.container.appendChild(this.renderer.domElement);

    this.scene = new THREE.Scene();
    // Exponential Martian butterscotch-pink haze (#d97b54) synced to the horizon & Tau weather system
    const dustColor = new THREE.Color(MARTIAN_HORIZON_HEX);
    this.daySkyColor.copy(dustColor);
    this.skyColor.copy(dustColor);
    this.fog = new THREE.FogExp2(dustColor, 0.015);
    this.scene.fog = this.fog;
    this.scene.background = this.skyColor.clone();
    this.renderer.setClearColor(this.fog.color);

    // Studio PBR environment reflections (kept identical on GPU & software WebGL)
    this.pmremGenerator = new THREE.PMREMGenerator(this.renderer);
    const roomEnv = new RoomEnvironment();
    this.envRenderTarget = this.pmremGenerator.fromScene(roomEnv, 0.04);
    this.scene.environment = this.envRenderTarget.texture;
    this.scene.environmentIntensity = 0.85;
    roomEnv.dispose();

    // Extended far plane (360) reveals the layered Jezero Crater rim on the distant horizon
    this.camera = new THREE.PerspectiveCamera(46, width / height, 0.25, 360);
    this.applyResponsiveCamera(width, height, true);

    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.target.set(0, 2.85, -5.5);
    this.controls.enableDamping = !this.isSoftwareGl;
    this.controls.dampingFactor = 0.06;
    this.controls.enablePan = false;
    this.controls.minDistance = 12;
    this.controls.maxDistance = 36;
    this.controls.minPolarAngle = Math.PI * 0.18;
    this.controls.maxPolarAngle = Math.PI * 0.49;
    this.controls.update();

    this.buildLights();
    this.buildSkyDomeAndStars();
    this.buildMarsTerrain();
    this.buildCraterRimMountains();
    this.buildMarsRocks();
    this.buildOutpostModules();
    this.buildStorytellingProps();
    this.buildHazardsAndRings();
    this.syncVisualState();
    this.renderer.render(this.scene, this.camera);
    this.emitProjectedAnchors();
  }

  private buildLights(): void {
    // Warm butterscotch sky fill paired with deep rust-umber shadow bounce for high Martian contrast
    this.hemiLight = new THREE.HemisphereLight('#F3B496', '#3A1912', 0.62);
    this.scene.add(this.hemiLight);

    // High-contrast Martian directional sun with 2048px crisp shadow map
    this.sunLight = new THREE.DirectionalLight('#FFF0D4', 2.85);
    this.sunLight.position.set(22, 19, 14);
    this.sunLight.shadow.mapSize.set(2048, 2048);
    this.sunLight.shadow.camera.near = 4;
    this.sunLight.shadow.camera.far = 76;
    const d = 22;
    this.sunLight.shadow.camera.left = -d;
    this.sunLight.shadow.camera.right = d;
    this.sunLight.shadow.camera.top = d;
    this.sunLight.shadow.camera.bottom = -d;
    this.sunLight.shadow.bias = -0.0004;
    this.sunLight.shadow.normalBias = 0.025;
    this.sunLight.shadow.radius = 2.0;
    if (!this.isSoftwareGl) {
      this.sunLight.castShadow = true;
    }
    this.scene.add(this.sunLight);

    // Pulsing crimson rim-light for X-Class flare / CME red-alert state
    this.rimLight = new THREE.DirectionalLight('#FF1F4B', 0);
    this.rimLight.position.set(-18, 11, -18);
    this.scene.add(this.rimLight);

    this.shelterInteriorLight = new THREE.PointLight('#FBBF24', 1.1, 16, 1.4);
    this.shelterInteriorLight.position.set(0.9, 2.2, 1.4);
    this.scene.add(this.shelterInteriorLight);

    this.moxieLight = new THREE.PointLight('#22D3EE', 1.15, 14, 1.4);
    this.moxieLight.position.set(-3.7, 2.2, -3.6);
    this.scene.add(this.moxieLight);

    // Warm emerald point-light glow inside the Hydroponic Bio-Dome
    this.biodomeLight = new THREE.PointLight('#10B981', 2.65, 18, 1.25);
    this.biodomeLight.position.set(6.8, 1.85, 2.0);
    this.scene.add(this.biodomeLight);
  }

  private buildSkyDomeAndStars(): void {
    // Butterscotch-pink horizon gradient dome unaffected by fog so distant rim silhouettes stand out
    const skyGeo = this.track(new THREE.SphereGeometry(240, 32, 20));
    this.skyDomeMat = this.track(
      new THREE.ShaderMaterial({
        side: THREE.BackSide,
        depthWrite: false,
        fog: false,
        uniforms: {
          uHorizonColor: { value: new THREE.Color(MARTIAN_HORIZON_HEX) },
          uZenithColor: { value: new THREE.Color('#B55A48') },
        },
        vertexShader: `
          varying vec3 vWorldDir;
          void main() {
            vec4 wp = modelMatrix * vec4(position, 1.0);
            vWorldDir = normalize(position);
            gl_Position = projectionMatrix * viewMatrix * wp;
          }
        `,
        fragmentShader: `
          uniform vec3 uHorizonColor;
          uniform vec3 uZenithColor;
          varying vec3 vWorldDir;
          void main() {
            float h = clamp(vWorldDir.y, 0.0, 1.0);
            float curve = pow(h, 0.55);
            vec3 col = mix(uHorizonColor, uZenithColor, curve);
            gl_FragColor = vec4(col, 1.0);
          }
        `,
      })
    );
    const skyDome = new THREE.Mesh(skyGeo, this.skyDomeMat);
    this.scene.add(skyDome);

    const count = 260;
    const positions = new Float32Array(count * 3);
    let seed = 109;
    const rand = () => {
      seed = (seed * 1664525 + 1013904223) % 4294967296;
      return seed / 4294967296;
    };
    for (let i = 0; i < count; i++) {
      const theta = rand() * Math.PI * 2;
      const phi = rand() * Math.PI * 0.44;
      const r = 180;
      positions[i * 3] = r * Math.sin(phi) * Math.cos(theta);
      positions[i * 3 + 1] = Math.max(18, r * Math.cos(phi));
      positions[i * 3 + 2] = r * Math.sin(phi) * Math.sin(theta);
    }
    const geo = this.track(new THREE.BufferGeometry());
    geo.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    this.starsMat = this.track(
      new THREE.PointsMaterial({
        color: '#E2E8F0',
        size: 0.95,
        transparent: true,
        opacity: 0.0,
        depthWrite: false,
        fog: false,
      })
    );
    this.starsPoints = new THREE.Points(geo, this.starsMat);
    this.scene.add(this.starsPoints);

    // Upper-sky auroral ribbons for M-Class / X-Class DONKI solar storms
    this.auroraGroup = new THREE.Group();
    this.auroraGroup.visible = false;
    const ribbonConfigs = [
      { z: -56, y: 14.5, x: -6, width: 78, height: 10.5, rotY: 0.14, phase: 0.0 },
      { z: -70, y: 17.8, x: 8, width: 92, height: 12.0, rotY: -0.18, phase: 2.1 },
      { z: -84, y: 16.0, x: -2, width: 102, height: 11.5, rotY: 0.05, phase: 4.3 },
    ];
    ribbonConfigs.forEach((cfg) => {
      const rGeo = this.track(new THREE.PlaneGeometry(cfg.width, cfg.height, 40, 4));
      const posAttr = rGeo.attributes.position;
      for (let i = 0; i < posAttr.count; i++) {
        const px = posAttr.getX(i);
        const arcZ = Math.sin((px / cfg.width) * Math.PI * 1.4 + cfg.phase) * 7.5;
        posAttr.setZ(i, arcZ);
      }
      rGeo.computeVertexNormals();
      const rMat = this.track(
        new THREE.ShaderMaterial({
          transparent: true,
          depthWrite: false,
          fog: false,
          blending: THREE.AdditiveBlending,
          side: THREE.DoubleSide,
          uniforms: {
            uTime: { value: 0 },
            uOpacity: { value: 0 },
            uPhase: { value: cfg.phase },
            uCrimsonMix: { value: 0 },
          },
          vertexShader: `
            uniform float uTime;
            uniform float uPhase;
            varying vec2 vUv;
            void main() {
              vUv = uv;
              vec3 p = position;
              p.z += sin(p.x * 0.09 + uTime * 0.9 + uPhase) * 2.2;
              p.y += cos(p.x * 0.06 - uTime * 0.65 + uPhase) * 1.1;
              gl_Position = projectionMatrix * modelViewMatrix * vec4(p, 1.0);
            }
          `,
          fragmentShader: `
            uniform float uTime;
            uniform float uOpacity;
            uniform float uPhase;
            uniform float uCrimsonMix;
            varying vec2 vUv;
            void main() {
              float edgeX = smoothstep(0.0, 0.22, vUv.x) * (1.0 - smoothstep(0.78, 1.0, vUv.x));
              float edgeY = pow(smoothstep(0.0, 0.35, vUv.y), 1.4) * (1.0 - smoothstep(0.35, 1.0, vUv.y));
              float curtains = 0.55 + 0.45 * sin(vUv.x * 26.0 + uTime * 1.4 + uPhase + sin(vUv.x * 8.0) * 1.8);
              float fineRays = 0.7 + 0.3 * sin(vUv.x * 64.0 - uTime * 2.2 + uPhase);
              vec3 mColA = vec3(0.28, 0.96, 0.84);
              vec3 mColB = vec3(0.76, 0.45, 1.0);
              vec3 xColA = vec3(1.0, 0.28, 0.46);
              vec3 xColB = vec3(0.9, 0.42, 1.0);
              vec3 baseCol = mix(mix(mColA, mColB, vUv.y), mix(xColA, xColB, vUv.y), uCrimsonMix);
              float alpha = clamp(edgeX * edgeY * curtains * fineRays * uOpacity, 0.0, 0.92);
              gl_FragColor = vec4(baseCol, alpha);
            }
          `,
        })
      );
      this.auroraMats.push(rMat);
      const rMesh = new THREE.Mesh(rGeo, rMat);
      rMesh.position.set(cfg.x, cfg.y, cfg.z);
      rMesh.rotation.y = cfg.rotY;
      this.auroraGroup.add(rMesh);
    });
    this.scene.add(this.auroraGroup);
  }

  private buildMarsTerrain(): void {
    const maxAniso = this.renderer.capabilities.getMaxAnisotropy();
    const { map, normalMap, bumpMap } = createMarsRegolithTextures(maxAniso);
    this.track(map);
    this.track(normalMap);
    this.track(bumpMap);

    const terrainGeo = this.track(new THREE.PlaneGeometry(240, 240, 144, 144));
    terrainGeo.rotateX(-Math.PI / 2);
    const pos = terrainGeo.attributes.position;
    for (let i = 0; i < pos.count; i++) {
      const x = pos.getX(i);
      const z = pos.getZ(i);
      pos.setY(i, groundHeightAt(x, z));
    }
    terrainGeo.computeVertexNormals();

    // High-res procedural Martian regolith PBR shader with world-space multi-octave FBM variation
    const terrainMat = this.track(
      new THREE.MeshStandardMaterial({
        map,
        normalMap,
        normalScale: new THREE.Vector2(1.15, 1.15),
        bumpMap,
        bumpScale: 0.14,
        roughness: 0.89,
        metalness: 0.04,
      })
    );

    terrainMat.onBeforeCompile = (shader) => {
      shader.vertexShader = shader.vertexShader
        .replace(
          '#include <common>',
          `#include <common>
varying vec3 vTerrainWorldPos;`
        )
        .replace(
          '#include <begin_vertex>',
          `#include <begin_vertex>
  vTerrainWorldPos = (modelMatrix * vec4(transformed, 1.0)).xyz;`
        );

      shader.fragmentShader = shader.fragmentShader
        .replace(
          '#include <common>',
          `#include <common>
varying vec3 vTerrainWorldPos;

float marsHash21(vec2 p) {
  p = fract(p * vec2(123.34, 456.21));
  p += dot(p, p + 45.32);
  return fract(p.x * p.y);
}

float marsValueNoise(vec2 p) {
  vec2 i = floor(p);
  vec2 f = fract(p);
  vec2 u = f * f * (3.0 - 2.0 * f);
  float a = marsHash21(i);
  float b = marsHash21(i + vec2(1.0, 0.0));
  float c = marsHash21(i + vec2(0.0, 1.0));
  float d = marsHash21(i + vec2(1.0, 1.0));
  return mix(mix(a, b, u.x), mix(c, d, u.x), u.y);
}

float marsFbm(vec2 p) {
  float v = 0.0;
  float a = 0.5;
  mat2 rot = mat2(0.8, 0.6, -0.6, 0.8);
  for (int i = 0; i < 4; i++) {
    v += a * marsValueNoise(p);
    p = rot * p * 2.05;
    a *= 0.5;
  }
  return v;
}`
        )
        .replace(
          '#include <color_fragment>',
          `#include <color_fragment>
  vec2 wp = vTerrainWorldPos.xz;
  float macroFbm = marsFbm(wp * 0.055);
  float midFbm = marsFbm(wp * 0.22 + vec2(4.1, -2.7));
  float windRipple = sin(wp.x * 1.9 + wp.y * 0.75 + macroFbm * 4.5) * 0.5 + 0.5;

  // Rust-orange, sunlit ochre, and dark basalt strata (linear space)
  vec3 ochreCol = vec3(0.56, 0.22, 0.09);
  vec3 rustCol = vec3(0.36, 0.11, 0.048);
  vec3 basaltCol = vec3(0.11, 0.048, 0.032);

  vec3 fbmTint = mix(rustCol, ochreCol, smoothstep(0.28, 0.78, macroFbm * 0.7 + windRipple * 0.3));
  fbmTint = mix(fbmTint, basaltCol, smoothstep(0.62, 0.88, midFbm) * 0.55);

  // Subtle dusty compaction around the outpost plateau
  float baseZone = 1.0 - smoothstep(5.0, 15.0, length(wp));
  fbmTint = mix(fbmTint, vec3(0.42, 0.16, 0.075), baseZone * 0.22);

  diffuseColor.rgb = mix(diffuseColor.rgb, diffuseColor.rgb * (fbmTint * 2.45), 0.62);`
        );
    };

    const terrain = new THREE.Mesh(terrainGeo, terrainMat);
    terrain.receiveShadow = !this.isSoftwareGl;
    this.scene.add(terrain);
  }

  private buildCraterRimMountains(): void {
    // Distant, layered Jezero Crater mountain rim silhouette rings catching the butterscotch-pink haze
    const buildRimRing = (
      innerR: number,
      outerR: number,
      minH: number,
      maxH: number,
      colorHex: string,
      seedOffset: number
    ) => {
      const segs = 96;
      const geo = this.track(new THREE.RingGeometry(innerR, outerR, segs, 6));
      geo.rotateX(-Math.PI / 2);
      const pos = geo.attributes.position;
      for (let i = 0; i < pos.count; i++) {
        const x = pos.getX(i);
        const z = pos.getZ(i);
        const r = Math.hypot(x, z);
        const angle = Math.atan2(z, x);
        const t = THREE.MathUtils.clamp((r - innerR) / Math.max(1, outerR - innerR), 0, 1);
        const ridgeProfile = Math.sin(t * Math.PI);
        const peaks =
          0.55 +
          0.25 * Math.sin(angle * 7 + seedOffset) +
          0.16 * Math.cos(angle * 13 - seedOffset * 1.7) +
          0.1 * Math.sin(angle * 23 + seedOffset * 0.5);
        const baseY = groundHeightAt(x * 0.45, z * 0.45) * 0.4;
        const peakY = baseY + ridgeProfile * THREE.MathUtils.lerp(minH, maxH, peaks) - 0.5;
        pos.setY(i, peakY);
      }
      geo.computeVertexNormals();
      const mat = this.track(
        new THREE.MeshStandardMaterial({
          color: colorHex,
          roughness: 0.96,
          metalness: 0.02,
          side: THREE.DoubleSide,
        })
      );
      const mesh = new THREE.Mesh(geo, mat);
      mesh.receiveShadow = false;
      this.scene.add(mesh);
    };

    buildRimRing(56, 86, 4.5, 10.5, '#8F4026', 1.2);
    buildRimRing(82, 128, 9.0, 19.0, '#A85237', 3.7);
  }

  private buildMarsRocks(): void {
    const dummy = new THREE.Object3D();
    const colorScratch = new THREE.Color();
    let seed = 77;
    const rand = () => {
      seed = (seed * 1664525 + 1013904223) % 4294967296;
      return seed / 4294967296;
    };

    const rockPalette = ['#5E291D', '#7A3825', '#42231E', '#94482E', '#6D3022'];

    const isClearOfOutpost = (x: number, z: number): boolean => {
      // Keep central pads, rover, and main connecting tracks clear of big boulders
      const r = Math.hypot(x, z);
      if (r < 11.4) return false;
      const dSolar = Math.hypot(x + 6.8, z - 2.2);
      const dBio = Math.hypot(x - 6.8, z - 2.0);
      const dMoxie = Math.hypot(x + 3.7, z + 4.4);
      const dRover = Math.hypot(x - 3.6, z - 4.4);
      return dSolar > 4.6 && dBio > 3.8 && dMoxie > 3.5 && dRover > 2.2;
    };

    // 1. Half-Buried Angular Boulders (~44)
    const boulderGeo = this.track(createAngularRockGeometry(17, 0.85, 1));
    const boulderMat = this.track(
      new THREE.MeshStandardMaterial({
        color: '#FFFFFF',
        roughness: 0.88,
        metalness: 0.06,
      })
    );
    const boulderCoords: Array<{ x: number; z: number }> = [];
    const boulderCount = 44;
    const boulders = new THREE.InstancedMesh(boulderGeo, boulderMat, boulderCount);
    boulders.castShadow = !this.isSoftwareGl;
    boulders.receiveShadow = !this.isSoftwareGl;

    let placed = 0;
    let attempts = 0;
    while (placed < boulderCount && attempts < 400) {
      attempts++;
      const angle = (placed / boulderCount) * Math.PI * 2 + (rand() - 0.5) * 0.45;
      const dist = 12.2 + rand() * 28.0;
      const x = Math.cos(angle) * dist;
      const z = Math.sin(angle) * dist;
      if (!isClearOfOutpost(x, z)) continue;

      const scale = 0.55 + rand() * 1.35;
      // Sunk ~35% into regolith so boulders appear half-buried
      const y = groundHeightAt(x, z) + scale * 0.14;
      dummy.position.set(x, y, z);
      dummy.rotation.set(rand() * 1.4, rand() * Math.PI * 2, rand() * 1.4);
      dummy.scale.set(
        scale * (0.85 + rand() * 0.55),
        scale * (0.58 + rand() * 0.35),
        scale * (0.8 + rand() * 0.5)
      );
      dummy.updateMatrix();
      boulders.setMatrixAt(placed, dummy.matrix);
      colorScratch.set(rockPalette[Math.floor(rand() * rockPalette.length)]);
      boulders.setColorAt(placed, colorScratch);
      boulderCoords.push({ x, z });
      placed++;
    }
    boulders.count = placed;
    boulders.instanceMatrix.needsUpdate = true;
    if (boulders.instanceColor) boulders.instanceColor.needsUpdate = true;
    boulders.computeBoundingSphere();
    this.scene.add(boulders);

    // 2. Clustered Gravel & Angular Stones (~160) clustered around boulders & crater edges
    const gravelGeo = this.track(createAngularRockGeometry(53, 0.28, 0));
    const gravelMat = this.track(
      new THREE.MeshStandardMaterial({
        color: '#FFFFFF',
        roughness: 0.92,
        metalness: 0.04,
      })
    );
    const gravelCount = 160;
    const gravel = new THREE.InstancedMesh(gravelGeo, gravelMat, gravelCount);
    gravel.castShadow = !this.isSoftwareGl;
    gravel.receiveShadow = !this.isSoftwareGl;

    for (let i = 0; i < gravelCount; i++) {
      const parent = boulderCoords[i % Math.max(1, boulderCoords.length)] ?? { x: 14, z: 10 };
      const a = rand() * Math.PI * 2;
      const offsetR = 0.7 + rand() * 2.8;
      const x = parent.x + Math.cos(a) * offsetR;
      const z = parent.z + Math.sin(a) * offsetR;
      const s = 0.45 + rand() * 0.95;
      const y = groundHeightAt(x, z) + s * 0.04;

      dummy.position.set(x, y, z);
      dummy.rotation.set(rand() * 3, rand() * 3, rand() * 3);
      dummy.scale.set(s * (0.8 + rand() * 0.5), s * 0.65, s * (0.8 + rand() * 0.5));
      dummy.updateMatrix();
      gravel.setMatrixAt(i, dummy.matrix);
      colorScratch.set(rockPalette[Math.floor(rand() * rockPalette.length)]);
      gravel.setColorAt(i, colorScratch);
    }
    gravel.instanceMatrix.needsUpdate = true;
    if (gravel.instanceColor) gravel.instanceColor.needsUpdate = true;
    gravel.computeBoundingSphere();
    this.scene.add(gravel);

    // 3. Pebble Patches (~240) scattered across regolith flats & module edges
    const pebbleGeo = this.track(createAngularRockGeometry(91, 0.11, 0));
    const pebbleMat = this.track(
      new THREE.MeshStandardMaterial({
        color: '#FFFFFF',
        roughness: 0.94,
        metalness: 0.03,
      })
    );
    const pebbleCount = 240;
    const pebbles = new THREE.InstancedMesh(pebbleGeo, pebbleMat, pebbleCount);
    pebbles.castShadow = false;
    pebbles.receiveShadow = !this.isSoftwareGl;

    for (let i = 0; i < pebbleCount; i++) {
      // Cluster patches in windward drifts
      const patchIdx = Math.floor(i / 8);
      const patchAngle = (patchIdx * 2.39996) % (Math.PI * 2);
      const patchDist = 4.5 + ((patchIdx * 7.3) % 19.5);
      const cx = Math.cos(patchAngle) * patchDist;
      const cz = Math.sin(patchAngle) * patchDist;
      const x = cx + (rand() - 0.5) * 2.4;
      const z = cz + (rand() - 0.5) * 2.4;
      const s = 0.55 + rand() * 0.85;

      dummy.position.set(x, groundHeightAt(x, z) + 0.02, z);
      dummy.rotation.set(rand() * 2, rand() * 3, rand() * 2);
      dummy.scale.set(s * 1.1, s * 0.55, s);
      dummy.updateMatrix();
      pebbles.setMatrixAt(i, dummy.matrix);
      colorScratch.set(rockPalette[Math.floor(rand() * rockPalette.length)]);
      pebbles.setColorAt(i, colorScratch);
    }
    pebbles.instanceMatrix.needsUpdate = true;
    if (pebbles.instanceColor) pebbles.instanceColor.needsUpdate = true;
    pebbles.computeBoundingSphere();
    this.scene.add(pebbles);
  }

  private addHitCollider(moduleId: ModuleId, pos: THREE.Vector3, radius: number, height: number): void {
    const hitGeo = this.track(new THREE.CylinderGeometry(radius, radius * 1.08, height, 14));
    const hitMat = this.track(new THREE.MeshBasicMaterial({ visible: false }));
    const hit = new THREE.Mesh(hitGeo, hitMat);
    hit.position.set(pos.x, pos.y + height * 0.5, pos.z);
    hit.layers.set(1);
    hit.userData.moduleId = moduleId;
    this.hitColliders.push(hit);
    this.scene.add(hit);
  }

  private addGroundDecal(
    x: number,
    z: number,
    radius: number,
    tex: THREE.Texture,
    opacity = 0.72,
    yOffset = 0.025
  ): void {
    const geo = this.track(new THREE.PlaneGeometry(radius * 2, radius * 2));
    geo.rotateX(-Math.PI / 2);
    const mat = this.track(
      new THREE.MeshBasicMaterial({
        map: tex,
        transparent: true,
        opacity,
        depthWrite: false,
        polygonOffset: true,
        polygonOffsetFactor: -1,
        polygonOffsetUnits: -1,
      })
    );
    const mesh = new THREE.Mesh(geo, mat);
    mesh.position.set(x, groundHeightAt(x, z) + yOffset, z);
    this.scene.add(mesh);
  }

  private addDustFoundationSkirt(x: number, z: number, innerR: number, outerR: number): void {
    const y = groundHeightAt(x, z);
    const skirtGeo = this.track(new THREE.CylinderGeometry(innerR, outerR, 0.24, 24));
    const skirtMat = this.track(
      new THREE.MeshStandardMaterial({
        color: '#8D4329',
        roughness: 0.96,
        metalness: 0.02,
      })
    );
    const skirt = new THREE.Mesh(skirtGeo, skirtMat);
    skirt.position.set(x, y + 0.05, z);
    skirt.receiveShadow = !this.isSoftwareGl;
    this.scene.add(skirt);
  }

  private buildTerrainTrackRibbon(
    points: THREE.Vector3[],
    width: number,
    tex: THREE.CanvasTexture
  ): void {
    const curve = new THREE.CatmullRomCurve3(points);
    const segments = 26;
    const positions: number[] = [];
    const uvs: number[] = [];
    const indices: number[] = [];
    const up = new THREE.Vector3(0, 1, 0);
    const right = new THREE.Vector3();

    const totalLen = curve.getLength();
    for (let i = 0; i <= segments; i++) {
      const t = i / segments;
      const pt = curve.getPoint(t);
      const tan = curve.getTangent(t).setY(0).normalize();
      right.crossVectors(tan, up).normalize().multiplyScalar(width * 0.5);

      const lx = pt.x - right.x;
      const lz = pt.z - right.z;
      const rx = pt.x + right.x;
      const rz = pt.z + right.z;

      positions.push(lx, groundHeightAt(lx, lz) + 0.045, lz);
      positions.push(rx, groundHeightAt(rx, rz) + 0.045, rz);

      const vCoord = t * (totalLen / 1.8);
      uvs.push(0, vCoord, 1, vCoord);

      if (i < segments) {
        const base = i * 2;
        indices.push(base, base + 1, base + 2, base + 1, base + 3, base + 2);
      }
    }

    const geo = this.track(new THREE.BufferGeometry());
    geo.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
    geo.setAttribute('uv', new THREE.Float32BufferAttribute(uvs, 2));
    geo.setIndex(indices);
    geo.computeVertexNormals();

    const mat = this.track(
      new THREE.MeshBasicMaterial({
        map: tex,
        transparent: true,
        opacity: 0.88,
        depthWrite: false,
        side: THREE.DoubleSide,
        polygonOffset: true,
        polygonOffsetFactor: -3,
        polygonOffsetUnits: -3,
      })
    );
    const ribbon = new THREE.Mesh(geo, mat);
    ribbon.renderOrder = 2;
    this.scene.add(ribbon);
  }

  private buildOutpostModules(): void {
    const shadowTex = this.track(createContactShadowTexture());
    const scorchTex = this.track(createScorchDecalTexture());
    const maxAniso = this.renderer.capabilities.getMaxAnisotropy();
    const solarTex = this.track(createSolarCellTexture(maxAniso));
    const hullSeamTex = this.track(createHabitatHullTexture(maxAniso));
    const dustOverlayTex = this.track(createDustOverlayTexture(maxAniso));

    this.habitatDustMat = this.track(
      new THREE.MeshStandardMaterial({
        map: dustOverlayTex,
        color: '#B84B2B',
        roughness: 0.98,
        metalness: 0.0,
        envMapIntensity: 0.12,
        transparent: true,
        opacity: 0.06,
        depthWrite: false,
      })
    );

    // Shared PBR aerospace engineering materials
    const hullWhite = this.track(
      new THREE.MeshStandardMaterial({
        map: hullSeamTex,
        color: '#E2E8F0',
        roughness: 0.42,
        metalness: 0.42,
      })
    );
    const hullSlate = this.track(
      new THREE.MeshStandardMaterial({ color: '#334155', roughness: 0.44, metalness: 0.65 })
    );
    const darkMetal = this.track(
      new THREE.MeshStandardMaterial({ color: '#1E293B', roughness: 0.34, metalness: 0.82 })
    );
    const accentAmber = this.track(
      new THREE.MeshStandardMaterial({
        color: '#F59E0B',
        emissive: '#B45309',
        emissiveIntensity: 0.45,
        roughness: 0.35,
        metalness: 0.35,
      })
    );

    // ------------------------------------------------------------------
    // 1. WATER-WALL STORM SHELTER (Center-Right: 0.9, y, 1.0)
    // ------------------------------------------------------------------
    const shelterPos = new THREE.Vector3(0.9, groundHeightAt(0.9, 1.0), 1.0);
    const shelterGroup = new THREE.Group();
    shelterGroup.position.copy(shelterPos);
    this.addGroundDecal(shelterPos.x, shelterPos.z, 6.7, scorchTex, 0.84, 0.03);
    this.addGroundDecal(shelterPos.x, shelterPos.z, 4.4, shadowTex, 0.68, 0.036);
    this.addDustFoundationSkirt(shelterPos.x, shelterPos.z, 3.35, 3.85);

    const baseDeckGeo = this.track(new THREE.CylinderGeometry(3.1, 3.5, 0.45, 20));
    const baseDeck = new THREE.Mesh(baseDeckGeo, hullSlate);
    baseDeck.position.y = 0.22;
    baseDeck.castShadow = !this.isSoftwareGl;
    baseDeck.receiveShadow = !this.isSoftwareGl;
    shelterGroup.add(baseDeck);

    this.shelterHullMat = this.track(
      new THREE.MeshStandardMaterial({
        map: hullSeamTex,
        color: '#E2E8F0',
        emissive: '#000000',
        emissiveIntensity: 0.0,
        roughness: 0.44,
        metalness: 0.42,
      })
    );
    const coreHullGeo = this.track(
      new THREE.SphereGeometry(2.45, 28, 18, 0, Math.PI * 2, 0, Math.PI * 0.52)
    );
    const coreHull = new THREE.Mesh(coreHullGeo, this.shelterHullMat);
    coreHull.position.y = 0.42;
    coreHull.castShadow = !this.isSoftwareGl;
    coreHull.receiveShadow = !this.isSoftwareGl;
    shelterGroup.add(coreHull);

    // Metallic hull structural meridian ribs
    for (let i = 0; i < 4; i++) {
      const ribGeo = this.track(new THREE.TorusGeometry(2.46, 0.045, 8, 28, Math.PI));
      const rib = new THREE.Mesh(ribGeo, darkMetal);
      rib.position.y = 0.42;
      rib.rotation.y = (i / 4) * Math.PI;
      shelterGroup.add(rib);
    }

    // Martian dust accumulation overlay shell on Storm Shelter dome
    const shelterDustGeo = this.track(
      new THREE.SphereGeometry(2.47, 24, 16, 0, Math.PI * 2, 0, Math.PI * 0.48)
    );
    const shelterDustDome = new THREE.Mesh(shelterDustGeo, this.habitatDustMat);
    shelterDustDome.position.y = 0.42;
    shelterDustDome.renderOrder = 1;
    shelterGroup.add(shelterDustDome);

    // Glowing interior crew habitat windows protruding cleanly outside dome shell
    this.shelterWindowMat = this.track(
      new THREE.MeshStandardMaterial({
        color: '#FEF08A',
        emissive: '#F59E0B',
        emissiveIntensity: 0.85,
        roughness: 0.18,
        metalness: 0.12,
      })
    );
    const windowBeltGeo = this.track(new THREE.CylinderGeometry(2.48, 2.52, 0.34, 22));
    const windowBelt = new THREE.Mesh(windowBeltGeo, this.shelterWindowMat);
    windowBelt.position.y = 1.02;
    shelterGroup.add(windowBelt);

    // Roof comms spire & beacon
    const mastGeo = this.track(new THREE.CylinderGeometry(0.08, 0.14, 1.35, 8));
    const mast = new THREE.Mesh(mastGeo, darkMetal);
    mast.position.y = 3.25;
    mast.castShadow = !this.isSoftwareGl;
    shelterGroup.add(mast);

    this.shelterBeaconMat = this.track(
      new THREE.MeshStandardMaterial({
        color: '#38BDF8',
        emissive: '#0EA5E9',
        emissiveIntensity: 1.3,
        roughness: 0.2,
      })
    );
    const beaconGeo = this.track(new THREE.SphereGeometry(0.18, 12, 12));
    const beacon = new THREE.Mesh(beaconGeo, this.shelterBeaconMat);
    beacon.position.y = 3.98;
    shelterGroup.add(beacon);

    // Water-pump toroidal collar
    const collarGeo = this.track(new THREE.TorusGeometry(2.85, 0.18, 10, 28));
    const collar = new THREE.Mesh(collarGeo, darkMetal);
    collar.rotation.x = Math.PI / 2;
    collar.position.y = 0.48;
    shelterGroup.add(collar);

    // Outer Translucent Blue Water-Wall Forcefield Dome
    const shieldGeo = this.track(
      new THREE.SphereGeometry(2.88, 28, 18, 0, Math.PI * 2, 0, Math.PI * 0.56)
    );
    this.waterShieldMat = this.track(
      new THREE.MeshPhysicalMaterial({
        color: '#38BDF8',
        emissive: '#0284C7',
        emissiveIntensity: 0.52,
        transparent: true,
        opacity: 0.34,
        roughness: 0.12,
        metalness: 0.05,
        clearcoat: 0.85,
        depthWrite: false,
      })
    );
    this.waterShieldMesh = new THREE.Mesh(shieldGeo, this.waterShieldMat);
    this.waterShieldMesh.position.y = 0.36;
    this.waterShieldMesh.renderOrder = 2;
    shelterGroup.add(this.waterShieldMesh);

    // Translucent cyan-blue hexagonal Water-Wall energy shimmer dome
    const hexShieldGeo = this.track(
      new THREE.SphereGeometry(3.04, 36, 22, 0, Math.PI * 2, 0, Math.PI * 0.56)
    );
    this.waterShieldHexMat = this.track(
      new THREE.ShaderMaterial({
        transparent: true,
        depthWrite: false,
        blending: THREE.AdditiveBlending,
        side: THREE.DoubleSide,
        uniforms: {
          uTime: { value: 0 },
          uStrength: { value: 0.0 },
          uImpactPulse: { value: 0.0 },
          uImpactPoint: { value: new THREE.Vector3(0.9, 2.8, 1.0) },
        },
        vertexShader: `
          varying vec3 vWorldPos;
          varying vec3 vWorldNormal;
          varying vec2 vUv;
          void main() {
            vUv = uv;
            vec4 wp = modelMatrix * vec4(position, 1.0);
            vWorldPos = wp.xyz;
            vWorldNormal = normalize(mat3(modelMatrix) * normal);
            gl_Position = projectionMatrix * viewMatrix * wp;
          }
        `,
        fragmentShader: `
          uniform float uTime;
          uniform float uStrength;
          uniform float uImpactPulse;
          uniform vec3 uImpactPoint;
          varying vec3 vWorldPos;
          varying vec3 vWorldNormal;
          varying vec2 vUv;

          float hexDist(vec2 p) {
            p = abs(p);
            float c = dot(p, normalize(vec2(1.0, 1.7320508)));
            return max(c, p.x);
          }

          float hexGrid(vec2 uv) {
            vec2 r = vec2(1.0, 1.7320508);
            vec2 h = r * 0.5;
            vec2 a = mod(uv, r) - h;
            vec2 b = mod(uv - h, r) - h;
            vec2 gv = dot(a, a) < dot(b, b) ? a : b;
            return 0.5 - hexDist(gv);
          }

          void main() {
            if (uStrength <= 0.01) discard;
            vec3 viewDir = normalize(cameraPosition - vWorldPos);
            float fresnel = pow(1.0 - abs(dot(normalize(vWorldNormal), viewDir)), 1.9);

            float dHex = hexGrid(vUv * vec2(22.0, 11.0));
            float hexEdge = 1.0 - smoothstep(0.015, 0.075, dHex);
            float waterWave = 0.5 + 0.5 * sin(vWorldPos.y * 5.2 - uTime * 3.8 + vWorldPos.x * 1.4);
            float shimmer = 0.65 + 0.35 * sin(uTime * 4.5 + vUv.x * 18.0 + vUv.y * 10.0);

            float hitDist = distance(vWorldPos, uImpactPoint);
            float impactRing = exp(-hitDist * 1.45) * uImpactPulse;
            float impactWave = (0.5 + 0.5 * sin(hitDist * 9.0 - uTime * 16.0)) * impactRing;

            vec3 deepCyan = vec3(0.05, 0.65, 0.96);
            vec3 brightAqua = vec3(0.45, 0.95, 1.0);
            vec3 col = mix(deepCyan, brightAqua, clamp(hexEdge * 0.75 + fresnel * 0.45 + impactRing, 0.0, 1.0));

            float alpha = (hexEdge * (0.34 + waterWave * 0.28) + fresnel * 0.36 + waterWave * 0.08 + impactWave * 0.65)
              * uStrength * shimmer;
            gl_FragColor = vec4(col, clamp(alpha, 0.0, 0.88));
          }
        `,
      })
    );
    this.waterShieldHexMesh = new THREE.Mesh(hexShieldGeo, this.waterShieldHexMat);
    this.waterShieldHexMesh.position.y = 0.36;
    this.waterShieldHexMesh.renderOrder = 5;
    this.waterShieldHexMesh.visible = false;
    shelterGroup.add(this.waterShieldHexMesh);

    this.scene.add(shelterGroup);
    this.addHitCollider('shelter', shelterPos, 3.4, 4.1);
    this.moduleWorldCenters.shelter.copy(shelterPos);
    this.anchorWorldMap.shelter.set(shelterPos.x + 0.15, shelterPos.y + 4.45, shelterPos.z);

    // ------------------------------------------------------------------
    // 2. SOLAR ARRAY FIELD (Left: -6.8, y, 2.2) — Glossy PBR Silicon Panels
    // ------------------------------------------------------------------
    const solarPos = new THREE.Vector3(-6.8, groundHeightAt(-6.8, 2.2), 2.2);
    const solarGroup = new THREE.Group();
    solarGroup.position.copy(solarPos);
    this.addGroundDecal(solarPos.x, solarPos.z - 0.1, 4.9, shadowTex, 0.65, 0.022);

    const junctionGeo = this.track(new THREE.BoxGeometry(1.5, 0.7, 1.1));
    const junction = new THREE.Mesh(junctionGeo, hullSlate);
    junction.position.set(2.2, 0.35, 0.05);
    junction.castShadow = !this.isSoftwareGl;
    junction.receiveShadow = !this.isSoftwareGl;
    solarGroup.add(junction);

    this.solarBatteryMat = this.track(
      new THREE.MeshStandardMaterial({
        color: '#F59E0B',
        emissive: '#B45309',
        emissiveIntensity: 0.5,
        roughness: 0.32,
        metalness: 0.35,
      })
    );
    const batteryCellGeo = this.track(new THREE.BoxGeometry(0.46, 0.55, 0.46));
    const batteryCell = new THREE.Mesh(batteryCellGeo, this.solarBatteryMat);
    batteryCell.position.set(2.2, 0.88, 0.05);
    batteryCell.castShadow = !this.isSoftwareGl;
    solarGroup.add(batteryCell);

    const frontPostGeo = this.track(new THREE.CylinderGeometry(0.12, 0.18, 1.15, 10));
    const rearPostGeo = this.track(new THREE.CylinderGeometry(0.12, 0.18, 1.65, 10));
    const crossbarGeo = this.track(new THREE.BoxGeometry(5.6, 0.12, 0.14));

    const panelFrameMat = hullWhite;
    this.solarCellMat = this.track(
      new THREE.MeshPhysicalMaterial({
        map: solarTex,
        roughness: 0.14,
        metalness: 0.78,
        clearcoat: 0.92,
        clearcoatRoughness: 0.08,
        envMapIntensity: 1.4,
        emissive: '#0284C7',
        emissiveIntensity: 0.35,
      })
    );
    this.solarDustMat = this.track(
      new THREE.MeshStandardMaterial({
        map: dustOverlayTex,
        color: '#B84B2B',
        roughness: 0.98,
        metalness: 0.0,
        envMapIntensity: 0.12,
        transparent: true,
        opacity: 0.16,
        depthWrite: false,
      })
    );

    const panelFrameGeo = this.track(new THREE.BoxGeometry(2.08, 0.09, 1.48));
    const panelCellGeo = this.track(new THREE.BoxGeometry(1.9, 0.11, 1.3));
    const dustOverlayGeo = this.track(new THREE.PlaneGeometry(1.88, 1.28));
    const wiperGeo = this.track(new THREE.BoxGeometry(0.1, 0.08, 1.36));

    const panelCols = [-2.35, 0, 2.35];
    const rowConfigs = [
      { z: 1.05, y: 1.15, postGeo: frontPostGeo, postY: 0.58 },
      { z: -1.15, y: 1.65, postGeo: rearPostGeo, postY: 0.82 },
    ];

    rowConfigs.forEach((row) => {
      const tiltGroup = new THREE.Group();
      tiltGroup.position.set(0, row.y, row.z);
      tiltGroup.rotation.x = 0.38;

      const crossbar = new THREE.Mesh(crossbarGeo, darkMetal);
      crossbar.position.set(0, -0.08, 0);
      tiltGroup.add(crossbar);

      panelCols.forEach((xOffset) => {
        const post = new THREE.Mesh(row.postGeo, darkMetal);
        post.position.set(xOffset, row.postY, row.z);
        post.castShadow = !this.isSoftwareGl;
        solarGroup.add(post);

        const frame = new THREE.Mesh(panelFrameGeo, panelFrameMat);
        frame.position.set(xOffset, 0, 0);
        frame.castShadow = !this.isSoftwareGl;
        frame.receiveShadow = !this.isSoftwareGl;
        tiltGroup.add(frame);

        const cell = new THREE.Mesh(panelCellGeo, this.solarCellMat);
        cell.position.set(xOffset, 0.02, 0);
        tiltGroup.add(cell);

        const dustOverlay = new THREE.Mesh(dustOverlayGeo, this.solarDustMat);
        dustOverlay.rotation.x = -Math.PI / 2;
        dustOverlay.position.set(xOffset, 0.09, 0);
        dustOverlay.renderOrder = 1;
        tiltGroup.add(dustOverlay);

        const wiper = new THREE.Mesh(wiperGeo, accentAmber);
        wiper.position.set(xOffset - 0.88, 0.125, 0);
        tiltGroup.add(wiper);
        this.solarWipers.push({ mesh: wiper, baseX: xOffset });
      });

      this.solarTiltGroups.push(tiltGroup);
      solarGroup.add(tiltGroup);
    });

    this.scene.add(solarGroup);
    this.addHitCollider('solar', solarPos, 4.1, 3.0);
    this.moduleWorldCenters.solar.copy(solarPos);
    this.anchorWorldMap.solar.set(solarPos.x - 0.5, solarPos.y + 3.05, solarPos.z);

    // ------------------------------------------------------------------
    // 3. MOXIE & LIFE SUPPORT CORE (Back-Left: -3.7, y, -4.4)
    // ------------------------------------------------------------------
    const moxiePos = new THREE.Vector3(-3.7, groundHeightAt(-3.7, -4.4), -4.4);
    const moxieGroup = new THREE.Group();
    moxieGroup.position.copy(moxiePos);
    this.addGroundDecal(moxiePos.x, moxiePos.z, 5.1, scorchTex, 0.72, 0.03);
    this.addGroundDecal(moxiePos.x, moxiePos.z, 3.4, shadowTex, 0.65, 0.036);
    this.addDustFoundationSkirt(moxiePos.x, moxiePos.z, 2.1, 2.45);

    const moxiePad = new THREE.Mesh(this.track(new THREE.BoxGeometry(3.8, 0.4, 2.8)), hullSlate);
    moxiePad.position.y = 0.2;
    moxiePad.castShadow = !this.isSoftwareGl;
    moxiePad.receiveShadow = !this.isSoftwareGl;
    moxieGroup.add(moxiePad);

    const moxieBody = new THREE.Mesh(this.track(new THREE.BoxGeometry(2.3, 1.65, 1.8)), hullWhite);
    moxieBody.position.set(-0.35, 1.18, 0);
    moxieBody.castShadow = !this.isSoftwareGl;
    moxieBody.receiveShadow = !this.isSoftwareGl;
    moxieGroup.add(moxieBody);

    // 3 glowing cyan O2 tanks
    this.moxieTankMat = this.track(
      new THREE.MeshStandardMaterial({
        color: '#22D3EE',
        emissive: '#06B6D4',
        emissiveIntensity: 0.72,
        roughness: 0.24,
        metalness: 0.42,
      })
    );
    const tankGeo = this.track(new THREE.CylinderGeometry(0.32, 0.32, 1.85, 14));
    for (let i = 0; i < 3; i++) {
      const tank = new THREE.Mesh(tankGeo, this.moxieTankMat);
      tank.position.set(1.22, 1.32, -0.58 + i * 0.58);
      tank.castShadow = !this.isSoftwareGl;
      moxieGroup.add(tank);
    }

    // Intake CO2 turbine fan on roof
    const intakeHood = new THREE.Mesh(
      this.track(new THREE.CylinderGeometry(0.62, 0.72, 0.45, 16)),
      darkMetal
    );
    intakeHood.position.set(-0.35, 2.22, 0);
    intakeHood.castShadow = !this.isSoftwareGl;
    moxieGroup.add(intakeHood);

    this.moxieFan = new THREE.Group();
    this.moxieFan.position.set(-0.35, 2.48, 0);
    const bladeGeo = this.track(new THREE.BoxGeometry(0.98, 0.05, 0.16));
    const blade1 = new THREE.Mesh(bladeGeo, accentAmber);
    const blade2 = new THREE.Mesh(bladeGeo, accentAmber);
    blade2.rotation.y = Math.PI / 2;
    this.moxieFan.add(blade1, blade2);
    moxieGroup.add(this.moxieFan);

    // Pooled rising O2 bubbles
    const bubbleGeo = this.track(new THREE.SphereGeometry(0.11, 8, 8));
    const bubbleMat = this.track(
      new THREE.MeshStandardMaterial({
        color: '#A5F3FC',
        emissive: '#22D3EE',
        emissiveIntensity: 1.1,
        transparent: true,
        opacity: 0.85,
      })
    );
    for (let i = 0; i < 12; i++) {
      const b = new THREE.Mesh(bubbleGeo, bubbleMat);
      b.position.set(1.05 + (i % 3) * 0.2, 2.1 + (i / 12) * 2.1, -0.35 + ((i * 7) % 5) * 0.16);
      b.userData.phase = i * 0.55;
      this.moxieBubbles.push(b);
      moxieGroup.add(b);
    }

    this.scene.add(moxieGroup);
    this.addHitCollider('moxie', moxiePos, 2.6, 3.2);
    this.moduleWorldCenters.moxie.copy(moxiePos);
    this.anchorWorldMap.moxie.set(moxiePos.x - 0.5, moxiePos.y + 3.65, moxiePos.z);

    // ------------------------------------------------------------------
    // 4. HYDROPONIC BIO-DOME (Right: 6.8, y, 2.0)
    // ------------------------------------------------------------------
    const bioPos = new THREE.Vector3(6.8, groundHeightAt(6.8, 2.0), 2.0);
    const bioGroup = new THREE.Group();
    bioGroup.position.copy(bioPos);
    this.addGroundDecal(bioPos.x, bioPos.z, 5.3, scorchTex, 0.68, 0.03);
    this.addGroundDecal(bioPos.x, bioPos.z, 3.6, shadowTex, 0.65, 0.036);
    this.addDustFoundationSkirt(bioPos.x, bioPos.z, 2.75, 3.18);

    const bioBase = new THREE.Mesh(
      this.track(new THREE.CylinderGeometry(2.55, 2.85, 0.45, 20)),
      hullSlate
    );
    bioBase.position.y = 0.22;
    bioBase.castShadow = !this.isSoftwareGl;
    bioBase.receiveShadow = !this.isSoftwareGl;
    bioGroup.add(bioBase);

    const soilTray = new THREE.Mesh(
      this.track(new THREE.CylinderGeometry(2.1, 2.1, 0.16, 18)),
      this.track(
        new THREE.MeshStandardMaterial({
          color: '#2D1B12',
          emissive: '#064E3B',
          emissiveIntensity: 0.35,
          roughness: 0.92,
        })
      )
    );
    soilTray.position.y = 0.48;
    soilTray.receiveShadow = !this.isSoftwareGl;
    bioGroup.add(soilTray);

    // 3D Green Hydroponic Plants (scale dynamically with cropGrowth)
    const stemMat = this.track(
      new THREE.MeshStandardMaterial({
        color: '#10B981',
        emissive: '#059669',
        emissiveIntensity: 0.52,
        roughness: 0.45,
      })
    );
    this.bioLeafMat = this.track(
      new THREE.MeshStandardMaterial({
        color: '#22C55E',
        emissive: '#10B981',
        emissiveIntensity: 0.65,
        roughness: 0.38,
      })
    );
    const stemGeo = this.track(new THREE.CylinderGeometry(0.06, 0.08, 0.65, 8));
    const leafGeo = this.track(new THREE.SphereGeometry(0.24, 10, 8));

    for (let i = 0; i < 8; i++) {
      const plant = new THREE.Group();
      const angle = (i / 8) * Math.PI * 2;
      const dist = i % 2 === 0 ? 1.22 : 0.72;
      plant.position.set(Math.cos(angle) * dist, 0.56, Math.sin(angle) * dist);

      const stem = new THREE.Mesh(stemGeo, stemMat);
      stem.position.y = 0.32;
      plant.add(stem);

      const crown = new THREE.Mesh(leafGeo, this.bioLeafMat);
      crown.scale.set(1.25, 0.85, 1.25);
      crown.position.y = 0.72;
      plant.add(crown);

      this.biodomeCrops.push(plant);
      bioGroup.add(plant);
    }

    // LED Grow Light Ring inside dome
    this.bioLedRingMat = this.track(
      new THREE.MeshStandardMaterial({
        color: '#E879F9',
        emissive: '#D946EF',
        emissiveIntensity: 0.9,
        roughness: 0.25,
      })
    );
    const ledRing = new THREE.Mesh(
      this.track(new THREE.TorusGeometry(1.45, 0.06, 8, 24)),
      this.bioLedRingMat
    );
    ledRing.rotation.x = Math.PI / 2;
    ledRing.position.y = 1.9;
    bioGroup.add(ledRing);

    // Outer Glossy Clearcoat Glass Dome
    this.biodomeGlassMat = this.track(
      new THREE.MeshPhysicalMaterial({
        color: '#A7F3D0',
        emissive: '#10B981',
        emissiveIntensity: 0.24,
        transparent: true,
        opacity: 0.28,
        roughness: 0.08,
        metalness: 0.05,
        clearcoat: 0.95,
        clearcoatRoughness: 0.06,
        depthWrite: false,
      })
    );
    const glassDome = new THREE.Mesh(
      this.track(new THREE.SphereGeometry(2.42, 26, 18, 0, Math.PI * 2, 0, Math.PI * 0.54)),
      this.biodomeGlassMat
    );
    glassDome.position.y = 0.42;
    glassDome.renderOrder = 3;
    bioGroup.add(glassDome);

    // Martian dust accumulation layer on Bio-Dome glass
    const bioDustDome = new THREE.Mesh(
      this.track(new THREE.SphereGeometry(2.435, 24, 16, 0, Math.PI * 2, 0, Math.PI * 0.5)),
      this.habitatDustMat
    );
    bioDustDome.position.y = 0.42;
    bioDustDome.renderOrder = 4;
    bioGroup.add(bioDustDome);

    this.scene.add(bioGroup);
    this.addHitCollider('biodome', bioPos, 2.7, 3.0);
    this.moduleWorldCenters.biodome.copy(bioPos);
    this.anchorWorldMap.biodome.set(bioPos.x + 0.35, bioPos.y + 3.35, bioPos.z);

    // ------------------------------------------------------------------
    // 5. CONNECTING PRESSURIZED AIRLOCK TUNNELS
    // ------------------------------------------------------------------
    const connectTunnel = (
      a: THREE.Vector3,
      b: THREE.Vector3,
      radius: number,
      mat: THREE.Material,
      addDustSleeve = false
    ) => {
      const dir = new THREE.Vector3().subVectors(b, a);
      const len = dir.length();
      const mid = new THREE.Vector3().addVectors(a, b).multiplyScalar(0.5);
      const q = new THREE.Quaternion().setFromUnitVectors(
        new THREE.Vector3(0, 1, 0),
        dir.clone().normalize()
      );
      const tunnelGeo = this.track(new THREE.CylinderGeometry(radius, radius, len, 14));
      const tunnel = new THREE.Mesh(tunnelGeo, mat);
      tunnel.position.set(mid.x, Math.max(a.y, b.y) + radius + 0.1, mid.z);
      tunnel.quaternion.copy(q);
      tunnel.castShadow = !this.isSoftwareGl;
      tunnel.receiveShadow = !this.isSoftwareGl;
      this.scene.add(tunnel);

      if (addDustSleeve) {
        const dustSleeveGeo = this.track(
          new THREE.CylinderGeometry(radius + 0.015, radius + 0.015, len * 0.92, 12, 1, true)
        );
        const sleeve = new THREE.Mesh(dustSleeveGeo, this.habitatDustMat);
        sleeve.position.copy(tunnel.position);
        sleeve.quaternion.copy(q);
        sleeve.renderOrder = 2;
        this.scene.add(sleeve);
      }
    };

    connectTunnel(shelterPos, moxiePos, 0.48, hullWhite, true);
    connectTunnel(shelterPos, bioPos, 0.48, hullWhite, true);
  }

  private buildStorytellingProps(): void {
    const maxAniso = this.renderer.capabilities.getMaxAnisotropy();
    const shadowTex = this.track(createContactShadowTexture());
    const trackTex = this.track(createRoverTrackTexture(maxAniso));

    const cableMat = this.track(
      new THREE.MeshStandardMaterial({ color: '#1E293B', roughness: 0.55, metalness: 0.45 })
    );
    const cableStripeMat = this.track(
      new THREE.MeshStandardMaterial({ color: '#EA580C', roughness: 0.45, metalness: 0.25 })
    );
    const steelMat = this.track(
      new THREE.MeshStandardMaterial({ color: '#475569', roughness: 0.35, metalness: 0.78 })
    );
    const darkTitaniumMat = this.track(
      new THREE.MeshStandardMaterial({ color: '#1E293B', roughness: 0.36, metalness: 0.85 })
    );
    const whiteAvionicsMat = this.track(
      new THREE.MeshStandardMaterial({ color: '#E2E8F0', roughness: 0.35, metalness: 0.35 })
    );
    const goldFoilMat = this.track(
      new THREE.MeshStandardMaterial({ color: '#F59E0B', roughness: 0.24, metalness: 0.88 })
    );

    // ------------------------------------------------------------------
    // 1. ROVER TRACKS & FOOTPRINT TRAILS CONNECTING MODULES
    // ------------------------------------------------------------------
    this.buildTerrainTrackRibbon(
      [
        new THREE.Vector3(-4.8, 0, 3.5),
        new THREE.Vector3(-2.8, 0, 4.1),
        new THREE.Vector3(-0.6, 0, 4.3),
        new THREE.Vector3(1.4, 0, 4.3),
        new THREE.Vector3(3.2, 0, 4.4),
      ],
      1.15,
      trackTex
    );
    this.buildTerrainTrackRibbon(
      [
        new THREE.Vector3(3.6, 0, 4.4),
        new THREE.Vector3(4.9, 0, 4.1),
        new THREE.Vector3(6.1, 0, 3.4),
        new THREE.Vector3(6.8, 0, 2.5),
      ],
      1.15,
      trackTex
    );

    // ------------------------------------------------------------------
    // 2. GROUND-HUGGING POWER CABLES
    // ------------------------------------------------------------------
    const addGroundCable = (waypoints: Array<[number, number]>, mat: THREE.Material) => {
      const pts = waypoints.map(([x, z]) => new THREE.Vector3(x, groundHeightAt(x, z) + 0.06, z));
      const curve = new THREE.CatmullRomCurve3(pts);
      const cableGeo = this.track(new THREE.TubeGeometry(curve, 24, 0.065, 8, false));
      const cable = new THREE.Mesh(cableGeo, mat);
      cable.castShadow = !this.isSoftwareGl;
      this.scene.add(cable);

      // Anchor clamps along the cable
      const clampGeo = this.track(new THREE.BoxGeometry(0.18, 0.1, 0.18));
      for (const t of [0.22, 0.5, 0.78]) {
        const p = curve.getPoint(t);
        const clamp = new THREE.Mesh(clampGeo, steelMat);
        clamp.position.set(p.x, p.y - 0.01, p.z);
        this.scene.add(clamp);
      }
    };

    addGroundCable(
      [
        [-4.5, 2.2],
        [-3.2, 1.9],
        [-1.8, 1.6],
        [-0.6, 1.2],
      ],
      cableStripeMat
    );
    addGroundCable(
      [
        [-4.5, 2.0],
        [-4.8, 0.2],
        [-4.4, -1.8],
        [-3.9, -3.3],
      ],
      cableMat
    );
    addGroundCable(
      [
        [2.6, 1.3],
        [3.9, 1.6],
        [4.9, 1.8],
      ],
      cableStripeMat
    );

    // ------------------------------------------------------------------
    // 3. COMMS DISH WITH BLINKING RED BEACON (Back-Right: 5.05, y, -4.1)
    // ------------------------------------------------------------------
    const commsPos = new THREE.Vector3(5.05, groundHeightAt(5.05, -4.1), -4.1);
    const commsGroup = new THREE.Group();
    commsGroup.position.copy(commsPos);
    this.addGroundDecal(commsPos.x, commsPos.z, 1.65, shadowTex, 0.6, 0.03);

    const commsPad = new THREE.Mesh(
      this.track(new THREE.CylinderGeometry(0.95, 1.1, 0.22, 12)),
      steelMat
    );
    commsPad.position.y = 0.11;
    commsPad.castShadow = !this.isSoftwareGl;
    commsPad.receiveShadow = !this.isSoftwareGl;
    commsGroup.add(commsPad);

    const commsBox = new THREE.Mesh(this.track(new THREE.BoxGeometry(0.7, 0.55, 0.55)), whiteAvionicsMat);
    commsBox.position.set(0.32, 0.48, 0);
    commsBox.castShadow = !this.isSoftwareGl;
    commsGroup.add(commsBox);

    const pedestal = new THREE.Mesh(
      this.track(new THREE.CylinderGeometry(0.14, 0.22, 1.55, 10)),
      darkTitaniumMat
    );
    pedestal.position.set(-0.12, 0.95, 0);
    pedestal.castShadow = !this.isSoftwareGl;
    commsGroup.add(pedestal);

    // Parabolic dish via LatheGeometry
    const dishProfile = [
      new THREE.Vector2(0.0, 0.0),
      new THREE.Vector2(0.35, 0.08),
      new THREE.Vector2(0.7, 0.24),
      new THREE.Vector2(0.98, 0.48),
    ];
    const dishGeo = this.track(new THREE.LatheGeometry(dishProfile, 20));
    const dishHolder = new THREE.Group();
    dishHolder.position.set(-0.12, 1.85, 0);
    dishHolder.rotation.set(-0.55, -0.35, 0);

    const dishMat = this.track(
      new THREE.MeshStandardMaterial({
        color: '#E2E8F0',
        roughness: 0.3,
        metalness: 0.55,
        side: THREE.DoubleSide,
      })
    );
    const dishMesh = new THREE.Mesh(dishGeo, dishMat);
    dishMesh.castShadow = !this.isSoftwareGl;
    dishHolder.add(dishMesh);

    const feedHorn = new THREE.Mesh(
      this.track(new THREE.CylinderGeometry(0.05, 0.09, 0.72, 8)),
      goldFoilMat
    );
    feedHorn.position.y = 0.42;
    dishHolder.add(feedHorn);
    commsGroup.add(dishHolder);

    // Secondary antenna mast with blinking red beacon
    const beaconPole = new THREE.Mesh(
      this.track(new THREE.CylinderGeometry(0.04, 0.06, 2.2, 8)),
      darkTitaniumMat
    );
    beaconPole.position.set(0.5, 1.3, -0.15);
    commsGroup.add(beaconPole);

    this.commsBeaconMat = this.track(
      new THREE.MeshStandardMaterial({
        color: '#EF4444',
        emissive: '#FF2424',
        emissiveIntensity: 2.2,
        roughness: 0.2,
      })
    );
    const redBeacon = new THREE.Mesh(
      this.track(new THREE.SphereGeometry(0.11, 10, 10)),
      this.commsBeaconMat
    );
    redBeacon.position.set(0.5, 2.45, -0.15);
    commsGroup.add(redBeacon);

    this.scene.add(commsGroup);

    // ------------------------------------------------------------------
    // 4. EXTERNAL O2 / H2O STORAGE TANK RACK (Foreground-Left: -3.55, y, 4.35)
    // ------------------------------------------------------------------
    const rackPos = new THREE.Vector3(-3.55, groundHeightAt(-3.55, 4.35), 4.35);
    const rackGroup = new THREE.Group();
    rackGroup.position.copy(rackPos);
    rackGroup.rotation.y = 0.28;
    this.addGroundDecal(rackPos.x, rackPos.z, 1.65, shadowTex, 0.62, 0.03);

    const skid = new THREE.Mesh(this.track(new THREE.BoxGeometry(2.0, 0.16, 1.2)), steelMat);
    skid.position.y = 0.08;
    skid.receiveShadow = !this.isSoftwareGl;
    rackGroup.add(skid);

    const o2TankMat = this.track(
      new THREE.MeshStandardMaterial({ color: '#0891B2', roughness: 0.3, metalness: 0.5 })
    );
    const h2oTankMat = this.track(
      new THREE.MeshStandardMaterial({ color: '#0284C7', roughness: 0.28, metalness: 0.52 })
    );
    const hTankGeo = this.track(new THREE.CylinderGeometry(0.26, 0.26, 1.35, 12));
    hTankGeo.rotateZ(Math.PI / 2);
    for (let i = 0; i < 2; i++) {
      const hTank = new THREE.Mesh(hTankGeo, i === 0 ? o2TankMat : h2oTankMat);
      hTank.position.set(-0.22, 0.48 + i * 0.42, 0.22);
      hTank.castShadow = !this.isSoftwareGl;
      rackGroup.add(hTank);
    }

    const vTank = new THREE.Mesh(
      this.track(new THREE.CylinderGeometry(0.28, 0.28, 1.25, 14)),
      whiteAvionicsMat
    );
    vTank.position.set(0.58, 0.78, -0.2);
    vTank.castShadow = !this.isSoftwareGl;
    rackGroup.add(vTank);

    const pipeAmberMat = this.track(
      new THREE.MeshStandardMaterial({ color: '#D97706', roughness: 0.4, metalness: 0.5 })
    );
    const pipeCollar = new THREE.Mesh(
      this.track(new THREE.CylinderGeometry(0.06, 0.06, 1.05, 8)),
      pipeAmberMat
    );
    pipeCollar.position.set(0.15, 0.68, -0.2);
    rackGroup.add(pipeCollar);

    this.scene.add(rackGroup);

    // ------------------------------------------------------------------
    // 5. PARKED MINI-ROVER (Front-Right: 3.6, y, 4.4)
    // ------------------------------------------------------------------
    const roverPos = new THREE.Vector3(3.6, groundHeightAt(3.6, 4.4), 4.4);
    const roverGroup = new THREE.Group();
    roverGroup.position.copy(roverPos);
    roverGroup.rotation.y = -0.62;
    this.addGroundDecal(roverPos.x, roverPos.z, 1.45, shadowTex, 0.65, 0.025);

    const roverBody = new THREE.Mesh(
      this.track(new THREE.BoxGeometry(1.15, 0.42, 0.76)),
      whiteAvionicsMat
    );
    roverBody.position.y = 0.46;
    roverBody.castShadow = !this.isSoftwareGl;
    roverGroup.add(roverBody);

    const foilPanel = new THREE.Mesh(
      this.track(new THREE.BoxGeometry(0.72, 0.44, 0.79)),
      goldFoilMat
    );
    foilPanel.position.set(-0.08, 0.46, 0);
    roverGroup.add(foilPanel);

    // 6 treaded rover wheels with rocker-bogie side struts
    const wheelGeo = this.track(new THREE.CylinderGeometry(0.19, 0.19, 0.14, 14));
    wheelGeo.rotateX(Math.PI / 2);
    const wheelMat = this.track(
      new THREE.MeshStandardMaterial({ color: '#27272A', roughness: 0.75, metalness: 0.35 })
    );
    for (const xW of [-0.46, 0.0, 0.46]) {
      for (const zW of [-0.46, 0.46]) {
        const wheel = new THREE.Mesh(wheelGeo, wheelMat);
        wheel.position.set(xW, 0.19, zW);
        wheel.castShadow = !this.isSoftwareGl;
        roverGroup.add(wheel);
      }
    }

    const strutGeo = this.track(new THREE.BoxGeometry(0.95, 0.06, 0.06));
    for (const zS of [-0.4, 0.4]) {
      const strut = new THREE.Mesh(strutGeo, darkTitaniumMat);
      strut.position.set(0, 0.31, zS);
      roverGroup.add(strut);
    }

    // Mast & optical ChemCam head
    const mastPole = new THREE.Mesh(
      this.track(new THREE.CylinderGeometry(0.04, 0.05, 0.68, 8)),
      whiteAvionicsMat
    );
    mastPole.position.set(0.38, 0.96, 0.16);
    mastPole.castShadow = !this.isSoftwareGl;
    roverGroup.add(mastPole);

    const camHead = new THREE.Mesh(
      this.track(new THREE.BoxGeometry(0.22, 0.14, 0.28)),
      whiteAvionicsMat
    );
    camHead.position.set(0.42, 1.32, 0.16);
    camHead.castShadow = !this.isSoftwareGl;
    roverGroup.add(camHead);

    const lensMat = this.track(
      new THREE.MeshStandardMaterial({
        color: '#38BDF8',
        emissive: '#0284C7',
        emissiveIntensity: 0.9,
        roughness: 0.15,
      })
    );
    const lens = new THREE.Mesh(this.track(new THREE.CylinderGeometry(0.045, 0.045, 0.05, 10)), lensMat);
    lens.rotation.z = Math.PI / 2;
    lens.position.set(0.54, 1.32, 0.16);
    roverGroup.add(lens);

    // Rear angled MMRTG power fins
    const rtg = new THREE.Mesh(
      this.track(new THREE.CylinderGeometry(0.11, 0.14, 0.46, 8)),
      darkTitaniumMat
    );
    rtg.position.set(-0.68, 0.52, 0);
    rtg.rotation.z = 0.95;
    rtg.castShadow = !this.isSoftwareGl;
    roverGroup.add(rtg);

    this.scene.add(roverGroup);
  }

  private buildHazardsAndRings(): void {
    const ringGeo = this.track(new THREE.RingGeometry(2.85, 3.25, 36));
    ringGeo.rotateX(-Math.PI / 2);
    const ringMat = this.track(
      new THREE.MeshBasicMaterial({
        color: '#22D3EE',
        side: THREE.DoubleSide,
        transparent: true,
        opacity: 0.72,
        depthWrite: false,
      })
    );
    this.targetRing = new THREE.Mesh(ringGeo, ringMat);
    this.scene.add(this.targetRing);

    const hoverGeo = this.track(new THREE.RingGeometry(2.6, 2.82, 36));
    hoverGeo.rotateX(-Math.PI / 2);
    const hoverMat = this.track(
      new THREE.MeshBasicMaterial({
        color: '#FBBF24',
        side: THREE.DoubleSide,
        transparent: true,
        opacity: 0.82,
        depthWrite: false,
      })
    );
    this.hoverRing = new THREE.Mesh(hoverGeo, hoverMat);
    this.hoverRing.visible = false;
    this.scene.add(this.hoverRing);

    // Subtle wind-blown Martian dust particles using soft radial sprite
    const dustSprite = this.track(createDustParticleSpriteTexture());
    const dustCount = 220;
    this.dustPositions = new Float32Array(dustCount * 3);
    let seed = 211;
    const rand = () => {
      seed = (seed * 1664525 + 1013904223) % 4294967296;
      return seed / 4294967296;
    };
    for (let i = 0; i < dustCount; i++) {
      this.dustPositions[i * 3] = (rand() - 0.5) * 38;
      this.dustPositions[i * 3 + 1] = 0.4 + rand() * 6.2;
      this.dustPositions[i * 3 + 2] = (rand() - 0.5) * 28;
    }
    const dustGeo = this.track(new THREE.BufferGeometry());
    dustGeo.setAttribute('position', new THREE.BufferAttribute(this.dustPositions, 3));
    this.dustStormMat = this.track(
      new THREE.PointsMaterial({
        map: dustSprite,
        color: '#F5B895',
        size: 0.55,
        transparent: true,
        opacity: 0.22,
        depthWrite: false,
      })
    );
    this.dustStormPoints = new THREE.Points(dustGeo, this.dustStormMat);
    this.scene.add(this.dustStormPoints);

    // Solar Flare & CME Radiation Particle Streaks (atmospheric + dome-targeted)
    this.flareBoltsGroup = new THREE.Group();
    this.flareBoltsGroup.visible = false;
    const boltGeo = this.track(new THREE.CylinderGeometry(0.038, 0.014, 2.35, 6));
    const streakCount = 32;
    for (let i = 0; i < streakCount; i++) {
      const boltMat = this.track(
        new THREE.MeshBasicMaterial({
          color: '#FB7185',
          transparent: true,
          opacity: 0.92,
          blending: THREE.AdditiveBlending,
          depthWrite: false,
        })
      );
      const bolt = new THREE.Mesh(boltGeo, boltMat);
      const angle = (i / streakCount) * Math.PI * 2;
      const skyStreak = i >= 14;
      const radius = skyStreak ? 4.5 + (i % 7) * 2.4 : 0.6 + (i % 4) * 0.78;
      bolt.position.set(
        0.9 + Math.cos(angle) * radius + (skyStreak ? 2.5 : 1.8),
        10.5 + (i % 6) * 2.1,
        1.0 + Math.sin(angle) * radius - (skyStreak ? 2.0 : 1.0)
      );
      const vel = new THREE.Vector3(-0.34, -1.0, 0.16).normalize();
      bolt.quaternion.setFromUnitVectors(this.streakUpAxis, vel);
      this.flareBoltsGroup.add(bolt);
      this.flareBolts.push({
        mesh: bolt,
        mat: boltMat,
        vel,
        speed: 12 + (i % 5) * 2.4,
        angle,
        radius,
        deflected: false,
        skyStreak,
      });
    }
    this.scene.add(this.flareBoltsGroup);
  }

  public updateMissionVisuals(
    resources: MissionResources,
    tau: number,
    targetModule: ModuleId | 'any',
    emergencyMode: EmergencyMode = 'none',
    timeline?: AtmosphereTimelineInput
  ): void {
    this.resources = resources;
    this.currentTau = tau;
    this.highlightedTarget = targetModule;
    this.emergencyMode = emergencyMode;

    if (timeline) {
      this.targetAtmosphere = timeline.atmosphere;
      this.targetHazardType = timeline.hazardType;
      this.activeSystemGlow = timeline.activeSystemGlow;
      this.simSpeedMultiplier = Math.max(1, timeline.speedMultiplier || 1);
      if (timeline.donki) {
        this.donkiTelemetry = timeline.donki;
      }
    } else {
      this.targetAtmosphere = {
        solProgress: 0,
        crisisIntensity: emergencyMode !== 'none' ? 0.65 : 0.15,
        effectiveTau: tau,
        fogDensity: computeTauFogDensity(tau, emergencyMode),
        sunElevation: 0.95,
        nightFactor: 0,
      };
    }

    this.syncVisualState();
  }

  private syncVisualState(): void {
    if (!this.waterShieldMesh) return;

    const wallRatio = THREE.MathUtils.clamp((this.resources.waterWallCm - 8) / 28, 0, 1);
    const shieldScale = 0.96 + wallRatio * 0.16;
    if (!this.activePulse || this.activePulse.moduleId !== 'shelter') {
      this.waterShieldMesh.scale.set(shieldScale, shieldScale, shieldScale);
    }
    this.waterShieldMat.opacity = 0.2 + wallRatio * 0.44;

    const dustRatio = THREE.MathUtils.clamp(this.resources.panelDustPct / 85, 0, 1);
    this.solarDustMat.opacity = THREE.MathUtils.clamp(dustRatio * 0.86, 0.03, 0.88);
    if (this.habitatDustMat) {
      const stormBoost = this.smoothedEffectiveTau >= 0.85 ? 0.28 : 0;
      const baseDust = Math.max(0, dustRatio - 0.12) * 0.65;
      this.habitatDustMat.opacity = THREE.MathUtils.clamp(baseDust + stormBoost, 0.04, 0.78);
    }

    const cropScale = 0.55 + (this.resources.cropGrowth - 1) * 0.24;
    for (let i = 0; i < this.biodomeCrops.length; i++) {
      const s = cropScale * (i % 2 === 0 ? 1.0 : 0.86);
      this.biodomeCrops[i].scale.set(s, s * 1.12, s);
    }

    if (this.highlightedTarget === 'any') {
      this.targetRing.visible = false;
    } else {
      this.targetRing.visible = true;
      const center = this.moduleWorldCenters[this.highlightedTarget];
      this.targetRing.position.set(center.x, center.y + 0.07, center.z);
    }

    if (this.fog && this.totalTime === 0) {
      this.fog.density = this.targetAtmosphere.fogDensity;
      this.smoothedSolProgress = this.targetAtmosphere.solProgress;
      this.smoothedCrisisIntensity = this.targetAtmosphere.crisisIntensity;
      this.smoothedEffectiveTau = this.targetAtmosphere.effectiveTau;
      this.smoothedSunElevation = this.targetAtmosphere.sunElevation;
      this.smoothedNightFactor = this.targetAtmosphere.nightFactor;
    }
  }

  public triggerModuleFeedback(moduleId: ModuleId): void {
    this.activePulse = {
      moduleId,
      startTimeMs: performance.now(),
      duration: 1.25,
    };
    this.moduleBoostTimers[moduleId] = performance.now() + 6500;
    if (moduleId === 'shelter') {
      this.shieldBoostUntilMs = performance.now() + 11000;
      this.shieldImpactPulse = Math.max(this.shieldImpactPulse, 0.9);
    }
  }

  public triggerNightCycle(
    hazardType: SolHazardType,
    shieldBlockedFlare: boolean,
    onComplete: () => void,
    durationSec = 3.0
  ): void {
    if (this.nightAnim.active) return;
    this.nightAnim = {
      active: true,
      startTimeMs: performance.now(),
      duration: Math.max(0.5, durationSec),
      hazardType,
      shieldBlockedFlare,
      completed: false,
      onComplete,
    };
    if (hazardType === 'flare') {
      this.flareBoltsGroup.visible = true;
      for (let i = 0; i < this.flareBolts.length; i++) {
        const b = this.flareBolts[i];
        b.mesh.position.set(
          Math.cos(b.angle) * b.radius,
          7 + (i % 4) * 1.6,
          Math.sin(b.angle) * b.radius
        );
        b.mesh.rotation.set(0, 0, 0);
        b.deflected = false;
      }
    }
  }

  private bindEvents(): void {
    const dom = this.renderer.domElement;
    dom.addEventListener('pointerdown', this.handlePointerDown);
    dom.addEventListener('pointerup', this.handlePointerUp);
    dom.addEventListener('pointermove', this.handlePointerMove);
    dom.addEventListener('webglcontextlost', this.handleContextLost);
    window.addEventListener('resize', this.handleResize);
  }

  private handleContextLost = (e: Event): void => {
    e.preventDefault();
    this.callbacks.onWebglError?.('3D WebGL context was interrupted.');
  };

  private applyResponsiveCamera(width: number, height: number, resetPos = false): void {
    const aspect = width / height;
    const isPortrait = aspect < 1.0;
    this.camera.fov = isPortrait ? 65 : 46;
    this.camera.aspect = aspect;
    this.camera.updateProjectionMatrix();
    if (resetPos) {
      if (isPortrait) {
        this.camera.position.set(0, 7.6, 30.5);
        this.controls?.target.set(0, 2.4, -4.5);
      } else {
        this.camera.position.set(0, 4.45, 21.8);
        this.controls?.target.set(0, 2.85, -5.5);
      }
    }
  }

  private handleResize = (): void => {
    if (this.destroyed || !this.renderer) return;
    const width = Math.max(this.container.clientWidth || window.innerWidth, 320);
    const height = Math.max(this.container.clientHeight || window.innerHeight, 320);
    this.applyResponsiveCamera(width, height, true);
    this.controls.update();
    this.renderer.setSize(width, height);
    this.renderer.render(this.scene, this.camera);
    this.emitProjectedAnchors();
  };

  private updatePointerNdc(e: PointerEvent): void {
    const rect = this.renderer.domElement.getBoundingClientRect();
    this.pointerNdc.x = ((e.clientX - rect.left) / rect.width) * 2 - 1;
    this.pointerNdc.y = -((e.clientY - rect.top) / rect.height) * 2 + 1;
  }

  private pickModule(e: PointerEvent): ModuleId | null {
    this.updatePointerNdc(e);
    this.raycaster.setFromCamera(this.pointerNdc, this.camera);
    const hits = this.raycaster.intersectObjects(this.hitColliders, false);
    if (hits.length === 0) return null;
    return (hits[0].object.userData.moduleId as ModuleId) ?? null;
  }

  private handlePointerDown = (e: PointerEvent): void => {
    this.pointerDownPos = { x: e.clientX, y: e.clientY };
  };

  private handlePointerMove = (e: PointerEvent): void => {
    if (this.nightAnim.active) return;
    const hit = this.pickModule(e);
    if (hit !== this.hoveredModule) {
      this.hoveredModule = hit;
      this.renderer.domElement.style.cursor = hit ? 'pointer' : 'grab';
      if (hit) {
        const center = this.moduleWorldCenters[hit];
        this.hoverRing.position.set(center.x, center.y + 0.08, center.z);
        this.hoverRing.visible = true;
      } else {
        this.hoverRing.visible = false;
      }
      this.callbacks.onHoverModule(hit);
    }
  };

  private handlePointerUp = (e: PointerEvent): void => {
    const travel = Math.hypot(e.clientX - this.pointerDownPos.x, e.clientY - this.pointerDownPos.y);
    if (travel > 6 || this.nightAnim.active) return;
    const hit = this.pickModule(e);
    if (hit) {
      this.callbacks.onModuleClick(hit);
    }
  };

  private startLoop(): void {
    const scheduleNext = () => {
      if (this.destroyed) return;
      if (this.isSoftwareGl) {
        this.afterDraw(() => {
          if (!this.destroyed) this.tick();
        });
      } else {
        this.rafId = requestAnimationFrame(() => this.tick());
      }
    };
    scheduleNext();
  }

  private afterDraw(cb: () => void): void {
    const gl = this.renderer.getContext() as WebGL2RenderingContext;
    if (typeof gl.fenceSync === 'function') {
      const sync = gl.fenceSync(gl.SYNC_GPU_COMMANDS_COMPLETE, 0);
      gl.flush();
      const poll = () => {
        if (this.destroyed) return;
        const status = sync ? gl.clientWaitSync(sync, 0, 0) : gl.ALREADY_SIGNALED;
        if (status === gl.ALREADY_SIGNALED || status === gl.CONDITION_SATISFIED) {
          if (sync) gl.deleteSync(sync);
          this.swTimerId = window.setTimeout(cb, 80);
        } else {
          this.swTimerId = window.setTimeout(poll, 16);
        }
      };
      this.swTimerId = window.setTimeout(poll, 16);
    } else {
      this.swTimerId = window.setTimeout(cb, 90);
    }
  }

  private tick(): void {
    if (this.destroyed || !this.renderer) return;

    const now = performance.now();
    const dt = Math.min((now - this.lastFrameTime) / 1000, 0.1);
    this.lastFrameTime = now;
    this.totalTime += dt;

    this.controls.update();
    this.updateAnimations(dt);
    this.renderer.render(this.scene, this.camera);
    this.emitProjectedAnchors();

    if (this.isSoftwareGl) {
      this.afterDraw(() => {
        if (!this.destroyed) this.tick();
      });
    } else {
      this.rafId = requestAnimationFrame(() => this.tick());
    }
  }

  private updateAnimations(dt: number): void {
    const nowMs = performance.now();
    const speedScale = 1 + Math.min(2.5, (this.simSpeedMultiplier - 1) * 0.45);
    const smoothAlpha = 1 - Math.exp(-3.2 * dt);
    const colorAlpha = 1 - Math.exp(-2.8 * dt);

    // Smoothly interpolate Sol timeline atmosphere targets frame-by-frame
    this.smoothedSolProgress = THREE.MathUtils.lerp(
      this.smoothedSolProgress,
      this.targetAtmosphere.solProgress,
      smoothAlpha
    );
    this.smoothedCrisisIntensity = THREE.MathUtils.lerp(
      this.smoothedCrisisIntensity,
      this.targetAtmosphere.crisisIntensity,
      smoothAlpha
    );
    this.smoothedEffectiveTau = THREE.MathUtils.lerp(
      this.smoothedEffectiveTau,
      this.targetAtmosphere.effectiveTau,
      smoothAlpha
    );
    this.smoothedSunElevation = THREE.MathUtils.lerp(
      this.smoothedSunElevation,
      this.targetAtmosphere.sunElevation,
      smoothAlpha
    );
    this.smoothedNightFactor = THREE.MathUtils.lerp(
      this.smoothedNightFactor,
      this.targetAtmosphere.nightFactor,
      smoothAlpha
    );

    const moduleIds: ModuleId[] = ['solar', 'moxie', 'shelter', 'biodome'];
    for (const id of moduleIds) {
      const boostLeft = Math.max(0, (this.moduleBoostTimers[id] - nowMs) / 6500);
      const targetGlow = THREE.MathUtils.clamp(
        Math.max(this.activeSystemGlow[id] ?? 0, boostLeft * 0.95),
        0,
        1
      );
      this.smoothedSystemGlow[id] = THREE.MathUtils.lerp(
        this.smoothedSystemGlow[id],
        targetGlow,
        smoothAlpha
      );
    }

    // Blinking red beacon on Comms Dish
    if (this.commsBeaconMat) {
      const blink = Math.pow(0.5 + 0.5 * Math.sin(this.totalTime * 4.5), 3);
      this.commsBeaconMat.emissiveIntensity = 0.25 + blink * 2.4;
    }

    // Ambient MOXIE turbine rotation & rising O2 bubbles
    const moxieActivity = this.smoothedSystemGlow.moxie;
    this.moxieFan.rotation.y += dt * (4.0 + moxieActivity * 4.5) * speedScale;
    const bubbleSpeed =
      (this.activePulse?.moduleId === 'moxie' ? 3.2 : 0.95 + moxieActivity * 1.15) * speedScale;
    for (let i = 0; i < this.moxieBubbles.length; i++) {
      const b = this.moxieBubbles[i];
      b.position.y += dt * bubbleSpeed;
      if (b.position.y > 4.4) b.position.y = 2.05;
      b.position.x = 1.05 + (i % 3) * 0.2 + Math.sin(this.totalTime * 3 + b.userData.phase) * 0.08;
    }

    // Gentle solar tilt & target ring pulse
    const tiltAngle =
      0.34 + (1 - this.smoothedSunElevation) * 0.16 + Math.sin(this.totalTime * 0.6) * 0.04;
    for (let i = 0; i < this.solarTiltGroups.length; i++) {
      this.solarTiltGroups[i].rotation.x = tiltAngle;
    }
    if (this.targetRing.visible) {
      const pulse = 1 + Math.sin(this.totalTime * 3.8) * 0.045;
      this.targetRing.scale.set(pulse, 1, pulse);
    }

    // Active module click feedback animation
    if (this.activePulse) {
      const elapsedSec = (nowMs - this.activePulse.startTimeMs) / 1000;
      const p = Math.min(elapsedSec / this.activePulse.duration, 1);

      if (this.activePulse.moduleId === 'solar') {
        const sweep = Math.sin(p * Math.PI * 2) * 0.88;
        for (let i = 0; i < this.solarWipers.length; i++) {
          const w = this.solarWipers[i];
          w.mesh.position.x = w.baseX + sweep;
        }
      } else if (this.activePulse.moduleId === 'shelter') {
        const ripple = Math.sin(p * Math.PI) * 0.12;
        const baseScale = 0.96 + ((this.resources.waterWallCm - 8) / 28) * 0.16;
        const s = baseScale + ripple;
        this.waterShieldMesh.scale.set(s, s, s);
      }

      if (p >= 1) {
        for (let i = 0; i < this.solarWipers.length; i++) {
          const w = this.solarWipers[i];
          w.mesh.position.x = w.baseX - 0.88;
        }
        this.activePulse = null;
        this.syncVisualState();
      }
    }

    // Subtle wind-blown Martian dust drift
    const stormStrength = THREE.MathUtils.clamp(
      (this.smoothedEffectiveTau - 0.45) * 0.85 +
        (this.targetHazardType === 'dust' ? this.smoothedCrisisIntensity * 0.75 : 0),
      0,
      1.25
    );
    const windSpeed = (1.6 + stormStrength * 10.5) * speedScale;
    const targetDustOpacity = THREE.MathUtils.clamp(
      0.16 + this.smoothedEffectiveTau * 0.22 + stormStrength * 0.34,
      0.14,
      0.72
    );
    this.dustStormMat.opacity = THREE.MathUtils.lerp(
      this.dustStormMat.opacity,
      targetDustOpacity,
      smoothAlpha
    );
    for (let i = 0; i < this.dustPositions.length; i += 3) {
      this.dustPositions[i] += dt * windSpeed;
      if (this.dustPositions[i] > 19) this.dustPositions[i] = -19;
    }
    this.dustStormPoints.geometry.attributes.position.needsUpdate = true;

    // DONKI space-weather smooth transitions (Quiet -> M-Class Amber-Violet -> X-Class/CME Red-Alert)
    const targetDonkiM =
      this.donkiTelemetry.level === 'm_class'
        ? 1
        : this.donkiTelemetry.level === 'x_class'
        ? 0.4
        : 0;
    const targetDonkiX = this.donkiTelemetry.level === 'x_class' ? 1 : 0;
    this.smoothedDonkiM = THREE.MathUtils.lerp(this.smoothedDonkiM, targetDonkiM, smoothAlpha);
    this.smoothedDonkiX = THREE.MathUtils.lerp(this.smoothedDonkiX, targetDonkiX, smoothAlpha);

    const shieldBoostActive = nowMs < this.shieldBoostUntilMs;
    const shieldStrong = this.donkiTelemetry.shieldActive || this.resources.waterWallCm >= 20 || shieldBoostActive;
    const targetShieldHex = shieldBoostActive
      ? 1.0
      : shieldStrong
      ? 0.78
      : this.smoothedDonkiM > 0.25 || this.smoothedDonkiX > 0.15
      ? 0.22
      : 0.08;
    this.smoothedShieldHex = THREE.MathUtils.lerp(
      this.smoothedShieldHex,
      targetShieldHex,
      smoothAlpha
    );
    this.shieldImpactPulse = Math.max(0, this.shieldImpactPulse - dt * 1.9);

    if (this.waterShieldHexMat) {
      this.waterShieldHexMat.uniforms.uTime.value = this.totalTime;
      this.waterShieldHexMat.uniforms.uStrength.value = this.smoothedShieldHex;
      this.waterShieldHexMat.uniforms.uImpactPulse.value = this.shieldImpactPulse;
      this.waterShieldHexMesh.visible = this.smoothedShieldHex > 0.02;
    }

    // Upper-sky auroral ribbons for M-Class and X-Class DONKI events
    const auroraVisibility = Math.max(this.smoothedDonkiM * 0.88, this.smoothedDonkiX * 1.15);
    if (this.auroraGroup) {
      this.auroraGroup.visible = auroraVisibility > 0.03;
      for (let i = 0; i < this.auroraMats.length; i++) {
        this.auroraMats[i].uniforms.uTime.value = this.totalTime;
        this.auroraMats[i].uniforms.uOpacity.value = auroraVisibility * (0.76 + 0.14 * i);
        this.auroraMats[i].uniforms.uCrimsonMix.value = this.smoothedDonkiX;
      }
    }

    // Radiation particle streaks: active during X-Class/CME impact or Sol flare crisis peak
    const flarePeakActive =
      !this.nightAnim.active &&
      (this.smoothedDonkiX > 0.12 ||
        ((this.targetHazardType === 'flare' || this.emergencyMode === 'flare') &&
          this.smoothedCrisisIntensity > 0.52));
    this.flareBoltsGroup.visible = this.nightAnim.active
      ? this.nightAnim.hazardType === 'flare'
      : flarePeakActive;

    const shelterDomeCenter = this.moduleWorldCenters.shelter;
    if (flarePeakActive) {
      const deflectEnabled = shieldStrong;
      for (let i = 0; i < this.flareBolts.length; i++) {
        const bolt = this.flareBolts[i];
        const streakSpeed = bolt.speed * (0.8 + this.smoothedDonkiX * 0.55);
        bolt.mesh.position.addScaledVector(bolt.vel, dt * streakSpeed);

        if (!bolt.deflected && !bolt.skyStreak && deflectEnabled) {
          const dx = bolt.mesh.position.x - shelterDomeCenter.x;
          const dy = bolt.mesh.position.y - (shelterDomeCenter.y + 0.45);
          const dz = bolt.mesh.position.z - shelterDomeCenter.z;
          const dist = Math.hypot(dx, dy, dz);
          if (dist <= 3.35) {
            bolt.deflected = true;
            this.streakNormScratch.set(dx, Math.max(0.25, dy), dz).normalize();
            bolt.vel.reflect(this.streakNormScratch).normalize();
            bolt.vel.y = Math.abs(bolt.vel.y) + 0.35;
            bolt.vel.normalize();
            bolt.mesh.quaternion.setFromUnitVectors(this.streakUpAxis, bolt.vel);
            bolt.mat.color.set('#67E8F9');
            bolt.mat.opacity = 1.0;
            this.waterShieldHexMat.uniforms.uImpactPoint.value.set(
              bolt.mesh.position.x,
              bolt.mesh.position.y,
              bolt.mesh.position.z
            );
            this.shieldImpactPulse = 1.0;
          }
        }

        const outOfBounds =
          bolt.mesh.position.y < (bolt.deflected ? -1 : 0.6) ||
          bolt.mesh.position.y > 22 ||
          Math.abs(bolt.mesh.position.x) > 24;
        if (outOfBounds) {
          bolt.deflected = false;
          bolt.vel.set(-0.34, -1.0, 0.16).normalize();
          bolt.mesh.quaternion.setFromUnitVectors(this.streakUpAxis, bolt.vel);
          bolt.mat.color.set(this.smoothedDonkiX > 0.35 ? '#FF4D6D' : '#FB7185');
          bolt.mat.opacity = 0.92;
          const spawnOffset = bolt.skyStreak ? 3.5 : 2.1;
          bolt.mesh.position.set(
            shelterDomeCenter.x + Math.cos(bolt.angle) * bolt.radius + spawnOffset,
            10.5 + (i % 6) * 2.0,
            shelterDomeCenter.z + Math.sin(bolt.angle) * bolt.radius - spawnOffset * 0.45
          );
        }
      }
    }

    const targetEmergency =
      this.emergencyMode !== 'none' || this.donkiTelemetry.level === 'x_class'
        ? THREE.MathUtils.clamp(
            Math.max(0.2 + this.smoothedCrisisIntensity * 0.8, this.smoothedDonkiX),
            0,
            1
          )
        : Math.max(this.smoothedCrisisIntensity * 0.25, this.smoothedDonkiM * 0.35);
    this.emergencyBlend = THREE.MathUtils.lerp(this.emergencyBlend, targetEmergency, smoothAlpha);
    const alertPulse = 0.5 + 0.5 * Math.sin(this.totalTime * 4.8);
    this.scratchAlertColor.lerpColors(this.alertRedColor, this.alertAmberColor, alertPulse);

    const nightDustBoost = this.nightAnim.active && this.nightAnim.hazardType === 'dust' ? 0.004 : 0;
    const targetFogDensity =
      this.targetAtmosphere.fogDensity + nightDustBoost + this.smoothedDonkiX * 0.0045;
    this.fog.density = THREE.MathUtils.lerp(this.fog.density, targetFogDensity, smoothAlpha);

    if (!this.nightAnim.active) {
      const hazardTintColor =
        this.emergencyMode === 'flare' || this.targetHazardType === 'flare'
          ? this.flareAlertSkyColor
          : this.dustAlertSkyColor;
      const hazardMix =
        this.emergencyMode !== 'none'
          ? THREE.MathUtils.clamp(this.smoothedCrisisIntensity * 0.85, 0, 0.9)
          : THREE.MathUtils.clamp(this.smoothedCrisisIntensity * 0.35, 0, 0.35);
      this.targetDayBaseColor.lerpColors(this.daySkyColor, hazardTintColor, hazardMix);

      // Apply DONKI M-Class amber-violet & X-Class red-alert sky shifts
      if (this.smoothedDonkiM > 0.01) {
        this.targetDayBaseColor.lerp(this.mClassHorizonColor, this.smoothedDonkiM * 0.72);
      }
      if (this.smoothedDonkiX > 0.01) {
        this.targetDayBaseColor.lerp(this.xClassHorizonColor, this.smoothedDonkiX * 0.88);
      }

      if (this.smoothedNightFactor < 0.45) {
        this.targetSkyColor.lerpColors(
          this.targetDayBaseColor,
          this.sunsetSkyColor,
          this.smoothedNightFactor / 0.45
        );
      } else {
        this.targetSkyColor.lerpColors(
          this.sunsetSkyColor,
          this.nightSkyColor,
          (this.smoothedNightFactor - 0.45) / 0.55
        );
      }
      if (this.targetHazardType === 'flare' && this.smoothedCrisisIntensity > 0.55) {
        const auroraMix = (this.smoothedCrisisIntensity - 0.55) * 0.65;
        this.targetSkyColor.lerp(this.flareSkyColor, auroraMix);
      }

      this.skyColor.lerp(this.targetSkyColor, colorAlpha);
      (this.scene.background as THREE.Color).copy(this.skyColor);
      this.fog.color.copy(this.skyColor);
      this.renderer.setClearColor(this.fog.color);

      if (this.skyDomeMat) {
        this.skyDomeMat.uniforms.uHorizonColor.value.copy(this.skyColor);
        this.scratchTargetZenith.copy(this.zenithDayColor);
        if (this.smoothedDonkiM > 0.01) {
          this.scratchTargetZenith.lerp(this.mClassZenithColor, this.smoothedDonkiM * 0.82);
        }
        if (this.smoothedDonkiX > 0.01) {
          this.scratchTargetZenith.lerp(this.xClassZenithColor, this.smoothedDonkiX * 0.92);
        }
        this.zenithColor.lerpColors(
          this.scratchTargetZenith,
          this.zenithNightColor,
          THREE.MathUtils.clamp(this.smoothedNightFactor * 1.15, 0, 1)
        );
        this.skyDomeMat.uniforms.uZenithColor.value.copy(this.zenithColor);
      }

      const sunAngle = this.smoothedSolProgress * Math.PI;
      this.targetSunPos.set(
        22 * Math.cos(sunAngle),
        Math.max(2.6, 19 * this.smoothedSunElevation),
        14
      );
      this.sunLight.position.lerp(this.targetSunPos, smoothAlpha);

      this.targetSunColor.lerpColors(
        this.calmSunColor,
        this.alertSunColor,
        THREE.MathUtils.clamp(this.emergencyBlend * 0.7 + this.smoothedNightFactor * 0.5, 0, 1)
      );
      if (this.smoothedDonkiX > 0.01) {
        this.targetSunColor.lerp(this.xClassSunColor, this.smoothedDonkiX * 0.75);
      }
      this.sunLight.color.lerp(this.targetSunColor, colorAlpha);

      const targetSunIntensity = THREE.MathUtils.lerp(
        2.85,
        0.55,
        THREE.MathUtils.clamp(this.smoothedNightFactor * 0.85 + this.emergencyBlend * 0.45, 0, 1)
      );
      this.sunLight.intensity = THREE.MathUtils.lerp(
        this.sunLight.intensity,
        targetSunIntensity,
        smoothAlpha
      );

      // Pulsing crimson rim-lighting during X-Class / CME Impact
      if (this.rimLight) {
        const rimPulse = 0.65 + 0.35 * Math.sin(this.totalTime * 5.2);
        this.rimLight.intensity = this.smoothedDonkiX * 4.4 * rimPulse;
      }

      this.hemiLight.color.lerpColors(this.calmHemiSky, this.xClassHemiSky, this.smoothedDonkiX * 0.7);
      this.hemiLight.groundColor.lerpColors(
        this.calmHemiGround,
        this.xClassHemiGround,
        this.smoothedDonkiX * 0.85
      );
      const targetHemiIntensity = THREE.MathUtils.lerp(
        0.62,
        0.26,
        THREE.MathUtils.clamp(this.smoothedNightFactor * 0.8 + this.emergencyBlend * 0.45, 0, 1)
      );
      this.hemiLight.intensity = THREE.MathUtils.lerp(
        this.hemiLight.intensity,
        targetHemiIntensity,
        smoothAlpha
      );

      const targetEnvIntensity = THREE.MathUtils.lerp(
        0.85,
        0.28,
        THREE.MathUtils.clamp(this.emergencyBlend * 0.75 + this.smoothedNightFactor * 0.5, 0, 1)
      );
      this.scene.environmentIntensity = THREE.MathUtils.lerp(
        this.scene.environmentIntensity,
        targetEnvIntensity,
        smoothAlpha
      );

      const targetStarsOpacity = THREE.MathUtils.clamp(
        Math.max(
          (this.smoothedNightFactor - 0.3) * 1.4,
          this.smoothedDonkiM * 0.35,
          this.smoothedDonkiX * 0.6
        ),
        0,
        0.95
      );
      this.starsMat.opacity = THREE.MathUtils.lerp(this.starsMat.opacity, targetStarsOpacity, smoothAlpha);

      // Active Base Glow + X-Class electrical interior flicker
      const solarPulse = 0.5 + 0.5 * Math.sin(this.totalTime * 2.8);
      const moxiePulse = 0.5 + 0.5 * Math.sin(this.totalTime * 3.4 + 1.1);
      const shelterPulse = 0.5 + 0.5 * Math.sin(this.totalTime * 3.0 + 2.2);
      const bioPulse = 0.5 + 0.5 * Math.sin(this.totalTime * 2.6 + 3.3);
      const xFlicker =
        1.0 -
        this.smoothedDonkiX *
          0.62 *
          Math.max(0, Math.sin(this.totalTime * 23.0) * Math.cos(this.totalTime * 13.7));

      const solarGlow = this.smoothedSystemGlow.solar;
      const moxieGlow = this.smoothedSystemGlow.moxie;
      const shelterGlow = this.smoothedSystemGlow.shelter;
      const bioGlow = this.smoothedSystemGlow.biodome;

      const solarActionBoost = this.activePulse?.moduleId === 'solar' ? 0.55 : 0;
      this.solarCellMat.emissiveIntensity =
        0.25 + solarGlow * (0.35 + solarPulse * 0.42) + solarActionBoost;
      this.solarBatteryMat.emissiveIntensity =
        0.42 + solarGlow * (0.45 + solarPulse * 0.5) + solarActionBoost;

      const wallRatio = THREE.MathUtils.clamp((this.resources.waterWallCm - 8) / 28, 0, 1);
      const shelterActionBoost = this.activePulse?.moduleId === 'shelter' ? 0.85 : 0;
      this.waterShieldMat.emissiveIntensity =
        0.3 +
        wallRatio * 0.65 +
        shelterGlow * (0.28 + shelterPulse * 0.48) +
        shelterActionBoost +
        this.shieldImpactPulse * 0.7;

      this.shelterInteriorLight.color.lerpColors(
        this.calmWindowEmissive,
        this.scratchAlertColor,
        this.emergencyBlend * 0.85
      );
      this.shelterInteriorLight.intensity =
        (1.05 +
          this.smoothedNightFactor * 1.45 +
          shelterGlow * (0.4 + shelterPulse * 0.45) +
          this.emergencyBlend * (2.6 + alertPulse * 1.8)) *
        xFlicker;

      this.shelterHullMat.color.lerpColors(
        this.calmHullColor,
        this.alertHullTint,
        this.emergencyBlend * 0.5
      );
      this.shelterHullMat.emissive.lerpColors(
        this.blackColor,
        this.scratchAlertColor,
        Math.max(this.emergencyBlend * 0.65, this.smoothedDonkiX * 0.75)
      );
      this.shelterHullMat.emissiveIntensity = Math.max(
        this.emergencyBlend * (0.32 + alertPulse * 0.34),
        this.smoothedDonkiX * (0.28 + alertPulse * 0.36)
      );

      this.shelterWindowMat.color.lerpColors(
        this.calmWindowColor,
        this.scratchAlertColor,
        this.emergencyBlend * 0.85
      );
      this.shelterWindowMat.emissive.lerpColors(
        this.calmWindowEmissive,
        this.alertRedColor,
        this.emergencyBlend * 0.85
      );
      this.shelterWindowMat.emissiveIntensity =
        (0.85 +
          this.smoothedNightFactor * 0.85 +
          shelterGlow * (0.2 + shelterPulse * 0.25) +
          this.emergencyBlend * (0.3 + alertPulse * 0.45)) *
        xFlicker;

      this.shelterBeaconMat.color.lerpColors(
        this.calmBeaconColor,
        this.alertRedColor,
        this.emergencyBlend
      );
      this.shelterBeaconMat.emissive.lerpColors(
        this.calmBeaconEmissive,
        this.alertRedColor,
        this.emergencyBlend
      );
      this.shelterBeaconMat.emissiveIntensity =
        1.15 + shelterGlow * 0.35 + this.emergencyBlend * (0.4 + alertPulse * 0.7);

      const o2Ratio = THREE.MathUtils.clamp(this.resources.lifeSupportPct / 100, 0.15, 1);
      const moxieActionBoost = this.activePulse?.moduleId === 'moxie' ? 0.9 : 0;
      this.moxieLight.color.lerpColors(
        this.calmMoxieEmissive,
        this.scratchAlertColor,
        this.emergencyBlend * 0.8
      );
      this.moxieLight.intensity =
        (1.05 + moxieGlow * (0.55 + moxiePulse * 0.55) + this.emergencyBlend * (2.4 + alertPulse * 1.6)) *
        (0.8 + 0.2 * xFlicker);

      this.moxieTankMat.color.lerpColors(
        this.calmMoxieColor,
        this.scratchAlertColor,
        this.emergencyBlend * 0.78
      );
      this.moxieTankMat.emissive.lerpColors(
        this.calmMoxieEmissive,
        this.scratchAlertColor,
        this.emergencyBlend * 0.82
      );
      this.moxieTankMat.emissiveIntensity =
        0.38 + o2Ratio * 0.55 + moxieGlow * (0.35 + moxiePulse * 0.52) + moxieActionBoost;

      const bioActionBoost = this.activePulse?.moduleId === 'biodome' ? 0.85 : 0;
      this.biodomeLight.color.lerpColors(
        this.calmBioEmissive,
        this.scratchAlertColor,
        this.emergencyBlend * 0.78
      );
      this.biodomeLight.intensity =
        2.55 +
        bioGlow * (0.85 + bioPulse * 0.7) +
        bioActionBoost * 1.45 +
        this.emergencyBlend * (2.3 + alertPulse * 1.6);

      this.biodomeGlassMat.color.lerpColors(
        this.calmBioColor,
        this.scratchAlertColor,
        this.emergencyBlend * 0.75
      );
      this.biodomeGlassMat.emissive.lerpColors(
        this.calmBioEmissive,
        this.scratchAlertColor,
        this.emergencyBlend * 0.82
      );
      this.biodomeGlassMat.emissiveIntensity =
        0.24 +
        bioGlow * (0.28 + bioPulse * 0.36) +
        bioActionBoost * 0.65 +
        this.emergencyBlend * (0.32 + alertPulse * 0.3);

      this.bioLedRingMat.color.lerpColors(
        this.calmLedColor,
        this.alertRedColor,
        this.emergencyBlend * 0.85
      );
      this.bioLedRingMat.emissive.lerpColors(
        this.calmLedEmissive,
        this.alertRedColor,
        this.emergencyBlend * 0.85
      );
      this.bioLedRingMat.emissiveIntensity =
        0.85 +
        bioGlow * (0.35 + bioPulse * 0.45) +
        bioActionBoost * 0.5 +
        this.emergencyBlend * (0.3 + alertPulse * 0.45);

      this.bioLeafMat.emissiveIntensity =
        0.42 + bioGlow * (0.28 + bioPulse * 0.35) + bioActionBoost * 0.4;
    }

    // Legacy Night Cycle Animation helper (if invoked directly)
    if (this.nightAnim.active) {
      const elapsedSec = (performance.now() - this.nightAnim.startTimeMs) / 1000;
      const t = Math.min(elapsedSec / this.nightAnim.duration, 1);
      const nightFactor = Math.sin(t * Math.PI);

      const sunAngle = t * Math.PI;
      this.sunLight.position.set(
        22 * Math.cos(sunAngle),
        Math.max(1.5, 19 * (1 - nightFactor * 0.92)),
        14
      );
      this.sunLight.intensity = THREE.MathUtils.lerp(2.85, 0.35, nightFactor);
      this.hemiLight.intensity = THREE.MathUtils.lerp(0.62, 0.24, nightFactor);

      if (nightFactor < 0.45) {
        this.skyColor.lerpColors(this.daySkyColor, this.sunsetSkyColor, nightFactor / 0.45);
      } else {
        this.skyColor.lerpColors(this.sunsetSkyColor, this.nightSkyColor, (nightFactor - 0.45) / 0.55);
      }
      if (this.nightAnim.hazardType === 'flare' && nightFactor > 0.4) {
        const auroraPulse = 0.35 + 0.25 * Math.sin(this.totalTime * 12);
        this.skyColor.lerp(this.flareSkyColor, auroraPulse);
      }
      (this.scene.background as THREE.Color).copy(this.skyColor);
      this.fog.color.copy(this.skyColor);
      this.renderer.setClearColor(this.fog.color);

      this.starsMat.opacity = Math.max(0, (nightFactor - 0.35) * 1.45);
      this.shelterInteriorLight.intensity = 0.85 + nightFactor * 2.8;
      this.shelterWindowMat.emissiveIntensity = 0.85 + nightFactor * 1.9;
      this.shelterBeaconMat.emissiveIntensity = 1.2 + Math.sin(this.totalTime * 10) * 0.9;

      if (this.nightAnim.hazardType === 'flare') {
        for (const bolt of this.flareBolts) {
          if (!bolt.deflected) {
            bolt.mesh.position.y -= dt * bolt.speed;
            const impactY = this.nightAnim.shieldBlockedFlare ? 3.2 : 1.6;
            if (bolt.mesh.position.y <= impactY) {
              if (this.nightAnim.shieldBlockedFlare) {
                bolt.deflected = true;
                bolt.mesh.rotation.z = 0.9;
              } else {
                bolt.mesh.position.y = 8.8;
                this.waterShieldMat.emissive.set('#EF4444');
              }
            }
          } else {
            bolt.mesh.position.y += dt * 7.5;
            bolt.mesh.position.x += Math.cos(bolt.angle) * dt * 6;
            bolt.mesh.position.z += Math.sin(bolt.angle) * dt * 6;
            if (bolt.mesh.position.y > 9.5) {
              bolt.deflected = false;
              bolt.mesh.rotation.set(0, 0, 0);
              bolt.mesh.position.set(
                Math.cos(bolt.angle) * bolt.radius,
                8.5,
                Math.sin(bolt.angle) * bolt.radius
              );
            }
          }
        }
      }

      if (t >= 1 && !this.nightAnim.completed) {
        this.nightAnim.completed = true;
        this.nightAnim.active = false;
        this.flareBoltsGroup.visible = false;
        this.waterShieldMat.emissive.set('#0284C7');
        this.sunLight.position.set(22, 19, 14);
        this.sunLight.intensity = 2.85;
        this.hemiLight.intensity = 0.62;
        this.starsMat.opacity = 0;
        this.shelterInteriorLight.intensity = 0.85;
        this.shelterWindowMat.emissiveIntensity = 0.85;
        (this.scene.background as THREE.Color).copy(this.daySkyColor);
        this.fog.color.copy(this.daySkyColor);
        this.renderer.setClearColor(this.fog.color);
        this.syncVisualState();
        this.nightAnim.onComplete();
      }
    }
  }

  private emitProjectedAnchors(): void {
    const width = this.renderer.domElement.clientWidth || window.innerWidth;
    const height = this.renderer.domElement.clientHeight || window.innerHeight;
    const moduleIds: ModuleId[] = ['solar', 'moxie', 'shelter', 'biodome'];

    const next: ProjectedModuleAnchor[] = moduleIds.map((id) => {
      this.projScratch.copy(this.anchorWorldMap[id]).project(this.camera);
      const visible =
        this.projScratch.z >= -1 &&
        this.projScratch.z <= 1 &&
        this.projScratch.x >= -0.92 &&
        this.projScratch.x <= 0.92 &&
        this.projScratch.y >= -0.88 &&
        this.projScratch.y <= 0.92;

      return {
        id,
        x: Math.round(((this.projScratch.x + 1) * 0.5) * width),
        y: Math.round(((-this.projScratch.y + 1) * 0.5) * height),
        visible,
      };
    });

    const changed =
      this.lastAnchors.length !== next.length ||
      next.some((item, idx) => {
        const prev = this.lastAnchors[idx];
        return (
          prev.visible !== item.visible ||
          Math.abs(prev.x - item.x) >= 1.5 ||
          Math.abs(prev.y - item.y) >= 1.5
        );
      });

    if (changed) {
      this.lastAnchors = next;
      this.callbacks.onAnchorsUpdate(next);
    }
  }

  public dispose(): void {
    this.destroyed = true;
    if (this.rafId !== null) cancelAnimationFrame(this.rafId);
    if (this.swTimerId !== null) clearTimeout(this.swTimerId);

    const dom = this.renderer?.domElement;
    if (dom) {
      dom.removeEventListener('pointerdown', this.handlePointerDown);
      dom.removeEventListener('pointerup', this.handlePointerUp);
      dom.removeEventListener('pointermove', this.handlePointerMove);
      dom.removeEventListener('webglcontextlost', this.handleContextLost);
      dom.remove();
    }
    window.removeEventListener('resize', this.handleResize);

    this.controls?.dispose();
    this.envRenderTarget?.dispose();
    this.pmremGenerator?.dispose();
    for (const item of this.disposables) {
      item.dispose();
    }
    this.disposables = [];
    this.renderer?.dispose();
  }
}
