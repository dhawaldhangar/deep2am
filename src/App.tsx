import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import {
  AlertTriangle,
  Clock,
  Database,
  FastForward,
  Lock,
  Moon,
  Radio,
  RotateCcw,
  Shield,
  Sparkles,
  Sun,
  Volume2,
  VolumeX,
  Wind,
  Zap,
} from 'lucide-react';
import { DilemmaChoice, ScienceTerm, VERIFIED_FALLBACK_FLARES } from './data/nasaTelemetry';
import {
  computeUnlockedLogIds,
  ModuleId,
  ModuleUsageCounts,
} from './data/campaignArc';
import {
  advanceOperationTick,
  advanceToNextSol,
  applyModuleActionToMission,
  computeSolAtmosphereTargets,
  createInitialMissionState,
  FAST_FORWARD_MULTIPLIER,
  FAST_FORWARD_UNLOCK_SEC,
  getSolChapter,
  lockInDecisionChoice,
  MAX_BATTERY_KWH,
  MAX_SOLS,
  MissionState,
  openDecisionCards,
  SOL_DURATION_SEC,
  toggleFastForward,
} from './game/simulation';
import {
  computeDonkiVisualState,
  DEFAULT_ARCHIVE_CME,
  DonkiFetchResult,
  fetchNasaDonkiTelemetry,
} from './utils/nasaApi';
import { missionAudio } from './utils/audio';
import { ActiveSystemGlow, DonkiSceneTelemetry, EmergencyMode } from './three/MarsOutpostScene';
import { TitleScreen } from './components/TitleScreen';
import { MarsViewport, MarsViewportHandle, ModulePopup } from './components/MarsViewport';
import { CharacterPortrait } from './components/CharacterPortrait';
import { ScienceTermButton } from './components/ScienceTermButton';
import { EarthDispatchDrawer } from './components/EarthDispatchDrawer';
import { DonkiInspectorCard } from './components/DonkiInspectorCard';
import {
  DebriefModal,
  DilemmaModal,
  ScienceDecoderModal,
  ScienceLogModal,
} from './components/Modals';

type ScreenPhase = 'title' | 'playing' | 'gameover';

interface HudDeltas {
  id: number;
  power: number;
  life: number;
  health: number;
}

const INITIAL_USAGE: ModuleUsageCounts = {
  solar: 0,
  moxie: 0,
  shelter: 0,
  biodome: 0,
};

function formatMmSs(totalSeconds: number): string {
  const clamped = Math.max(0, Math.ceil(totalSeconds));
  const mins = Math.floor(clamped / 60);
  const secs = clamped % 60;
  return `${String(mins).padStart(2, '0')}:${String(secs).padStart(2, '0')}`;
}

function formatRate(perSec: number, unit: string): string {
  const rounded = Math.abs(perSec) < 0.005 ? 0 : perSec;
  const sign = rounded > 0 ? '+' : '';
  return `${sign}${rounded.toFixed(2)} ${unit}/s`;
}

export default function App() {
  const [phase, setPhase] = useState<ScreenPhase>(() => {
    if (typeof window !== 'undefined') {
      const params = new URLSearchParams(window.location.search);
      if (params.get('autostart') === '1') return 'playing';
    }
    return 'title';
  });

  const [donkiResult, setDonkiResult] = useState<DonkiFetchResult>({
    flares: VERIFIED_FALLBACK_FLARES,
    cme: DEFAULT_ARCHIVE_CME,
    mode: 'VERIFIED_ARCHIVE_FALLBACK',
    fetchedAt: 'Archive Ready',
    statusNote: 'Initialized with verified NASA CCMC DONKI & Perseverance MEDA telemetry.',
  });

  const [mission, setMission] = useState<MissionState>(() =>
    createInitialMissionState(VERIFIED_FALLBACK_FLARES)
  );
  const [moduleUsage, setModuleUsage] = useState<ModuleUsageCounts>(INITIAL_USAGE);
  const [isNightCycle, setIsNightCycle] = useState(false);
  const [muted, setMuted] = useState(false);
  const [showScienceLog, setShowScienceLog] = useState(false);
  const [donkiFeedOpen, setDonkiFeedOpen] = useState(false);
  const [simulateXFlare, setSimulateXFlare] = useState(false);
  const [shieldBoostActive, setShieldBoostActive] = useState(false);
  const [dispatchOpen, setDispatchOpen] = useState(false);
  const [decryptedSols, setDecryptedSols] = useState<number[]>([]);
  const [activeTerm, setActiveTerm] = useState<ScienceTerm | null>(null);
  const [activePopup, setActivePopup] = useState<ModulePopup | null>(null);
  const [hudDeltas, setHudDeltas] = useState<HudDeltas | null>(null);

  const viewportRef = useRef<MarsViewportHandle | null>(null);
  const popupTimerRef = useRef<number | null>(null);
  const deltaTimerRef = useRef<number | null>(null);
  const shieldTimerRef = useRef<number | null>(null);
  const prevOpPhaseRef = useRef(mission.opPhase);

  const clearTimers = useCallback(() => {
    if (popupTimerRef.current !== null) {
      window.clearTimeout(popupTimerRef.current);
      popupTimerRef.current = null;
    }
    if (deltaTimerRef.current !== null) {
      window.clearTimeout(deltaTimerRef.current);
      deltaTimerRef.current = null;
    }
    if (shieldTimerRef.current !== null) {
      window.clearTimeout(shieldTimerRef.current);
      shieldTimerRef.current = null;
    }
  }, []);

  useEffect(() => clearTimers, [clearTimers]);

  useEffect(() => {
    document.body.dataset.state = mission.debrief ? 'gameover' : phase;
  }, [phase, mission.debrief]);

  useEffect(() => {
    let active = true;
    fetchNasaDonkiTelemetry().then((res) => {
      if (!active) return;
      setDonkiResult(res);
    });
    return () => {
      active = false;
    };
  }, []);

  const chapter = useMemo(() => getSolChapter(mission.sol), [mission.sol]);

  const donkiVisualState = useMemo(
    () => computeDonkiVisualState(mission.activeFlare, donkiResult.cme, simulateXFlare),
    [mission.activeFlare, donkiResult.cme, simulateXFlare]
  );

  const shieldActive = mission.resources.waterWallCm >= 20 || shieldBoostActive;

  const donkiTelemetry: DonkiSceneTelemetry = useMemo(
    () => ({
      level: donkiVisualState.level,
      intensity: donkiVisualState.intensity,
      cmeActive: donkiVisualState.cme.impactActive,
      shieldActive,
    }),
    [donkiVisualState, shieldActive]
  );

  const emergencyMode: EmergencyMode = useMemo(() => {
    const severeSolarFlare =
      chapter.hazardType === 'flare' ||
      (mission.sol >= 4 && /^[MX]/i.test(mission.activeFlare.classType));
    if (severeSolarFlare) return 'flare';
    if (chapter.hazardType === 'dust' || mission.weather.tau >= 0.85) return 'dust';
    return 'none';
  }, [chapter.hazardType, mission.sol, mission.activeFlare.classType, mission.weather.tau]);

  const unlockedLogIds = useMemo(
    () => computeUnlockedLogIds(mission.sol, moduleUsage),
    [mission.sol, moduleUsage]
  );

  const atmosphereTargets = useMemo(
    () =>
      computeSolAtmosphereTargets({
        elapsedSec: mission.operationElapsedSec,
        durationSec: mission.operationDurationSec,
        decisionElapsedSec: mission.decisionElapsedSec,
        baseTau: mission.weather.tau,
        hazardType: chapter.hazardType,
        emergencyMode,
      }),
    [
      mission.operationElapsedSec,
      mission.operationDurationSec,
      mission.decisionElapsedSec,
      mission.weather.tau,
      chapter.hazardType,
      emergencyMode,
    ]
  );

  const activeSystemGlow: ActiveSystemGlow = useMemo(() => {
    const isWorking = mission.opPhase === 'operation' || mission.opPhase === 'sol_complete';
    const rates = mission.operationRates;
    const choiceId = mission.lockedChoice?.id ?? '';

    const solarActive = isWorking
      ? rates && rates.powerPerSec > 0
        ? 0.85
        : mission.resources.panelDustPct < 60
        ? 0.55
        : 0.35
      : 0.22;
    const moxieActive = isWorking
      ? choiceId.includes('moxie') || (rates && rates.lifePerSec > -0.025)
        ? 0.88
        : 0.62
      : 0.25;
    const shelterActive = isWorking
      ? emergencyMode === 'flare' || mission.resources.waterWallCm >= 22
        ? 0.86
        : 0.5
      : 0.2;
    const biodomeActive = isWorking
      ? Math.min(0.9, 0.38 + mission.resources.cropGrowth * 0.1)
      : 0.22;

    return {
      solar: solarActive,
      moxie: moxieActive,
      shelter: shelterActive,
      biodome: biodomeActive,
    };
  }, [
    mission.opPhase,
    mission.operationRates,
    mission.lockedChoice,
    mission.resources.panelDustPct,
    mission.resources.waterWallCm,
    mission.resources.cropGrowth,
    emergencyMode,
  ]);

  // Drive synthesized Martian ambient wind soundscape based on mission state
  useEffect(() => {
    missionAudio.setAmbientCondition({
      active: phase === 'playing' && !mission.debrief,
      isDustStorm: emergencyMode === 'dust' || atmosphereTargets.effectiveTau >= 1.0,
      isEmergency: emergencyMode !== 'none',
    });
  }, [phase, mission.debrief, emergencyMode, atmosphereTargets.effectiveTau]);

  // Continuous 5-minute (300s) Sol timeline loop: runs smoothly from 0:00 across Phase 1 & Phase 2
  useEffect(() => {
    if (
      phase !== 'playing' ||
      mission.opPhase === 'sol_complete' ||
      mission.debrief ||
      isNightCycle
    ) {
      return;
    }

    let lastTime = performance.now();
    const intervalId = window.setInterval(() => {
      const now = performance.now();
      const dtSec = Math.min((now - lastTime) / 1000, 1.0);
      lastTime = now;
      if (dtSec <= 0) return;

      setMission((prev) => {
        const next = advanceOperationTick(prev, dtSec);
        if (!prev.debrief && next.debrief) {
          missionAudio.playMissionComplete(false);
        }
        return next;
      });
    }, 250);

    return () => window.clearInterval(intervalId);
  }, [phase, mission.opPhase, mission.debrief, isNightCycle]);

  // Play chime when 300s Sol finishes and End Sol / Next Operation unlocks
  useEffect(() => {
    if (prevOpPhaseRef.current === 'operation' && mission.opPhase === 'sol_complete') {
      missionAudio.playTelemetryChime();
    }
    prevOpPhaseRef.current = mission.opPhase;
  }, [mission.opPhase]);

  const showModulePopup = useCallback((moduleId: ModuleId, text: string, positive: boolean) => {
    if (popupTimerRef.current !== null) {
      window.clearTimeout(popupTimerRef.current);
    }
    setActivePopup({
      id: Date.now() + Math.random(),
      moduleId,
      text,
      positive,
    });
    popupTimerRef.current = window.setTimeout(() => {
      setActivePopup(null);
      popupTimerRef.current = null;
    }, 3200);
  }, []);

  const showHudDeltaBadges = useCallback((power: number, life: number, health: number) => {
    if (deltaTimerRef.current !== null) {
      window.clearTimeout(deltaTimerRef.current);
    }
    setHudDeltas({
      id: Date.now() + Math.random(),
      power,
      life,
      health,
    });
    deltaTimerRef.current = window.setTimeout(() => {
      setHudDeltas(null);
      deltaTimerRef.current = null;
    }, 3200);
  }, []);

  const startMission = useCallback(() => {
    clearTimers();
    setMission(createInitialMissionState(donkiResult.flares));
    setModuleUsage(INITIAL_USAGE);
    setDecryptedSols([]);
    setDispatchOpen(false);
    setDonkiFeedOpen(false);
    setShieldBoostActive(false);
    setIsNightCycle(false);
    setActivePopup(null);
    setHudDeltas(null);
    setPhase('playing');
    setShowScienceLog(false);
    setActiveTerm(null);
  }, [clearTimers, donkiResult.flares]);

  useEffect(() => {
    const onKeyDown = (e: KeyboardEvent) => {
      if (phase === 'title' && (e.key === 'Enter' || e.key === ' ')) {
        e.preventDefault();
        missionAudio.playSolAdvance();
        startMission();
        return;
      }
      if (e.key === 'Escape') {
        if (activeTerm) setActiveTerm(null);
        else if (showScienceLog) setShowScienceLog(false);
        else if (donkiFeedOpen) setDonkiFeedOpen(false);
        else if (dispatchOpen) setDispatchOpen(false);
      }
    };
    window.addEventListener('keydown', onKeyDown);
    return () => window.removeEventListener('keydown', onKeyDown);
  }, [phase, activeTerm, showScienceLog, donkiFeedOpen, dispatchOpen, startMission]);

  const handleModuleAction = useCallback(
    (moduleId: ModuleId) => {
      if (
        isNightCycle ||
        mission.debrief ||
        (mission.opPhase !== 'operation' && mission.opPhase !== 'sol_complete')
      ) {
        return;
      }

      const { nextState, result } = applyModuleActionToMission(mission, moduleId);
      if (!result.applied) {
        missionAudio.playWarningKlaxon();
        showModulePopup(moduleId, result.feedbackText, false);
        return;
      }

      if (moduleId === 'moxie') missionAudio.playMoxieTurbine();
      else if (moduleId === 'solar') missionAudio.playSolarServoSweep();
      else if (moduleId === 'shelter') {
        missionAudio.playShieldSurge();
        setShieldBoostActive(true);
        if (shieldTimerRef.current !== null) {
          window.clearTimeout(shieldTimerRef.current);
        }
        shieldTimerRef.current = window.setTimeout(() => {
          setShieldBoostActive(false);
          shieldTimerRef.current = null;
        }, 11000);
      } else missionAudio.playBiodomeMist();

      viewportRef.current?.triggerModulePulse(moduleId);
      setMission(nextState);
      setModuleUsage((prev) => ({ ...prev, [moduleId]: prev[moduleId] + 1 }));
      showModulePopup(moduleId, result.feedbackText, true);
      showHudDeltaBadges(result.deltaPower, result.deltaLife, result.deltaHealth);
    },
    [isNightCycle, mission, showModulePopup, showHudDeltaBadges]
  );

  const handleOpenDecisionCards = () => {
    if (isNightCycle || mission.debrief || mission.opPhase !== 'crisis_brief') return;
    missionAudio.playClick();
    setMission((prev) => openDecisionCards(prev));
  };

  const handleChooseDilemma = (choice: DilemmaChoice) => {
    const next = lockInDecisionChoice(mission, choice);
    setMission(next);
    if (next.debrief) {
      missionAudio.playMissionComplete(next.debrief.outcome === 'VICTORY');
      return;
    }
    showHudDeltaBadges(
      choice.effects.powerKwh,
      choice.effects.lifeSupportPct,
      choice.effects.crewHealthPct
    );
    if (chapter.hazardType !== 'none' || choice.effects.dustDelta > 0) {
      missionAudio.playHazardImpact();
    } else {
      missionAudio.playTelemetryChime();
    }
  };

  const handleToggleFastForward = () => {
    if (!mission.fastForwardUnlocked || mission.debrief || mission.opPhase !== 'operation') return;
    missionAudio.playClick();
    setMission((prev) => toggleFastForward(prev));
  };

  const handleAdvanceNextSol = () => {
    if (isNightCycle || mission.debrief || mission.opPhase !== 'sol_complete') return;
    setDispatchOpen(false);
    if (chapter.hazardType !== 'none') {
      missionAudio.playHazardImpact();
    } else {
      missionAudio.playSolAdvance();
    }

    setMission((prev) => {
      const next = advanceToNextSol(prev, donkiResult.flares);
      if (next.debrief) {
        missionAudio.playMissionComplete(next.debrief.outcome === 'VICTORY');
      } else if (
        next.resources.powerKwh < 30 ||
        next.resources.lifeSupportPct < 35 ||
        next.resources.crewHealthPct < 35
      ) {
        missionAudio.playWarningKlaxon();
      }
      return next;
    });
  };

  const handleMarkDecrypted = useCallback((solNum: number) => {
    setDecryptedSols((prev) => (prev.includes(solNum) ? prev : [...prev, solNum]));
  }, []);

  const toggleMute = () => {
    const nextMuted = !muted;
    setMuted(nextMuted);
    missionAudio.setMuted(nextMuted);
    if (!nextMuted) missionAudio.playClick();
  };

  if (phase === 'title') {
    return <TitleScreen onStart={startMission} nasaStatusNote={donkiResult.statusNote} />;
  }

  const powerPct = Math.round((mission.exactResources.powerKwh / MAX_BATTERY_KWH) * 100);
  const lifePct = Math.round(mission.exactResources.lifeSupportPct);
  const healthPct = Math.round(mission.exactResources.crewHealthPct);
  const isDecisionModalOpen = mission.opPhase === 'decision_cards' && Boolean(mission.pendingDilemma);
  const modalOpen = Boolean(
    isDecisionModalOpen || mission.debrief || showScienceLog || activeTerm
  );
  const isEmergency = emergencyMode !== 'none' || donkiVisualState.level === 'x_class';
  const emBorderClass =
    donkiVisualState.level === 'x_class'
      ? 'hud-em-static-x'
      : donkiVisualState.level === 'm_class'
      ? 'hud-em-static-m'
      : '';
  const hasUnreadDispatch = !decryptedSols.includes(mission.sol);

  const isPhase1 = mission.opPhase === 'crisis_brief' || mission.opPhase === 'decision_cards';
  const isPhase2Or3 = mission.opPhase === 'operation' || mission.opPhase === 'sol_complete';
  const remainingSec = Math.max(0, mission.operationDurationSec - mission.operationElapsedSec);
  const fastForwardWaitSec = Math.max(0, FAST_FORWARD_UNLOCK_SEC - mission.operationElapsedSec);
  const opProgressPct = Math.min(
    100,
    Math.round((mission.operationElapsedSec / mission.operationDurationSec) * 100)
  );

  const phaseLabel =
    mission.opPhase === 'crisis_brief'
      ? 'Phase 1 · Early Shift'
      : mission.opPhase === 'decision_cards'
      ? 'Phase 1 · Tactical Cards'
      : mission.opPhase === 'operation'
      ? mission.fastForwardActive
        ? `Phase 2 · ${FAST_FORWARD_MULTIPLIER}x Fast Forward`
        : 'Phase 2 · Live Operation'
      : 'Phase 3 · Ready to Advance';

  return (
    <div
      data-emergency={emergencyMode}
      data-donki-level={donkiVisualState.level}
      data-op-phase={mission.opPhase}
      className="relative w-screen h-screen overflow-hidden bg-[#070B14] text-slate-100 select-none"
    >
      {/* Full-Screen 3D Mars Outpost */}
      <MarsViewport
        ref={viewportRef}
        resources={mission.resources}
        tau={atmosphereTargets.effectiveTau}
        targetModule={chapter.targetModule}
        emergencyMode={emergencyMode}
        atmosphere={atmosphereTargets}
        hazardType={chapter.hazardType}
        activeSystemGlow={activeSystemGlow}
        donkiTelemetry={donkiTelemetry}
        speedMultiplier={mission.speedMultiplier}
        isNightCycle={isNightCycle}
        disabled={modalOpen || isPhase1}
        activePopup={activePopup}
        onModuleAction={handleModuleAction}
      />

      {/* Minimal Translucent Top HUD Bar */}
      <header className="relative z-20 pointer-events-none px-2.5 sm:px-5 pt-2.5">
        <div
          data-testid="top-hud-bar"
          className={`pointer-events-auto max-w-7xl mx-auto rounded-2xl px-3 sm:px-4 py-2.5 shadow-xl flex flex-col gap-2 border transition-colors ${emBorderClass} ${
            isEmergency
              ? 'bg-[#19090E]/92 border-rose-500/80 shadow-[0_0_24px_rgba(239,68,68,0.24)]'
              : donkiVisualState.level === 'm_class'
              ? 'bg-[#100918]/92 border-fuchsia-400/55 shadow-[0_0_20px_rgba(192,132,252,0.16)]'
              : 'bg-[#070B14]/90 border-slate-700/80'
          }`}
        >
          {/* Row 1: Sol, Operational Phase, Emergency Banner, MEDA Weather & Utility Buttons */}
          <div className="flex flex-wrap items-center justify-between gap-2">
            <div className="flex flex-wrap items-center gap-x-2 gap-y-1 min-w-0">
              <h1 className="text-sm sm:text-base font-bold tracking-tight text-white">
                OUTPOST COMMAND
              </h1>
              <span
                data-testid="hud-sol-counter"
                className="text-xs sm:text-sm font-mono font-semibold text-cyan-300 whitespace-nowrap"
              >
                <ScienceTermButton termKey="sol" onSelectTerm={setActiveTerm} /> {mission.sol} /{' '}
                {MAX_SOLS}
              </span>
              <span
                data-testid="hud-phase-badge"
                className={`inline-flex items-center gap-1 px-2 py-0.5 rounded-md border text-[11px] font-mono font-semibold whitespace-nowrap ${
                  mission.opPhase === 'sol_complete'
                    ? 'bg-emerald-950/90 border-emerald-400/70 text-emerald-200'
                    : mission.opPhase === 'operation'
                    ? 'bg-cyan-950/90 border-cyan-400/70 text-cyan-200'
                    : 'bg-amber-950/90 border-amber-400/70 text-amber-200'
                }`}
              >
                {phaseLabel}
              </span>
              {isEmergency && (
                <span
                  data-testid="emergency-protocol-badge"
                  className="inline-flex items-center gap-1 px-2 py-0.5 rounded-md bg-rose-950/90 border border-rose-500/80 text-[11px] font-mono font-bold text-rose-200 whitespace-nowrap"
                >
                  <AlertTriangle className="w-3.5 h-3.5 text-amber-400 shrink-0" aria-hidden="true" />
                  <span>
                    {donkiVisualState.level === 'x_class' || emergencyMode === 'flare'
                      ? 'SOLAR FLARE ALERT'
                      : 'DUST STORM ALERT'}
                  </span>
                </span>
              )}
              <span className="hidden md:inline-flex items-center gap-2 text-xs font-mono text-slate-300">
                <span data-testid="hud-sol-clock" className="text-cyan-300 font-semibold">
                  {formatMmSs(mission.operationElapsedSec)} / {formatMmSs(SOL_DURATION_SEC)}
                </span>
                <span className="text-slate-600">·</span>
                <span>{mission.weather.tempMinC}°C</span>
                <span className="text-slate-600">·</span>
                <span>
                  <ScienceTermButton termKey="tau" label="Tau" onSelectTerm={setActiveTerm} />{' '}
                  {atmosphereTargets.effectiveTau.toFixed(2)}
                </span>
              </span>
              <button
                type="button"
                data-testid="donki-telemetry-pill"
                aria-expanded={donkiFeedOpen}
                onClick={() => {
                  missionAudio.playClick();
                  setDonkiFeedOpen((prev) => !prev);
                }}
                title="Inspect Live NASA DONKI Event & 3D Space Weather"
                className={`inline-flex items-center gap-1 px-2 py-0.5 rounded-md border text-xs font-mono font-bold cursor-pointer transition-colors whitespace-nowrap ${
                  donkiVisualState.level === 'x_class'
                    ? 'bg-rose-950/90 border-rose-400 text-rose-200 hover:bg-rose-900'
                    : donkiVisualState.level === 'm_class'
                    ? 'bg-fuchsia-950/75 border-fuchsia-400/70 text-amber-200 hover:bg-fuchsia-900/75'
                    : 'bg-slate-800/90 border-slate-600 text-amber-300 hover:border-cyan-400'
                }`}
              >
                <Sun className="w-3.5 h-3.5 shrink-0" aria-hidden="true" />
                <span>{donkiVisualState.flare.classType}</span>
                {donkiVisualState.cme.impactActive && (
                  <span className="text-[10px] text-rose-300">· CME</span>
                )}
              </button>
            </div>

            <div className="flex items-center gap-1.5 sm:gap-2 shrink-0">
              <button
                type="button"
                data-testid="nasa-data-log-btn"
                aria-expanded={donkiFeedOpen}
                onClick={() => {
                  missionAudio.playClick();
                  setDonkiFeedOpen((prev) => !prev);
                }}
                className={`inline-flex items-center gap-1.5 px-2.5 py-1.5 rounded-lg text-xs font-medium border cursor-pointer whitespace-nowrap transition-colors ${
                  donkiFeedOpen
                    ? 'bg-cyan-500/20 border-cyan-400 text-cyan-100'
                    : 'bg-slate-800 hover:bg-slate-700 text-cyan-300 border-slate-700'
                }`}
              >
                <Database className="w-3.5 h-3.5" aria-hidden="true" />
                <span>NASA Data Log</span>
                <span className="font-mono text-[10px] text-emerald-300">
                  ({unlockedLogIds.length}/9)
                </span>
              </button>
              <button
                type="button"
                onClick={toggleMute}
                aria-label={muted ? 'Unmute sound effects' : 'Mute sound effects'}
                className="p-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-200 border border-slate-700 cursor-pointer"
              >
                {muted ? <VolumeX className="w-4 h-4" /> : <Volume2 className="w-4 h-4" />}
              </button>
              <button
                type="button"
                onClick={() => {
                  missionAudio.playClick();
                  startMission();
                }}
                aria-label="Restart mission"
                className="p-1.5 sm:px-2.5 sm:py-1.5 inline-flex items-center gap-1 rounded-lg bg-slate-800 hover:bg-slate-700 text-xs font-medium text-slate-200 border border-slate-700 cursor-pointer"
              >
                <RotateCcw className="w-3.5 h-3.5" aria-hidden="true" />
                <span className="hidden sm:inline">Reset</span>
              </button>
            </div>
          </div>

          {/* Row 2: The 3 Coupled Resource Bars */}
          <div className="grid grid-cols-3 gap-2 sm:gap-4">
            {/* 1. Power Grid */}
            <div
              className={`relative bg-[#0B1220] border rounded-xl px-2.5 py-1.5 ${
                isEmergency ? 'border-rose-800/80' : 'border-slate-800'
              }`}
            >
              <div className="flex items-center justify-between gap-1 text-[11px] sm:text-xs">
                <span className="inline-flex items-center gap-1 font-medium text-amber-200 truncate">
                  <Zap className="w-3.5 h-3.5 text-amber-400 shrink-0" aria-hidden="true" />
                  <span className="hidden sm:inline">Power</span>
                </span>
                <span
                  data-testid="meter-power-value"
                  className="font-mono font-bold text-white tabular-nums"
                >
                  {mission.opPhase === 'operation'
                    ? mission.exactResources.powerKwh.toFixed(1)
                    : mission.resources.powerKwh}{' '}
                  <span className="text-[10px] text-slate-400">kWh</span>
                </span>
              </div>
              <div className="mt-1 h-1.5 rounded-full bg-slate-800 overflow-hidden">
                <div
                  className={`h-full rounded-full transition-all duration-200 ${
                    powerPct < 25 ? 'bg-rose-500' : 'bg-amber-400'
                  }`}
                  style={{ width: `${powerPct}%` }}
                />
              </div>
              {mission.opPhase === 'operation' && mission.operationRates && (
                <p className="mt-0.5 text-[10px] font-mono text-slate-400 text-right tabular-nums">
                  {formatRate(mission.operationRates.powerPerSec, 'kWh')}
                </p>
              )}
              {hudDeltas && hudDeltas.power !== 0 && (
                <span
                  key={`p-${hudDeltas.id}`}
                  data-testid="hud-delta-power"
                  className={`anim-float-pop absolute -bottom-4 right-2 text-[11px] font-mono font-bold px-1.5 py-0.5 rounded bg-[#070B14] border ${
                    hudDeltas.power > 0
                      ? 'text-emerald-300 border-emerald-500/60'
                      : 'text-amber-300 border-amber-500/60'
                  }`}
                >
                  {hudDeltas.power > 0 ? `+${hudDeltas.power}` : hudDeltas.power} kWh
                </span>
              )}
            </div>

            {/* 2. Life Support O2 & Water */}
            <div
              className={`relative bg-[#0B1220] border rounded-xl px-2.5 py-1.5 ${
                isEmergency ? 'border-rose-800/80' : 'border-slate-800'
              }`}
            >
              <div className="flex items-center justify-between gap-1 text-[11px] sm:text-xs">
                <span className="inline-flex items-center gap-1 font-medium text-cyan-200 truncate">
                  <Wind className="w-3.5 h-3.5 text-cyan-400 shrink-0" aria-hidden="true" />
                  <span className="hidden sm:inline">O₂ & Water</span>
                </span>
                <span
                  data-testid="meter-life-value"
                  className="font-mono font-bold text-white tabular-nums"
                >
                  {mission.opPhase === 'operation'
                    ? `${mission.exactResources.lifeSupportPct.toFixed(1)}%`
                    : `${lifePct}%`}
                </span>
              </div>
              <div className="mt-1 h-1.5 rounded-full bg-slate-800 overflow-hidden">
                <div
                  className={`h-full rounded-full transition-all duration-200 ${
                    lifePct < 30 ? 'bg-rose-500' : 'bg-cyan-400'
                  }`}
                  style={{ width: `${lifePct}%` }}
                />
              </div>
              {mission.opPhase === 'operation' && mission.operationRates && (
                <p className="mt-0.5 text-[10px] font-mono text-slate-400 text-right tabular-nums">
                  {formatRate(mission.operationRates.lifePerSec, '%')}
                </p>
              )}
              {hudDeltas && hudDeltas.life !== 0 && (
                <span
                  key={`l-${hudDeltas.id}`}
                  data-testid="hud-delta-life"
                  className={`anim-float-pop absolute -bottom-4 right-2 text-[11px] font-mono font-bold px-1.5 py-0.5 rounded bg-[#070B14] border ${
                    hudDeltas.life > 0
                      ? 'text-cyan-300 border-cyan-500/60'
                      : 'text-amber-300 border-amber-500/60'
                  }`}
                >
                  {hudDeltas.life > 0 ? `+${hudDeltas.life}%` : `${hudDeltas.life}%`}
                </span>
              )}
            </div>

            {/* 3. Radiation Shield & Crew Health */}
            <div
              className={`relative bg-[#0B1220] border rounded-xl px-2.5 py-1.5 ${
                isEmergency ? 'border-rose-800/80' : 'border-slate-800'
              }`}
            >
              <div className="flex items-center justify-between gap-1 text-[11px] sm:text-xs">
                <span className="inline-flex items-center gap-1 font-medium text-emerald-200 truncate">
                  <Shield className="w-3.5 h-3.5 text-emerald-400 shrink-0" aria-hidden="true" />
                  <span className="hidden sm:inline">Shield & Crew</span>
                </span>
                <span
                  data-testid="meter-health-value"
                  className="font-mono font-bold text-white tabular-nums"
                >
                  {mission.opPhase === 'operation'
                    ? `${mission.exactResources.crewHealthPct.toFixed(1)}%`
                    : `${healthPct}%`}{' '}
                  <span className="hidden md:inline text-[10px] text-sky-300">
                    ({mission.resources.waterWallCm}cm)
                  </span>
                </span>
              </div>
              <div className="mt-1 h-1.5 rounded-full bg-slate-800 overflow-hidden">
                <div
                  className={`h-full rounded-full transition-all duration-200 ${
                    healthPct < 35 ? 'bg-rose-500' : 'bg-emerald-400'
                  }`}
                  style={{ width: `${healthPct}%` }}
                />
              </div>
              {mission.opPhase === 'operation' && mission.operationRates && (
                <p className="mt-0.5 text-[10px] font-mono text-slate-400 text-right tabular-nums">
                  {formatRate(mission.operationRates.healthPerSec, '%')}
                </p>
              )}
              {hudDeltas && hudDeltas.health !== 0 && (
                <span
                  key={`h-${hudDeltas.id}`}
                  data-testid="hud-delta-health"
                  className={`anim-float-pop absolute -bottom-4 right-2 text-[11px] font-mono font-bold px-1.5 py-0.5 rounded bg-[#070B14] border ${
                    hudDeltas.health > 0
                      ? 'text-emerald-300 border-emerald-500/60'
                      : 'text-rose-300 border-rose-500/60'
                  }`}
                >
                  {hudDeltas.health > 0 ? `+${hudDeltas.health}%` : `${hudDeltas.health}%`}
                </span>
              )}
            </div>
          </div>
        </div>
      </header>

      {/* Non-Intrusive Live NASA DONKI Feed Inspector Card */}
      <DonkiInspectorCard
        open={donkiFeedOpen && !modalOpen}
        donkiResult={donkiResult}
        weatherState={donkiVisualState}
        waterWallCm={mission.resources.waterWallCm}
        shieldActive={shieldActive}
        unlockedCount={unlockedLogIds.length}
        onToggleSimulateXFlare={() => setSimulateXFlare((prev) => !prev)}
        onOpenArchive={() => {
          setDonkiFeedOpen(false);
          setShowScienceLog(true);
        }}
        onClose={() => setDonkiFeedOpen(false)}
      />

      {/* Bottom Dock: Optional Earth Audio Dispatch Drawer + Operational Phase Console */}
      <footer className="fixed bottom-2.5 sm:bottom-4 inset-x-2.5 sm:inset-x-5 z-20 pointer-events-none">
        <EarthDispatchDrawer
          open={dispatchOpen && !isNightCycle}
          currentSol={mission.sol}
          emergencyMode={emergencyMode}
          decryptedSols={decryptedSols}
          onMarkDecrypted={handleMarkDecrypted}
          onClose={() => setDispatchOpen(false)}
        />

        <div
          data-testid="story-banner"
          className={`pointer-events-auto max-w-5xl mx-auto rounded-2xl p-3 sm:p-4 shadow-2xl flex flex-col gap-3 border transition-colors ${emBorderClass} ${
            isEmergency
              ? 'bg-[#19090E]/94 border-rose-500/80'
              : donkiVisualState.level === 'm_class'
              ? 'bg-[#100918]/92 border-fuchsia-400/55'
              : 'bg-[#070B14]/92 border-slate-700/85'
          }`}
        >
          {/* Phase 2 / Phase 3 5-Minute Sol Timeline & Progress Bar */}
          {isPhase2Or3 && (
            <div
              data-testid="operation-timer-panel"
              className="rounded-xl bg-[#0B1220] border border-slate-800 px-3 py-2 flex flex-col gap-1.5"
            >
              <div className="flex flex-wrap items-center justify-between gap-2 text-xs">
                <div className="flex items-center gap-2 min-w-0">
                  <Clock
                    className={`w-3.5 h-3.5 shrink-0 ${
                      mission.opPhase === 'sol_complete' ? 'text-emerald-400' : 'text-cyan-400'
                    }`}
                    aria-hidden="true"
                  />
                  <span className="font-mono font-semibold text-white">
                    {mission.opPhase === 'sol_complete'
                      ? '5-Minute Sol Complete — Ready to Advance'
                      : mission.fastForwardActive
                      ? `${FAST_FORWARD_MULTIPLIER}× Fast Forward — Atmosphere & Systems Advancing`
                      : '5-Minute Sol Operation — Monitor Base'}
                  </span>
                  {mission.lockedChoice && (
                    <span className="hidden sm:inline-flex items-center gap-1 px-2 py-0.5 rounded bg-slate-800 border border-slate-700 text-[11px] font-mono text-amber-200 truncate">
                      Directive: {mission.lockedChoice.label}
                    </span>
                  )}
                </div>

                <div className="flex items-center gap-3 font-mono text-xs tabular-nums shrink-0">
                  <span
                    data-testid="operation-speed-badge"
                    className={`px-1.5 py-0.5 rounded border text-[10px] font-bold ${
                      mission.fastForwardActive
                        ? 'bg-cyan-500/20 border-cyan-400 text-cyan-200'
                        : 'bg-slate-800 border-slate-700 text-slate-300'
                    }`}
                  >
                    {mission.speedMultiplier}x
                  </span>
                  <span data-testid="operation-timer-text" className="text-cyan-300 font-bold">
                    {mission.opPhase === 'sol_complete'
                      ? '00:00 remaining'
                      : `${formatMmSs(remainingSec)} remaining`}
                  </span>
                  <span className="text-slate-400">{opProgressPct}%</span>
                </div>
              </div>

              <div
                role="progressbar"
                aria-label="5-Minute Sol Progress"
                aria-valuemin={0}
                aria-valuemax={100}
                aria-valuenow={opProgressPct}
                className="h-2 rounded-full bg-slate-800 overflow-hidden"
              >
                <div
                  data-testid="operation-progress-fill"
                  className={`h-full rounded-full transition-all duration-200 ${
                    mission.opPhase === 'sol_complete'
                      ? 'bg-emerald-400'
                      : mission.fastForwardActive
                      ? 'bg-amber-400'
                      : 'bg-cyan-400'
                  }`}
                  style={{ width: `${opProgressPct}%` }}
                />
              </div>
            </div>
          )}

          {/* Main Command Row: Briefing / Crisis Alert + Phase Action Controls */}
          <div className="flex flex-col md:flex-row items-stretch md:items-center justify-between gap-3">
            <div className="flex items-start gap-3 min-w-0 flex-1">
              <CharacterPortrait speaker={chapter.speaker} />
              <div className="min-w-0 flex-1 space-y-1">
                <div className="flex flex-wrap items-center gap-x-2 gap-y-0.5 text-xs">
                  <strong className="text-sm text-white">{chapter.speakerName}</strong>
                  <span
                    className={
                      isEmergency ? 'text-rose-300 font-mono' : 'text-amber-300 font-mono'
                    }
                  >
                    {chapter.arcTitle}
                  </span>
                  <span className="text-slate-400 hidden sm:inline">· {chapter.commNote}</span>
                </div>

                {isPhase1 && mission.pendingDilemma ? (
                  <div
                    data-testid="crisis-alert-box"
                    className="rounded-lg bg-amber-950/35 border border-amber-500/40 px-2.5 py-1.5 text-xs text-slate-200 leading-snug"
                  >
                    <span className="font-mono font-bold text-amber-300 uppercase mr-1.5">
                      Crisis Alert:
                    </span>
                    <strong className="text-white">{mission.pendingDilemma.title} — </strong>
                    <span>{mission.pendingDilemma.description}</span>
                  </div>
                ) : (
                  <p className="text-xs sm:text-sm text-slate-200 leading-snug">
                    {isNightCycle
                      ? `Simulating Sol ${mission.sol} sunset & overnight transition across Jezero Crater...`
                      : chapter.storyPrompt}
                  </p>
                )}

                <div className="flex flex-wrap items-center gap-x-3 gap-y-1 text-[11px] text-slate-400 pt-0.5">
                  <button
                    type="button"
                    data-testid="toggle-dispatch-btn"
                    disabled={isNightCycle}
                    onClick={() => {
                      missionAudio.playClick();
                      setDispatchOpen((prev) => !prev);
                    }}
                    className={`inline-flex items-center gap-1.5 px-2 py-0.5 rounded-md border font-mono text-[11px] cursor-pointer transition-colors ${
                      dispatchOpen
                        ? 'bg-cyan-500/20 border-cyan-400 text-cyan-200'
                        : hasUnreadDispatch
                        ? 'bg-amber-500/15 border-amber-400/70 text-amber-200 hover:bg-amber-500/25'
                        : 'bg-[#0B1220] border-slate-700 text-cyan-300 hover:border-cyan-400'
                    }`}
                  >
                    <Radio className="w-3 h-3 shrink-0" aria-hidden="true" />
                    <span>{dispatchOpen ? 'Hide DSN Dispatch' : 'Houston DSN Dispatch'}</span>
                    {hasUnreadDispatch && !dispatchOpen && (
                      <span className="text-amber-300 font-bold">· NEW</span>
                    )}
                  </button>
                  <span className="hidden sm:inline">Decoder:</span>
                  <ScienceTermButton termKey="moxie" onSelectTerm={setActiveTerm} />
                  <ScienceTermButton termKey="waterwall" onSelectTerm={setActiveTerm} />
                  <ScienceTermButton termKey="hydroponics" onSelectTerm={setActiveTerm} />
                  <ScienceTermButton termKey="commdelay" onSelectTerm={setActiveTerm} />
                  <span className="font-mono text-cyan-300 ml-auto">
                    {mission.resources.sciencePoints} Sci Pts
                  </span>
                </div>
              </div>
            </div>

            {/* Phase-Specific Right Action Area */}
            <div className="flex flex-wrap sm:flex-nowrap items-center gap-2 shrink-0">
              {isPhase1 && (
                <button
                  type="button"
                  data-testid="open-decision-cards-btn"
                  disabled={isNightCycle || isDecisionModalOpen}
                  onClick={handleOpenDecisionCards}
                  className="w-full md:w-auto inline-flex items-center justify-center gap-2 px-5 py-3 rounded-xl bg-amber-500 hover:bg-amber-400 text-slate-950 font-semibold text-xs sm:text-sm transition-colors cursor-pointer whitespace-nowrap focus-visible:outline-2 focus-visible:outline-amber-300"
                >
                  <Sparkles className="w-4 h-4 shrink-0" aria-hidden="true" />
                  <span>Open Decision Cards</span>
                </button>
              )}

              {isPhase2Or3 && (
                <>
                  <button
                    type="button"
                    data-testid="fast-forward-btn"
                    disabled={
                      !mission.fastForwardUnlocked ||
                      mission.opPhase !== 'operation' ||
                      isNightCycle ||
                      modalOpen
                    }
                    onClick={handleToggleFastForward}
                    className={`inline-flex items-center justify-center gap-1.5 px-3.5 py-3 rounded-xl font-semibold text-xs sm:text-sm border transition-colors whitespace-nowrap focus-visible:outline-2 focus-visible:outline-cyan-300 ${
                      mission.fastForwardUnlocked && mission.opPhase === 'operation' && !isNightCycle
                        ? mission.fastForwardActive
                          ? 'bg-cyan-500/20 hover:bg-cyan-500/30 border-cyan-400 text-cyan-100 cursor-pointer'
                          : 'bg-slate-800 hover:bg-slate-700 border-cyan-400/70 text-cyan-200 cursor-pointer'
                        : 'bg-[#0B1220] border-slate-800 text-slate-500 opacity-65 cursor-not-allowed'
                    }`}
                  >
                    {mission.fastForwardUnlocked ? (
                      <FastForward className="w-4 h-4 shrink-0 text-cyan-300" aria-hidden="true" />
                    ) : (
                      <Lock className="w-3.5 h-3.5 shrink-0" aria-hidden="true" />
                    )}
                    <span>
                      {mission.opPhase === 'sol_complete'
                        ? 'Sol Complete'
                        : !mission.fastForwardUnlocked
                        ? `Fast Forward (${formatMmSs(fastForwardWaitSec)})`
                        : mission.fastForwardActive
                        ? `${FAST_FORWARD_MULTIPLIER}x Speed Active`
                        : `Fast Forward (${FAST_FORWARD_MULTIPLIER}x)`}
                    </span>
                  </button>

                  {mission.opPhase === 'sol_complete' && (
                    <button
                      type="button"
                      data-testid="end-sol-btn"
                      disabled={isNightCycle || modalOpen}
                      onClick={handleAdvanceNextSol}
                      className="inline-flex items-center justify-center gap-2 px-4 sm:px-5 py-3 rounded-xl font-semibold text-xs sm:text-sm transition-colors cursor-pointer whitespace-nowrap focus-visible:outline-2 focus-visible:outline-cyan-300 bg-emerald-400 hover:bg-emerald-300 text-slate-950"
                    >
                      <Moon className="w-4 h-4 shrink-0" aria-hidden="true" />
                      <span>End Sol / Next Operation</span>
                    </button>
                  )}
                </>
              )}
            </div>
          </div>
        </div>
      </footer>

      {/* Modals */}
      {isDecisionModalOpen && mission.pendingDilemma && !mission.debrief && (
        <DilemmaModal
          dilemma={mission.pendingDilemma}
          sol={mission.sol}
          onChoose={handleChooseDilemma}
          onOpenScienceLog={() => setShowScienceLog(true)}
        />
      )}

      {showScienceLog && (
        <ScienceLogModal
          donkiResult={donkiResult}
          activeSol={mission.sol}
          unlockedLogIds={unlockedLogIds}
          onClose={() => setShowScienceLog(false)}
        />
      )}

      {activeTerm && (
        <ScienceDecoderModal term={activeTerm} onClose={() => setActiveTerm(null)} />
      )}

      {mission.debrief && (
        <DebriefModal debrief={mission.debrief} onReplay={startMission} />
      )}
    </div>
  );
}
