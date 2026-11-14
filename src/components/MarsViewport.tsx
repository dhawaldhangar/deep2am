import React, { useEffect, useImperativeHandle, useRef, useState } from 'react';
import { Shield, Sparkles, Sprout, Wind, Zap } from 'lucide-react';
import { ModuleId, SolHazardType } from '../data/campaignArc';
import { MissionResources, SolAtmosphereTargets } from '../game/simulation';
import {
  ActiveSystemGlow,
  DonkiSceneTelemetry,
  EmergencyMode,
  MarsOutpostScene,
  ProjectedModuleAnchor,
} from '../three/MarsOutpostScene';

export interface ModulePopup {
  id: number;
  moduleId: ModuleId;
  text: string;
  positive: boolean;
}

export interface MarsViewportHandle {
  triggerModulePulse: (moduleId: ModuleId) => void;
  triggerNightCycle: (
    hazardType: SolHazardType,
    shieldBlockedFlare: boolean,
    onComplete: () => void,
    durationSec?: number
  ) => void;
}

interface MarsViewportProps {
  resources: MissionResources;
  tau: number;
  targetModule: ModuleId | 'any';
  emergencyMode: EmergencyMode;
  atmosphere: SolAtmosphereTargets;
  hazardType: SolHazardType;
  activeSystemGlow: ActiveSystemGlow;
  donkiTelemetry: DonkiSceneTelemetry;
  speedMultiplier: number;
  isNightCycle: boolean;
  disabled: boolean;
  activePopup: ModulePopup | null;
  onModuleAction: (moduleId: ModuleId) => void;
}

const MODULE_META: Record<
  ModuleId,
  {
    name: string;
    actionLabel: string;
    tradeoffShort: string;
    accentClass: string;
    icon: React.FC<{ className?: string }>;
  }
> = {
  solar: {
    name: 'Solar Array Field',
    actionLabel: 'Clean Dust',
    tradeoffShort: '+Power · -3% H₂O',
    accentClass: 'border-amber-400/70 text-amber-200 hover:border-amber-300',
    icon: Zap,
  },
  moxie: {
    name: 'MOXIE O₂ Core',
    actionLabel: 'Boost O₂',
    tradeoffShort: '+14% O₂ · -12 kWh',
    accentClass: 'border-cyan-400/70 text-cyan-200 hover:border-cyan-300',
    icon: Wind,
  },
  shelter: {
    name: 'Water-Wall Shelter',
    actionLabel: 'Shield Up',
    tradeoffShort: '+6cm Wall · -14 kWh',
    accentClass: 'border-sky-400/70 text-sky-200 hover:border-sky-300',
    icon: Shield,
  },
  biodome: {
    name: 'Hydroponic Bio-Dome',
    actionLabel: 'Water Crops',
    tradeoffShort: '+8% O₂/Food · -10 kWh',
    accentClass: 'border-emerald-400/70 text-emerald-200 hover:border-emerald-300',
    icon: Sprout,
  },
};

export const MarsViewport = React.forwardRef<MarsViewportHandle, MarsViewportProps>(
  (
    {
      resources,
      tau,
      targetModule,
      emergencyMode,
      atmosphere,
      hazardType,
      activeSystemGlow,
      donkiTelemetry,
      speedMultiplier,
      isNightCycle,
      disabled,
      activePopup,
      onModuleAction,
    },
    ref
  ) => {
    const mountRef = useRef<HTMLDivElement | null>(null);
    const sceneRef = useRef<MarsOutpostScene | null>(null);

    const onActionRef = useRef(onModuleAction);
    onActionRef.current = onModuleAction;

    const disabledRef = useRef(disabled || isNightCycle);
    disabledRef.current = disabled || isNightCycle;

    const [anchors, setAnchors] = useState<ProjectedModuleAnchor[]>([]);
    const [hoveredId, setHoveredId] = useState<ModuleId | null>(null);
    const [webglError, setWebglError] = useState<string | null>(null);

    useImperativeHandle(ref, () => ({
      triggerModulePulse: (moduleId: ModuleId) => {
        sceneRef.current?.triggerModuleFeedback(moduleId);
      },
      triggerNightCycle: (hazardType, shieldBlockedFlare, onComplete, durationSec) => {
        sceneRef.current?.triggerNightCycle(hazardType, shieldBlockedFlare, onComplete, durationSec);
      },
    }));

    useEffect(() => {
      const el = mountRef.current;
      if (!el) return;

      const controller = new MarsOutpostScene(el, {
        onModuleClick: (moduleId) => {
          if (!disabledRef.current) {
            onActionRef.current(moduleId);
          }
        },
        onHoverModule: (moduleId) => {
          setHoveredId(moduleId);
        },
        onAnchorsUpdate: (next) => {
          setAnchors(next);
        },
        onWebglError: (reason) => {
          setWebglError(reason);
        },
      });
      sceneRef.current = controller;

      return () => {
        controller.dispose();
        sceneRef.current = null;
      };
    }, []);

    useEffect(() => {
      sceneRef.current?.updateMissionVisuals(resources, tau, targetModule, emergencyMode, {
        atmosphere,
        hazardType,
        activeSystemGlow,
        speedMultiplier,
        donki: donkiTelemetry,
      });
    }, [
      resources,
      tau,
      targetModule,
      emergencyMode,
      atmosphere,
      hazardType,
      activeSystemGlow,
      speedMultiplier,
      donkiTelemetry,
    ]);

    const getDynamicStatus = (id: ModuleId): string => {
      if (id === 'solar') return `Dust ${resources.panelDustPct}%`;
      if (id === 'moxie') return `O₂ ${resources.lifeSupportPct}%`;
      if (id === 'shelter') return `Wall ${resources.waterWallCm}cm`;
      return `Crop Lv.${resources.cropGrowth}`;
    };

    return (
      <div className="fixed inset-0 overflow-hidden bg-[#1D0F0C]" data-testid="mars-3d-viewport">
        {/* Isolated WebGL Canvas Container */}
        <div ref={mountRef} className="absolute inset-0" />

        {webglError && (
          <div className="absolute inset-0 flex items-center justify-center p-6 bg-[#070B14] text-center">
            <div className="max-w-md bg-[#0F172A] border border-slate-700 rounded-2xl p-6 space-y-2">
              <p className="text-base font-bold text-white">3D Viewport Unavailable</p>
              <p className="text-xs text-slate-300">{webglError}</p>
            </div>
          </div>
        )}

        {/* Projected On-Object 3D Module Hotspots */}
        {!isNightCycle && !webglError && (
          <div className="absolute inset-0 pointer-events-none z-10">
            {anchors.map((anchor) => {
              if (!anchor.visible) return null;
              const meta = MODULE_META[anchor.id];
              const Icon = meta.icon;
              const isTarget = targetModule === anchor.id;
              const isHovered = hoveredId === anchor.id;
              const hasPopup = activePopup?.moduleId === anchor.id;

              return (
                <div
                  key={anchor.id}
                  style={{
                    left: `${anchor.x}px`,
                    top: `${anchor.y}px`,
                  }}
                  className="absolute -translate-x-1/2 -translate-y-full pointer-events-auto"
                >
                  {/* Floating Action Feedback Callout */}
                  {hasPopup && activePopup && (
                    <div
                      key={activePopup.id}
                      data-testid={`module-feedback-${anchor.id}`}
                      className={`anim-float-pop mb-1.5 px-2.5 py-1 rounded-lg text-[11px] font-mono font-bold text-center whitespace-nowrap border shadow-lg ${
                        activePopup.positive
                          ? 'bg-[#071F1B] border-emerald-400 text-emerald-200'
                          : 'bg-[#290F14] border-rose-400 text-rose-200'
                      }`}
                    >
                      {activePopup.text}
                    </div>
                  )}

                  <button
                    type="button"
                    data-testid={`hotspot-${anchor.id}`}
                    disabled={disabled}
                    onClick={() => onModuleAction(anchor.id)}
                    aria-label={`${meta.name}: ${meta.actionLabel} (${getDynamicStatus(anchor.id)} · ${meta.tradeoffShort})`}
                    className={`group relative flex items-center gap-1.5 px-2.5 py-1 rounded-lg border text-left transition-all duration-150 cursor-pointer shadow-md focus-visible:outline-2 focus-visible:outline-cyan-300 ${
                      isHovered
                        ? 'bg-[#0B1220] opacity-100 -translate-y-0.5 border-amber-300 shadow-lg'
                        : isTarget
                        ? 'bg-[#070B14]/70 opacity-80 hover:opacity-100 hover:bg-[#0B1220] focus-visible:opacity-100 focus-visible:bg-[#0B1220] border-cyan-400/85 ring-1 ring-cyan-400/35 hover:-translate-y-0.5'
                        : `bg-[#070B14]/65 opacity-70 hover:opacity-100 hover:bg-[#0B1220] focus-visible:opacity-100 focus-visible:bg-[#0B1220] hover:-translate-y-0.5 ${meta.accentClass}`
                    }`}
                  >
                    <Icon className="w-3 h-3 shrink-0" aria-hidden="true" />
                    <span className="text-[11px] font-semibold tracking-tight text-white whitespace-nowrap">
                      {meta.actionLabel}
                    </span>
                    {isTarget && (
                      <Sparkles className="w-2.5 h-2.5 text-cyan-300 shrink-0" aria-hidden="true" />
                    )}

                    {/* Compact hover/focus detail chip */}
                    {!hasPopup && (
                      <span
                        className={`pointer-events-none absolute bottom-full left-1/2 -translate-x-1/2 mb-1.5 px-2 py-0.5 rounded-md bg-[#0B1220] border border-slate-700/90 text-[10px] font-mono text-slate-200 whitespace-nowrap shadow-lg transition-opacity duration-150 ${
                          isHovered
                            ? 'opacity-100'
                            : 'opacity-0 group-hover:opacity-100 group-focus-visible:opacity-100'
                        }`}
                      >
                        <span>{getDynamicStatus(anchor.id)}</span>
                        <span className="text-slate-500 mx-1">·</span>
                        <span className="text-cyan-300">{meta.tradeoffShort}</span>
                      </span>
                    )}
                  </button>
                </div>
              );
            })}
          </div>
        )}
      </div>
    );
  }
);

MarsViewport.displayName = 'MarsViewport';
