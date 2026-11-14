import test from 'node:test';
import assert from 'node:assert/strict';
import {
  activateFastForward,
  advanceOperationTick,
  advanceToNextSol,
  applyModuleActionToMission,
  calculateNightCycle,
  checkDefeatReason,
  computeSolAtmosphereTargets,
  computeTauFogDensity,
  createInitialMissionState,
  FAST_FORWARD_MULTIPLIER,
  FAST_FORWARD_UNLOCK_SEC,
  getSolChapter,
  lockInDecisionChoice,
  MAX_SOLS,
  openDecisionCards,
  OPERATION_DURATION_SEC,
  performModuleAction,
  resolveDilemmaChoice,
  resolveNightCycleStep,
  SOL_DURATION_SEC,
  toggleFastForward,
} from './simulation';
import { VERIFIED_FALLBACK_FLARES } from '../data/nasaTelemetry';
import { computeUnlockedLogIds, NASA_ARCHIVE_ENTRIES } from '../data/campaignArc';
import {
  classifyDonkiSeverity,
  computeDonkiVisualState,
  DEFAULT_ARCHIVE_CME,
  HISTORICAL_X9_CME,
  HISTORICAL_X9_FLARE,
} from '../utils/nasaApi';

test('initial mission state starts at Sol 1 in Phase 1 crisis_brief with 2 tactical choices and 300s Sol duration', () => {
  const state = createInitialMissionState(VERIFIED_FALLBACK_FLARES);
  assert.equal(SOL_DURATION_SEC, 300);
  assert.equal(FAST_FORWARD_UNLOCK_SEC, 120);
  assert.equal(FAST_FORWARD_MULTIPLIER, 5);
  assert.equal(state.sol, 1);
  assert.equal(state.opPhase, 'crisis_brief');
  assert.equal(state.operationElapsedSec, 0);
  assert.equal(state.operationDurationSec, SOL_DURATION_SEC);
  assert.equal(state.decisionElapsedSec, null);
  assert.equal(state.fastForwardUnlocked, false);
  assert.equal(state.speedMultiplier, 1);
  assert.equal(state.resources.powerKwh, 88);
  assert.equal(state.resources.lifeSupportPct, 80);
  assert.equal(state.resources.crewHealthPct, 88);
  assert.equal(state.resources.waterWallCm, 16);
  assert.ok(state.pendingDilemma);
  assert.equal(state.pendingDilemma.choices.length, 2);
  assert.equal(state.debrief, null);
});

test('on-object module actions apply coupled trade-offs directly', () => {
  const state = createInitialMissionState(VERIFIED_FALLBACK_FLARES);

  // MOXIE raises O2 (+14%) but costs 12 kWh power
  const moxieRes = performModuleAction(state.resources, 'moxie');
  assert.equal(moxieRes.applied, true);
  assert.equal(moxieRes.deltaPower, -12);
  assert.equal(moxieRes.deltaLife, 14);

  // Shelter thickens Water-Wall (+6 cm) and health, while draining 14 kWh power & 5% water
  const shelterRes = performModuleAction(state.resources, 'shelter');
  assert.equal(shelterRes.applied, true);
  assert.equal(shelterRes.deltaPower, -14);
  assert.equal(shelterRes.deltaLife, -5);
  assert.equal(shelterRes.deltaShieldCm, 6);

  // Solar cleans dust (-16% from 16% -> 0%) and harvests +14 kWh for -3% cooling water
  const solarRes = performModuleAction(state.resources, 'solar');
  assert.equal(solarRes.applied, true);
  assert.ok(solarRes.deltaDust < 0);
  assert.equal(solarRes.deltaPower, 14);
  assert.equal(solarRes.deltaLife, -3);

  // Bio-Dome grows crops & boosts science (+5) while spending 10 kWh
  const bioRes = performModuleAction(state.resources, 'biodome');
  assert.equal(bioRes.applied, true);
  assert.equal(bioRes.nextResources.cropGrowth, state.resources.cropGrowth + 1);
  assert.equal(bioRes.nextResources.sciencePoints, 5);
});

test('thick Water-Wall deflects Sol 4 Solar Flare while thin wall suffers radiation damage', () => {
  const state = createInitialMissionState(VERIFIED_FALLBACK_FLARES);
  const sol4 = getSolChapter(4);
  assert.equal(sol4.hazardType, 'flare');

  const thinNight = calculateNightCycle(
    { ...state.resources, waterWallCm: 14 },
    state.weather,
    state.activeFlare,
    sol4
  );
  const thickNight = calculateNightCycle(
    { ...state.resources, waterWallCm: 26 },
    state.weather,
    state.activeFlare,
    sol4
  );

  assert.equal(thinNight.shieldBlockedFlare, false);
  assert.equal(thickNight.shieldBlockedFlare, true);
  assert.ok(thickNight.netCrewHealthPct > thinNight.netCrewHealthPct);
});

test('defeat triggers when any critical resource hits 0', () => {
  assert.ok(
    checkDefeatReason({
      powerKwh: 0,
      lifeSupportPct: 50,
      crewHealthPct: 50,
      panelDustPct: 10,
      waterWallCm: 16,
      cropGrowth: 2,
      sciencePoints: 0,
    }) !== null
  );
  assert.ok(
    checkDefeatReason({
      powerKwh: 50,
      lifeSupportPct: 0,
      crewHealthPct: 50,
      panelDustPct: 10,
      waterWallCm: 16,
      cropGrowth: 2,
      sciencePoints: 0,
    }) !== null
  );
  assert.ok(
    checkDefeatReason({
      powerKwh: 50,
      lifeSupportPct: 50,
      crewHealthPct: 0,
      panelDustPct: 10,
      waterWallCm: 16,
      cropGrowth: 2,
      sciencePoints: 0,
    }) !== null
  );
});

test('300s Sol timeline locks Fast Forward for first 120s and unlocks 5x speed multiplier afterward', () => {
  let state = createInitialMissionState(VERIFIED_FALLBACK_FLARES);
  assert.equal(state.opPhase, 'crisis_brief');

  // 0:00 Early Phase ticks smoothly up to 20s before decision
  state = advanceOperationTick(state, 20);
  assert.equal(state.operationElapsedSec, 20);
  assert.equal(state.fastForwardUnlocked, false);

  // Phase 1: Open decision cards (clock continues advancing smoothly)
  state = openDecisionCards(state);
  assert.equal(state.opPhase, 'decision_cards');
  assert.ok(state.pendingDilemma);
  assert.equal(state.pendingDilemma.choices.length, 2);
  state = advanceOperationTick(state, 10);
  assert.equal(state.operationElapsedSec, 30);
  assert.ok(state.pendingDilemma);

  // Lock in card 2 -> Phase 2 starts WITHOUT snapping operationElapsedSec back to 0
  const choice = state.pendingDilemma.choices[1];
  state = lockInDecisionChoice(state, choice);
  assert.equal(state.opPhase, 'operation');
  assert.equal(state.operationElapsedSec, 30);
  assert.equal(state.decisionElapsedSec, 30);
  assert.equal(state.lockedChoice?.id, choice.id);
  assert.equal(state.fastForwardUnlocked, false);
  assert.equal(state.speedMultiplier, 1);
  assert.ok(state.operationRates);

  // Cannot fast-forward before 120s completes
  const prematureFf = activateFastForward(state);
  assert.equal(prematureFf.fastForwardActive, false);
  assert.equal(prematureFf.speedMultiplier, 1);

  // Advance 89 seconds (30 -> 119s) -> still strictly locked at 1x speed
  const initialExactLife = state.exactResources.lifeSupportPct;
  state = advanceOperationTick(state, 89);
  assert.equal(state.opPhase, 'operation');
  assert.equal(state.operationElapsedSec, 119);
  assert.equal(state.fastForwardUnlocked, false);
  assert.notEqual(state.exactResources.lifeSupportPct, initialExactLife);

  // Module action during Phase 2 updates resources and refreshes rates
  const beforeWall = state.resources.waterWallCm;
  const { nextState: afterModule, result } = applyModuleActionToMission(state, 'shelter');
  assert.equal(result.applied, true);
  assert.equal(afterModule.resources.waterWallCm, beforeWall + 6);
  state = afterModule;

  // Advance 1 more second (119 -> 120s) -> Fast Forward unlocks, Sol still running until 300s
  state = advanceOperationTick(state, 1);
  assert.equal(state.operationElapsedSec, 120);
  assert.equal(state.opPhase, 'operation');
  assert.equal(state.fastForwardUnlocked, true);
  assert.equal(state.speedMultiplier, 1);

  // Toggle Fast Forward -> engages 5x speed multiplier rather than skipping to end
  state = toggleFastForward(state);
  assert.equal(state.fastForwardActive, true);
  assert.equal(state.speedMultiplier, 5);
  assert.equal(state.opPhase, 'operation');

  // 10 real seconds at 5x speed = 50 simulated seconds (120 -> 170s)
  state = advanceOperationTick(state, 10);
  assert.equal(state.operationElapsedSec, 170);
  assert.equal(state.opPhase, 'operation');

  // Remaining 130s takes 26 real seconds at 5x speed -> reaches 300s (sol_complete)
  state = advanceOperationTick(state, 26);
  assert.equal(state.operationElapsedSec, 300);
  assert.equal(state.opPhase, 'sol_complete');

  // Transition to Sol 2 Phase 1
  state = advanceToNextSol(state, VERIFIED_FALLBACK_FLARES);
  assert.equal(state.sol, 2);
  assert.equal(state.opPhase, 'crisis_brief');
  assert.equal(state.operationElapsedSec, 0);
  assert.equal(state.decisionElapsedSec, null);
  assert.equal(state.fastForwardUnlocked, false);
  assert.equal(state.speedMultiplier, 1);
  assert.ok(state.pendingDilemma);
});

test('completing 10 Sols through all 3 phases achieves VICTORY debrief', () => {
  let state = createInitialMissionState(VERIFIED_FALLBACK_FLARES);
  for (let s = 1; s <= MAX_SOLS; s++) {
    assert.equal(state.sol, s);
    state = openDecisionCards(state);
    assert.ok(state.pendingDilemma, `Expected dilemma on Sol ${s}`);

    const sortedByPower = [...state.pendingDilemma.choices].sort(
      (a, b) => b.effects.powerKwh - a.effects.powerKwh
    );
    const choice =
      state.resources.powerKwh < 50 ? sortedByPower[0] : state.pendingDilemma.choices[0];

    state = lockInDecisionChoice(state, choice);
    assert.equal(state.opPhase, 'operation');

    if (state.resources.panelDustPct > 20 && state.resources.powerKwh > 20) {
      state = applyModuleActionToMission(state, 'solar').nextState;
    }
    if (state.resources.lifeSupportPct < 60 && state.resources.powerKwh > 25) {
      state = applyModuleActionToMission(state, 'moxie').nextState;
    }
    if (
      (s === 4 || s === 5 || state.resources.crewHealthPct < 65) &&
      state.resources.waterWallCm < 24 &&
      state.resources.powerKwh > 28
    ) {
      state = applyModuleActionToMission(state, 'shelter').nextState;
    }

    state = advanceOperationTick(state, OPERATION_DURATION_SEC);
    assert.equal(state.opPhase, 'sol_complete');
    state = advanceToNextSol(state, VERIFIED_FALLBACK_FLARES);
  }

  assert.ok(state.debrief);
  assert.equal(state.debrief.outcome, 'VICTORY');
  assert.equal(state.debrief.solsCompleted, 10);
});

test('legacy resolveNightCycleStep and resolveDilemmaChoice remain consistent', () => {
  let state = createInitialMissionState(VERIFIED_FALLBACK_FLARES);
  state = resolveNightCycleStep(state, VERIFIED_FALLBACK_FLARES);
  assert.equal(state.sol, 2);
  assert.ok(state.pendingDilemma);
  state = resolveDilemmaChoice(state, state.pendingDilemma.choices[0], VERIFIED_FALLBACK_FLARES);
  assert.equal(state.sol, 3);
});

test('NASA Data Log archive unlocks progressively by Sol and 3D module interactions', () => {
  const sol1Empty = computeUnlockedLogIds(1, { solar: 0, moxie: 0, shelter: 0, biodome: 0 });
  assert.deepEqual(sol1Empty, ['geo-delta']);

  const sol1Moxie = computeUnlockedLogIds(1, { solar: 0, moxie: 1, shelter: 0, biodome: 0 });
  assert.ok(sol1Moxie.includes('isru-moxie'));
  assert.ok(!sol1Moxie.includes('isru-rad-shield'));

  const sol8Full = computeUnlockedLogIds(8, { solar: 2, moxie: 1, shelter: 2, biodome: 1 });
  assert.equal(sol8Full.length, NASA_ARCHIVE_ENTRIES.length);
});

test('computeTauFogDensity scales monotonically with Tau dust opacity and emergency modes', () => {
  const clearMorning = computeTauFogDensity(0.35, 'none');
  const sol1 = computeTauFogDensity(0.38, 'none');
  const sol3 = computeTauFogDensity(0.72, 'none');
  const sol4Storm = computeTauFogDensity(1.35, 'dust');
  const dustAlertLowTau = computeTauFogDensity(0.44, 'dust');
  const flareAlert = computeTauFogDensity(0.54, 'flare');
  const extremeTau = computeTauFogDensity(5.0, 'dust');

  assert.equal(clearMorning, 0.015);
  assert.ok(sol1 > clearMorning);
  assert.ok(sol3 > sol1);
  assert.ok(sol4Storm > sol3);
  assert.ok(dustAlertLowTau >= 0.026);
  assert.ok(flareAlert >= 0.0205);
  assert.equal(extremeTau, 0.045);
});

test('computeSolAtmosphereTargets shifts smoothly from early buildup to peak and stabilization', () => {
  const t0 = computeSolAtmosphereTargets({
    elapsedSec: 0,
    decisionElapsedSec: null,
    baseTau: 0.72,
    hazardType: 'dust',
    emergencyMode: 'dust',
  });
  const t30Early = computeSolAtmosphereTargets({
    elapsedSec: 30,
    decisionElapsedSec: null,
    baseTau: 0.72,
    hazardType: 'dust',
    emergencyMode: 'dust',
  });
  const tPeak = computeSolAtmosphereTargets({
    elapsedSec: 90,
    decisionElapsedSec: 30,
    baseTau: 0.72,
    hazardType: 'dust',
    emergencyMode: 'dust',
  });
  const tStabilize = computeSolAtmosphereTargets({
    elapsedSec: 255,
    decisionElapsedSec: 30,
    baseTau: 0.72,
    hazardType: 'dust',
    emergencyMode: 'dust',
  });

  assert.equal(t0.solProgress, 0);
  assert.ok(t30Early.crisisIntensity > t0.crisisIntensity);
  assert.ok(tPeak.crisisIntensity >= 0.98);
  assert.ok(tPeak.fogDensity >= t30Early.fogDensity);
  assert.ok(tStabilize.crisisIntensity < tPeak.crisisIntensity);
  assert.ok(tStabilize.nightFactor > t0.nightFactor);
});

test('DONKI severity classification distinguishes quiet/C-class, M-class, and X-class/CME states', () => {
  const cFlare = {
    ...VERIFIED_FALLBACK_FLARES[0],
    classType: 'C4.8',
  };
  const mFlare = VERIFIED_FALLBACK_FLARES[0]; // M4.2

  assert.equal(classifyDonkiSeverity(cFlare, DEFAULT_ARCHIVE_CME, false), 'quiet');
  assert.equal(classifyDonkiSeverity(mFlare, DEFAULT_ARCHIVE_CME, false), 'm_class');
  assert.equal(classifyDonkiSeverity(cFlare, HISTORICAL_X9_CME, false), 'x_class');
  assert.equal(classifyDonkiSeverity(cFlare, DEFAULT_ARCHIVE_CME, true), 'x_class');

  const mState = computeDonkiVisualState(mFlare, DEFAULT_ARCHIVE_CME, false);
  assert.equal(mState.level, 'm_class');
  assert.equal(mState.isSimulatedXFlare, false);
  assert.ok(mState.intensity > 0.45 && mState.intensity <= 0.75);

  const simXState = computeDonkiVisualState(mFlare, DEFAULT_ARCHIVE_CME, true);
  assert.equal(simXState.level, 'x_class');
  assert.equal(simXState.isSimulatedXFlare, true);
  assert.equal(simXState.flare.id, HISTORICAL_X9_FLARE.id);
  assert.equal(simXState.flare.classType, 'X9.3');
  assert.equal(simXState.intensity, 1.0);
});


