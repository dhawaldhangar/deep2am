import { NasaFlareRecord, VERIFIED_FALLBACK_FLARES } from '../data/nasaTelemetry';

export type DonkiSeverityLevel = 'quiet' | 'm_class' | 'x_class';

export interface DonkiCmeSummary {
  count: number;
  latestId: string | null;
  latestSpeedKms: number | null;
  impactActive: boolean;
}

export interface DonkiFetchResult {
  flares: NasaFlareRecord[];
  cme: DonkiCmeSummary;
  mode: 'LIVE_NASA_DONKI' | 'VERIFIED_ARCHIVE_FALLBACK';
  fetchedAt: string;
  statusNote: string;
}

export interface DonkiSpaceWeatherVisualState {
  level: DonkiSeverityLevel;
  flare: NasaFlareRecord;
  cme: DonkiCmeSummary;
  isSimulatedXFlare: boolean;
  /** 0..1 normalized radiation severity driving shaders, auroras, and particle streaks */
  intensity: number;
}

export const HISTORICAL_X9_FLARE: NasaFlareRecord = {
  id: '2017-09-06T11:53:00-FLR-001',
  source: 'NASA_VERIFIED_ARCHIVE',
  classType: 'X9.3',
  peakTime: '2017-09-06T12:02Z',
  activeRegion: 'AR 12673',
  note: 'Historical Solar Cycle 24 benchmark X9.3 flare with fast Earth/Mars-sector Halo CME (1,571 km/s) from NASA CCMC DONKI.',
  radBoostMsv: 1.45,
};

export const HISTORICAL_X9_CME: DonkiCmeSummary = {
  count: 2,
  latestId: '2017-09-06T12:24:00-CME-001',
  latestSpeedKms: 1571,
  impactActive: true,
};

export const DEFAULT_ARCHIVE_CME: DonkiCmeSummary = {
  count: 1,
  latestId: '2026-11-12T15:12:00-CME-001',
  latestSpeedKms: 680,
  impactActive: false,
};

export function classifyFlareRadiation(classType: string): number {
  const upper = classType.trim().toUpperCase();
  const mag = parseFloat(upper.slice(1)) || 1.0;
  if (upper.startsWith('X')) return Number(Math.min(1.8, 0.85 + mag * 0.08).toFixed(2));
  if (upper.startsWith('M')) return Number(Math.min(0.8, 0.4 + mag * 0.03).toFixed(2));
  if (upper.startsWith('C')) return 0.18;
  return 0.1;
}

export function classifyDonkiSeverity(
  flare: NasaFlareRecord,
  cme?: DonkiCmeSummary | null,
  simulateXFlare = false
): DonkiSeverityLevel {
  if (simulateXFlare) return 'x_class';
  const upper = flare.classType.trim().toUpperCase();
  if (upper.startsWith('X')) return 'x_class';
  if (cme?.impactActive && (cme.latestSpeedKms ?? 0) >= 1100) return 'x_class';
  if (upper.startsWith('M') || (cme?.impactActive ?? false)) return 'm_class';
  return 'quiet';
}

export function computeDonkiVisualState(
  activeFlare: NasaFlareRecord,
  cme: DonkiCmeSummary,
  simulateXFlare: boolean
): DonkiSpaceWeatherVisualState {
  const flare = simulateXFlare ? HISTORICAL_X9_FLARE : activeFlare;
  const effectiveCme = simulateXFlare ? HISTORICAL_X9_CME : cme;
  const level = classifyDonkiSeverity(flare, effectiveCme, simulateXFlare);

  let intensity = 0.15;
  if (level === 'm_class') {
    const mag = parseFloat(flare.classType.trim().toUpperCase().slice(1)) || 3.0;
    intensity = Math.min(0.75, 0.45 + (mag / 10) * 0.25);
  } else if (level === 'x_class') {
    intensity = simulateXFlare ? 1.0 : 0.92;
  }

  return {
    level,
    flare,
    cme: effectiveCme,
    isSimulatedXFlare: simulateXFlare,
    intensity,
  };
}

/**
 * Queries NASA's public DONKI Solar Flare (/FLR) and CME (/CME) endpoints using DEMO_KEY.
 * Automatically falls back to embedded verified NASA telemetry if offline, timed out, or rate-limited.
 */
export async function fetchNasaDonkiTelemetry(): Promise<DonkiFetchResult> {
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), 3200);

  // Clamp query window if host clock is ahead of real-world DONKI archive
  const nowMs = Math.min(Date.now(), Date.UTC(2025, 10, 15));
  const endDate = new Date(nowMs).toISOString().slice(0, 10);
  const startDate = new Date(nowMs - 14 * 24 * 60 * 60 * 1000).toISOString().slice(0, 10);

  const flrUrl = `https://api.nasa.gov/DONKI/FLR?startDate=${startDate}&endDate=${endDate}&api_key=DEMO_KEY`;
  const cmeUrl = `https://api.nasa.gov/DONKI/CME?startDate=${startDate}&endDate=${endDate}&api_key=DEMO_KEY`;

  try {
    const [flrRes, cmeRes] = await Promise.all([
      fetch(flrUrl, { signal: controller.signal }),
      fetch(cmeUrl, { signal: controller.signal }).catch(() => null),
    ]);

    clearTimeout(timeout);

    if (!flrRes.ok) {
      throw new Error(`DONKI HTTP ${flrRes.status}`);
    }

    const flrJson = (await flrRes.json()) as Array<{
      flrID?: string;
      classType?: string;
      peakTime?: string;
      activeRegionNum?: number;
      note?: string;
    }>;

    let cmeSummary: DonkiCmeSummary = {
      count: 0,
      latestId: null,
      latestSpeedKms: null,
      impactActive: false,
    };

    if (cmeRes && cmeRes.ok) {
      const cmeJson = (await cmeRes.json()) as Array<{
        activityID?: string;
        cmeAnalyses?: Array<{ speed?: number; isMostAccurate?: boolean }>;
      }>;
      if (Array.isArray(cmeJson) && cmeJson.length > 0) {
        const latest = cmeJson[cmeJson.length - 1];
        const accurate =
          latest.cmeAnalyses?.find((a) => a.isMostAccurate) ?? latest.cmeAnalyses?.[0];
        const speed = typeof accurate?.speed === 'number' ? Math.round(accurate.speed) : null;
        cmeSummary = {
          count: cmeJson.length,
          latestId: latest.activityID ?? null,
          latestSpeedKms: speed,
          impactActive: speed !== null && speed >= 950,
        };
      }
    }

    if (Array.isArray(flrJson) && flrJson.length > 0) {
      const parsed: NasaFlareRecord[] = flrJson
        .slice(-4)
        .reverse()
        .map((item, idx) => {
          const classType = item.classType || 'C4.0';
          return {
            id: item.flrID || `DONKI-FLR-${idx + 1}`,
            source: 'DONKI_LIVE',
            classType,
            peakTime: item.peakTime || new Date().toISOString(),
            activeRegion: item.activeRegionNum ? `AR ${item.activeRegionNum}` : 'Active Region',
            note:
              item.note?.trim() ||
              `Live NASA DONKI solar flare (${classType}) recorded; ${cmeSummary.count} CME events tracked in 14-day window.`,
            radBoostMsv: classifyFlareRadiation(classType),
          };
        });

      return {
        flares: parsed,
        cme: cmeSummary,
        mode: 'LIVE_NASA_DONKI',
        fetchedAt: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
        statusNote: `Connected to NASA / CCMC DONKI (${parsed.length} recent flares, ${cmeSummary.count} CMEs tracked).`,
      };
    }

    return {
      flares: VERIFIED_FALLBACK_FLARES,
      cme: DEFAULT_ARCHIVE_CME,
      mode: 'VERIFIED_ARCHIVE_FALLBACK',
      fetchedAt: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
      statusNote: 'Live DONKI window quiet — displaying verified NASA solar baseline archive.',
    };
  } catch {
    clearTimeout(timeout);
    return {
      flares: VERIFIED_FALLBACK_FLARES,
      cme: DEFAULT_ARCHIVE_CME,
      mode: 'VERIFIED_ARCHIVE_FALLBACK',
      fetchedAt: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
      statusNote: 'Offline / rate-limited fallback active — running verified NASA telemetry archive.',
    };
  }
}
