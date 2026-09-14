import type { ProductDetectorMode } from "@/types/scanner";
import {
  DEFAULT_USER_PREFERENCES,
  loadUserPreferences,
  saveUserPreferences,
} from "./userPreferences";

export interface DetectorPreferences {
  detectorMode: ProductDetectorMode;
  debugDiagnostics: boolean;
}

export const DEFAULT_DETECTOR_PREFERENCES: DetectorPreferences = {
  detectorMode: DEFAULT_USER_PREFERENCES.detectorMode,
  debugDiagnostics: DEFAULT_USER_PREFERENCES.debugDiagnostics,
};

export function loadDetectorPreferences(): DetectorPreferences {
  const preferences = loadUserPreferences();
  return {
    detectorMode: preferences.detectorMode,
    debugDiagnostics: preferences.debugDiagnostics,
  };
}

export function saveDetectorPreferences(preferences: DetectorPreferences): void {
  saveUserPreferences({
    detectorMode: preferences.detectorMode,
    debugDiagnostics: preferences.debugDiagnostics,
  });
}
