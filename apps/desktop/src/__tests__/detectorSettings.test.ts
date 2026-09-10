import { beforeEach, describe, expect, it } from "vitest";
import {
  DEFAULT_DETECTOR_PREFERENCES,
  loadDetectorPreferences,
  saveDetectorPreferences,
} from "@/lib/detectorSettings";

describe("detector settings persistence", () => {
  beforeEach(() => {
    window.localStorage.clear();
  });

  it("uses AI Enhanced and diagnostics off by default", () => {
    expect(loadDetectorPreferences()).toEqual(DEFAULT_DETECTOR_PREFERENCES);
  });

  it("round-trips only the documented product preferences", () => {
    saveDetectorPreferences({ detectorMode: "classic", debugDiagnostics: true });
    expect(loadDetectorPreferences()).toEqual({
      detectorMode: "classic",
      debugDiagnostics: true,
    });
  });

  it("recovers safely from malformed or dev-only stored values", () => {
    window.localStorage.setItem(
      "attendance-scanner.detector-settings.v1",
      JSON.stringify({ detectorMode: "hybrid", debugDiagnostics: "yes" }),
    );
    expect(loadDetectorPreferences()).toEqual(DEFAULT_DETECTOR_PREFERENCES);
  });
});
