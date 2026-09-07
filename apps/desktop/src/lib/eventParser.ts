/**
 * Parser for scanner JSONL event stream with forward compatibility and strict wire validation.
 * Enforces protocolVersion, timestamp, valid error codes, and complete counter fields.
 */

import {
  PROTOCOL_VERSION,
  ScannerEvent,
  VALID_SCANNER_ERROR_CODES,
} from "@/types/scanner";

const VALID_EVENT_TYPES = new Set([
  "scan_plan",
  "file_started",
  "file_completed",
  "file_failed",
  "scan_completed",
]);

/**
 * Safely parse a single line of JSON string into a strongly typed ScannerEvent.
 * Returns null if the line is empty, not valid JSON, missing protocolVersion/timestamp,
 * or does not conform to event contract and valid enum values.
 */
export function parseScannerEvent(line: string): ScannerEvent | null {
  const trimmed = line.trim();
  if (!trimmed) {
    return null;
  }

  let parsed: unknown;
  try {
    parsed = JSON.parse(trimmed);
  } catch {
    return null;
  }

  if (
    typeof parsed !== "object" ||
    parsed === null ||
    Array.isArray(parsed)
  ) {
    return null;
  }

  const raw = parsed as Record<string, unknown>;

  // P2: Strict protocolVersion validation
  if (
    typeof raw.protocolVersion !== "number" ||
    raw.protocolVersion !== PROTOCOL_VERSION
  ) {
    return null;
  }

  // P2: Strict timestamp validation
  if (
    typeof raw.timestamp !== "string" ||
    raw.timestamp.trim().length === 0
  ) {
    return null;
  }

  if (typeof raw.type !== "string" || !VALID_EVENT_TYPES.has(raw.type)) {
    return null;
  }

  // Validate required fields per event type
  switch (raw.type) {
    case "scan_plan": {
      // P1: Support both System Design canonical fields (employees, new, modified, unchanged)
      // and backward-compatible aliases (totalEmployees, newCount, modifiedCount, unchangedCount)
      const employees =
        typeof raw.employees === "number"
          ? raw.employees
          : typeof raw.totalEmployees === "number"
          ? raw.totalEmployees
          : undefined;

      const totalImages =
        typeof raw.totalImages === "number" ? raw.totalImages : undefined;

      const newCount =
        typeof raw.new === "number"
          ? raw.new
          : typeof raw.newCount === "number"
          ? raw.newCount
          : undefined;

      const modified =
        typeof raw.modified === "number"
          ? raw.modified
          : typeof raw.modifiedCount === "number"
          ? raw.modifiedCount
          : undefined;

      const rebuild =
        typeof raw.rebuild === "number"
          ? raw.rebuild
          : typeof raw.rebuildCount === "number"
          ? raw.rebuildCount
          : 0;

      const unchanged =
        typeof raw.unchanged === "number"
          ? raw.unchanged
          : typeof raw.unchangedCount === "number"
          ? raw.unchangedCount
          : undefined;

      if (
        typeof raw.inputRoot !== "string" ||
        raw.inputRoot.trim().length === 0 ||
        typeof raw.outputRoot !== "string" ||
        raw.outputRoot.trim().length === 0 ||
        employees === undefined ||
        employees < 0 ||
        totalImages === undefined ||
        totalImages < 0 ||
        newCount === undefined ||
        newCount < 0 ||
        modified === undefined ||
        modified < 0 ||
        rebuild < 0 ||
        unchanged === undefined ||
        unchanged < 0
      ) {
        return null;
      }

      const expectedFilesToProcess = newCount + modified + rebuild;
      if (
        raw.filesToProcess !== undefined &&
        (typeof raw.filesToProcess !== "number" ||
          raw.filesToProcess < 0 ||
          raw.filesToProcess !== expectedFilesToProcess)
      ) {
        return null;
      }

      // Ensure both canonical and legacy alias fields exist for UI consumption
      raw.employees = employees;
      raw.totalEmployees = employees;
      raw.totalImages = totalImages;
      raw.new = newCount;
      raw.newCount = newCount;
      raw.modified = modified;
      raw.modifiedCount = modified;
      raw.rebuild = rebuild;
      raw.rebuildCount = rebuild;
      raw.unchanged = unchanged;
      raw.unchangedCount = unchanged;
      raw.filesToProcess = expectedFilesToProcess;
      raw.outdatedPipelineCount =
        typeof raw.outdatedPipelineCount === "number" &&
        raw.outdatedPipelineCount >= 0
          ? raw.outdatedPipelineCount
          : 0;
      raw.collisions = Array.isArray(raw.collisions) ? raw.collisions : [];

      return raw as unknown as ScannerEvent;
    }

    case "file_started": {
      if (
        typeof raw.relativePath !== "string" ||
        raw.relativePath.trim().length === 0 ||
        typeof raw.employeeName !== "string" ||
        raw.employeeName.trim().length === 0 ||
        typeof raw.index !== "number" ||
        raw.index < 0 ||
        typeof raw.total !== "number" ||
        raw.total < 0
      ) {
        return null;
      }
      return raw as unknown as ScannerEvent;
    }

    case "file_completed": {
      if (
        typeof raw.relativePath !== "string" ||
        raw.relativePath.trim().length === 0 ||
        typeof raw.employeeName !== "string" ||
        raw.employeeName.trim().length === 0 ||
        typeof raw.outputRelativePath !== "string" ||
        raw.outputRelativePath.trim().length === 0 ||
        typeof raw.documentDetected !== "boolean" ||
        typeof raw.durationMs !== "number" ||
        raw.durationMs < 0
      ) {
        return null;
      }
      if (
        raw.warning !== null &&
        raw.warning !== undefined &&
        typeof raw.warning !== "string"
      ) {
        return null;
      }
      return raw as unknown as ScannerEvent;
    }

    case "file_failed": {
      if (
        typeof raw.relativePath !== "string" ||
        raw.relativePath.trim().length === 0 ||
        typeof raw.employeeName !== "string" ||
        raw.employeeName.trim().length === 0 ||
        typeof raw.errorCode !== "string" ||
        !VALID_SCANNER_ERROR_CODES.has(raw.errorCode) || // P2: Strict enum validation
        typeof raw.message !== "string" ||
        raw.message.trim().length === 0
      ) {
        return null;
      }
      return raw as unknown as ScannerEvent;
    }

    case "scan_completed": {
      if (
        typeof raw.totalProcessed !== "number" ||
        raw.totalProcessed < 0 ||
        typeof raw.success !== "number" ||
        raw.success < 0 ||
        typeof raw.failed !== "number" ||
        raw.failed < 0 ||
        typeof raw.warning !== "number" ||
        raw.warning < 0 ||
        typeof raw.skipped !== "number" ||
        raw.skipped < 0 ||
        typeof raw.durationMs !== "number" ||
        raw.durationMs < 0
      ) {
        return null;
      }
      return raw as unknown as ScannerEvent;
    }

    default:
      return null;
  }
}

/**
 * Parses multiple lines of JSONL text into an array of valid events.
 * Invalid or empty lines are silently skipped.
 */
export function parseScannerEvents(lines: string[]): ScannerEvent[] {
  const results: ScannerEvent[] = [];
  for (const line of lines) {
    const event = parseScannerEvent(line);
    if (event !== null) {
      results.push(event);
    }
  }
  return results;
}

/**
 * Handles incoming stream chunks that might contain incomplete lines at the boundary.
 * Returns parsed events and leftover incomplete buffer.
 */
export function parseEventChunk(
  chunk: string,
  bufferedRemainder = ""
): { events: ScannerEvent[]; remainder: string } {
  const combined = bufferedRemainder + chunk;
  const lines = combined.split("\n");

  // The last element is either empty (if ending in \n) or incomplete
  const remainder = lines.pop() ?? "";
  const events = parseScannerEvents(lines);

  return { events, remainder };
}
