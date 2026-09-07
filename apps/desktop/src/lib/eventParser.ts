/**
 * Parser for scanner JSONL event stream with forward compatibility.
 * Safely parses individual lines, validates essential protocol fields,
 * and allows unknown extra fields from future versions.
 */

import { ScannerEvent } from "@/types/scanner";

const VALID_EVENT_TYPES = new Set([
  "scan_plan",
  "file_started",
  "file_completed",
  "file_failed",
  "scan_completed",
]);

/**
 * Safely parse a single line of JSON string into a strongly typed ScannerEvent.
 * Returns null if the line is empty, not valid JSON, or does not conform to event contract.
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

  if (typeof raw.type !== "string" || !VALID_EVENT_TYPES.has(raw.type)) {
    return null;
  }

  // Validate required fields per event type
  switch (raw.type) {
    case "scan_plan": {
      if (
        typeof raw.inputRoot !== "string" ||
        typeof raw.outputRoot !== "string"
      ) {
        return null;
      }
      return raw as unknown as ScannerEvent;
    }

    case "file_started": {
      if (
        typeof raw.relativePath !== "string" ||
        typeof raw.employeeName !== "string" ||
        typeof raw.index !== "number" ||
        typeof raw.total !== "number"
      ) {
        return null;
      }
      return raw as unknown as ScannerEvent;
    }

    case "file_completed": {
      if (
        typeof raw.relativePath !== "string" ||
        typeof raw.employeeName !== "string" ||
        typeof raw.outputRelativePath !== "string"
      ) {
        return null;
      }
      return raw as unknown as ScannerEvent;
    }

    case "file_failed": {
      if (
        typeof raw.relativePath !== "string" ||
        typeof raw.employeeName !== "string" ||
        typeof raw.errorCode !== "string" ||
        typeof raw.message !== "string"
      ) {
        return null;
      }
      return raw as unknown as ScannerEvent;
    }

    case "scan_completed": {
      if (
        typeof raw.totalProcessed !== "number" ||
        typeof raw.success !== "number" ||
        typeof raw.failed !== "number"
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
