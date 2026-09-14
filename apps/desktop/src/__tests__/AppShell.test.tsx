import { act, fireEvent, render, screen } from "@testing-library/react";
import "@testing-library/jest-dom/vitest";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { AppShell } from "@/components/scanner/AppShell";
import { useExecutionCoordinator } from "@/components/scanner/executionCoordinator";
import { loadWorkflowPreference, WORKFLOW_PREFERENCE_STORAGE_KEY } from "@/lib/userPreferences";

vi.mock("@/lib/scannerBridge", () => ({
  listenScannerEvents: vi.fn().mockResolvedValue(() => undefined),
  listenScannerDiagnostics: vi.fn().mockResolvedValue(() => undefined),
  planScan: vi.fn(),
  startScan: vi.fn(),
  scannerErrorMessage: () => "Sidecar unavailable",
  scannerDiagnosticMessage: () => "Scanner diagnostic",
}));

function CoordinatorProbe() {
  const { beginExecution, endExecution, isAnyExecuting } = useExecutionCoordinator();
  return (
    <div>
      <button type="button" onClick={() => beginExecution("quick_scan")}>
        begin
      </button>
      <button type="button" onClick={() => endExecution("quick_scan")}>
        end
      </button>
      <span>{isAnyExecuting ? "busy" : "idle"}</span>
    </div>
  );
}

describe("AppShell execution coordinator", () => {
  beforeEach(() => {
    window.localStorage.clear();
  });

  it("restores the workflow preference and keeps all workflow views mounted", () => {
    window.localStorage.setItem(WORKFLOW_PREFERENCE_STORAGE_KEY, JSON.stringify("quick_scan"));

    const { container } = render(<AppShell />);

    expect(loadWorkflowPreference()).toBe("quick_scan");
    expect(screen.getByRole("heading", { name: "Quét nhanh một ảnh" })).toBeInTheDocument();
    expect(screen.getByText("Quét thư mục tự do")).toBeInTheDocument();
    expect(container.querySelector('[hidden]')).toBeInTheDocument();
  });

  it("persists workflow selection and routes navigation through the shell", () => {
    render(<AppShell />);

    fireEvent.click(screen.getByRole("tab", { name: "Thư mục tự do" }));

    expect(screen.getByRole("heading", { name: "Quét thư mục tự do" })).toBeInTheDocument();
    expect(loadWorkflowPreference()).toBe("folder_scan");
  });

  it("exposes a single execution lock to workflow consumers", () => {
    render(
      <AppShell>
        <CoordinatorProbe />
      </AppShell>,
    );

    const probe = screen.getByRole("button", { name: "begin" });
    act(() => {
      fireEvent.click(probe);
    });

    expect(screen.getByText("busy")).toBeInTheDocument();
    expect(screen.getByRole("tab", { name: "Quét nhanh" })).toBeDisabled();

    fireEvent.click(screen.getByRole("button", { name: "end" }));
    expect(screen.getByText("idle")).toBeInTheDocument();
    expect(screen.getByRole("tab", { name: "Quét nhanh" })).toBeEnabled();
  });
});
