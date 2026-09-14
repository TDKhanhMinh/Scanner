import { useCallback, useEffect, useRef, useState } from "react";
import type { ReactNode } from "react";
import { AttendanceBatchView } from "@/components/scanner/AttendanceBatchView";
import { FolderScanView } from "@/components/scanner/FolderScanView";
import { QuickScanView } from "@/components/scanner/QuickScanView";
import {
  listenScannerDiagnostics,
  listenScannerEvents,
} from "@/lib/scannerBridge";
import {
  DEFAULT_WORKFLOW_MODE,
  loadWorkflowPreference,
  saveWorkflowPreference,
} from "@/lib/userPreferences";
import type { ScannerEvent, WorkflowMode } from "@/types/scanner";
import {
  ExecutionCoordinatorContext,
  type ExecutionCoordinatorValue,
  type WorkflowDiagnosticHandler,
  type WorkflowEventHandler,
} from "@/components/scanner/executionCoordinator";

function workflowForEvent(event: ScannerEvent): WorkflowMode {
  if (event.type.startsWith("quick_")) return "quick_scan";
  if (event.type.startsWith("flat_")) return "folder_scan";
  return "attendance_batch";
}

interface AppShellProps {
  children?: ReactNode;
}

export function AppShell({ children }: AppShellProps) {
  const [activeWorkflow, setActiveWorkflow] = useState<WorkflowMode>(() => {
    return loadWorkflowPreference() || DEFAULT_WORKFLOW_MODE;
  });
  const [executingWorkflow, setExecutingWorkflow] = useState<WorkflowMode | null>(null);
  const executingWorkflowRef = useRef<WorkflowMode | null>(null);
  const eventHandlersRef = useRef<Map<WorkflowMode, Set<WorkflowEventHandler>>>(new Map());
  const diagnosticHandlersRef = useRef<Map<WorkflowMode, Set<WorkflowDiagnosticHandler>>>(
    new Map(),
  );

  const selectWorkflow = useCallback((workflow: WorkflowMode) => {
    if (executingWorkflowRef.current) return;
    setActiveWorkflow(workflow);
    saveWorkflowPreference(workflow);
  }, []);

  const beginExecution = useCallback((workflow: WorkflowMode): boolean => {
    if (executingWorkflowRef.current) return false;
    executingWorkflowRef.current = workflow;
    setExecutingWorkflow(workflow);
    return true;
  }, []);

  const endExecution = useCallback((workflow: WorkflowMode) => {
    if (executingWorkflowRef.current !== workflow) return;
    executingWorkflowRef.current = null;
    setExecutingWorkflow(null);
  }, []);

  const registerEventHandler = useCallback(
    (workflow: WorkflowMode, handler: WorkflowEventHandler) => {
      const handlers = eventHandlersRef.current.get(workflow) ?? new Set<WorkflowEventHandler>();
      handlers.add(handler);
      eventHandlersRef.current.set(workflow, handlers);
      return () => {
        handlers.delete(handler);
        if (handlers.size === 0) eventHandlersRef.current.delete(workflow);
      };
    },
    [],
  );

  const registerDiagnosticHandler = useCallback(
    (workflow: WorkflowMode, handler: WorkflowDiagnosticHandler) => {
      const handlers =
        diagnosticHandlersRef.current.get(workflow) ?? new Set<WorkflowDiagnosticHandler>();
      handlers.add(handler);
      diagnosticHandlersRef.current.set(workflow, handlers);
      return () => {
        handlers.delete(handler);
        if (handlers.size === 0) diagnosticHandlersRef.current.delete(workflow);
      };
    },
    [],
  );

  useEffect(() => {
    let mounted = true;
    let unlistenEvents: (() => void) | undefined;
    let unlistenDiagnostics: (() => void) | undefined;

    try {
      void Promise.resolve(
        listenScannerEvents((event) => {
          const workflow = workflowForEvent(event);
          eventHandlersRef.current.get(workflow)?.forEach((handler) => handler(event));
        }),
      )
        .then((cleanup) => {
          if (typeof cleanup !== "function") return;
          if (mounted) unlistenEvents = cleanup;
          else cleanup();
        })
        .catch(() => undefined);
    } catch {
      // Browser-based previews/tests do not expose Tauri event internals.
    }

    try {
      void Promise.resolve(
        listenScannerDiagnostics((diagnostic) => {
          const workflow = executingWorkflowRef.current;
          if (!workflow) return;
          diagnosticHandlersRef.current.get(workflow)?.forEach((handler) => handler(diagnostic));
        }),
      )
        .then((cleanup) => {
          if (typeof cleanup !== "function") return;
          if (mounted) unlistenDiagnostics = cleanup;
          else cleanup();
        })
        .catch(() => undefined);
    } catch {
      // Browser-based previews/tests do not expose Tauri event internals.
    }

    return () => {
      mounted = false;
      unlistenEvents?.();
      unlistenDiagnostics?.();
    };
  }, []);

  const value: ExecutionCoordinatorValue = {
    activeWorkflow,
    isAnyExecuting: executingWorkflow !== null,
    selectWorkflow,
    beginExecution,
    endExecution,
    registerEventHandler,
    registerDiagnosticHandler,
  };

  return (
    <ExecutionCoordinatorContext.Provider value={value}>
      <div className="min-h-screen bg-background text-foreground">
        <nav className="border-b border-border/80 bg-background/95 px-4 py-3">
          <div className="mx-auto flex max-w-7xl gap-2" role="tablist" aria-label="Workflow">
            {(
              [
                ["attendance_batch", "Hồ sơ nhân sự"],
                ["folder_scan", "Thư mục tự do"],
                ["quick_scan", "Quét nhanh"],
              ] as const
            ).map(([workflow, label]) => (
              <button
                key={workflow}
                type="button"
                role="tab"
                aria-selected={activeWorkflow === workflow}
                disabled={value.isAnyExecuting}
                onClick={() => selectWorkflow(workflow)}
                className={`min-h-11 rounded-lg px-4 text-sm font-medium transition-colors ${
                  activeWorkflow === workflow
                    ? "bg-primary text-primary-foreground"
                    : "bg-secondary/50 text-muted-foreground hover:bg-secondary"
                } disabled:cursor-not-allowed disabled:opacity-50`}
              >
                {label}
              </button>
            ))}
          </div>
        </nav>

        <div hidden={activeWorkflow !== "attendance_batch"}>
          <AttendanceBatchView />
        </div>
        <div hidden={activeWorkflow !== "folder_scan"}>
          <FolderScanView />
        </div>
        <div hidden={activeWorkflow !== "quick_scan"}>
          <QuickScanView />
        </div>
        {children}
      </div>
    </ExecutionCoordinatorContext.Provider>
  );
}

export default AppShell;
