import { createContext, useContext } from "react";
import type { ScannerEvent, WorkflowMode } from "@/types/scanner";
import type { ScannerDiagnostic } from "@/lib/scannerBridge";

export type WorkflowEventHandler = (event: ScannerEvent) => void;
export type WorkflowDiagnosticHandler = (diagnostic: ScannerDiagnostic) => void;

export interface ExecutionCoordinatorValue {
  activeWorkflow: WorkflowMode;
  isAnyExecuting: boolean;
  selectWorkflow: (workflow: WorkflowMode) => void;
  beginExecution: (workflow: WorkflowMode) => boolean;
  endExecution: (workflow: WorkflowMode) => void;
  registerEventHandler: (workflow: WorkflowMode, handler: WorkflowEventHandler) => () => void;
  registerDiagnosticHandler: (
    workflow: WorkflowMode,
    handler: WorkflowDiagnosticHandler,
  ) => () => void;
}

export const ExecutionCoordinatorContext = createContext<ExecutionCoordinatorValue | null>(null);

export function useExecutionCoordinator(): ExecutionCoordinatorValue {
  const context = useContext(ExecutionCoordinatorContext);
  if (!context) {
    throw new Error("useExecutionCoordinator must be used inside AppShell");
  }
  return context;
}
