import { useEffect, useState } from "react";
import { ArrowDown, ArrowUp, Eye, SkipForward, ShieldAlert } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import type { ReviewGroup } from "@/types/scanner";

export interface PageOrderReviewPanelProps {
  groups: ReviewGroup[];
  onResolve: (groupId: string, orderedPaths: string[]) => void;
  onSkip: (groupId: string) => void;
  onPreview?: (sourcePath: string) => void;
  resolvedGroups?: Record<string, "resolved" | "skipped">;
}

const REASON_LABELS: Record<string, string> = {
  unknown_page_identity: "Không nhận diện chắc chắn loại page",
  duplicate_page_order: "Trùng thứ tự page được nhận diện",
  duplicate_or_missing_page_order: "Thiếu hoặc trùng thứ tự page",
  missing_expected_page: "Thiếu page bắt buộc (Page 1 hoặc Page 2)",
  source_removed: "Một source page đã bị xóa",
  manual_order_invalidated: "Source đã thay đổi sau manual override",
  missing_grouped_output: "Thiếu grouped PDF cần dựng lại",
  grouped_artifact_stale: "Grouped PDF cũ không còn hợp lệ sau lỗi rebuild",
  source_new: "Có source page mới trong group",
  source_modified: "Có source page đã thay đổi",
  source_rebuild: "Một source cần rebuild",
};

function groupId(group: ReviewGroup): string {
  return `${group.key.employeeRelativeDir}:${group.key.year}-${String(group.key.month).padStart(2, "0")}`;
}

function reasonLabel(reason: string): string {
  return REASON_LABELS[reason] ?? reason;
}

function orderedPaths(group: ReviewGroup): string[] {
  return group.manualOrder.length > 0
    ? group.manualOrder
    : group.sourcePages.map((page) => page.sourceRelativePath);
}

export function PageOrderReviewPanel({
  groups,
  onResolve,
  onSkip,
  onPreview,
  resolvedGroups = {},
}: PageOrderReviewPanelProps) {
  const [workingOrders, setWorkingOrders] = useState<Record<string, string[]>>({});

  useEffect(() => {
    const next: Record<string, string[]> = {};
    for (const group of groups) {
      next[groupId(group)] = orderedPaths(group);
    }
    setWorkingOrders(next);
  }, [groups]);

  if (groups.length === 0) {
    return null;
  }

  return (
    <section
      aria-labelledby="page-order-review-heading"
      className="rounded-2xl border border-amber-500/30 bg-amber-500/10 p-4 sm:p-5"
    >
      <div className="flex items-start gap-3">
        <ShieldAlert className="mt-0.5 h-5 w-5 shrink-0 text-amber-300" />
        <div className="min-w-0">
          <h2 id="page-order-review-heading" className="text-base font-semibold text-amber-100">
            Cần xác nhận thứ tự page trước khi ghép PDF
          </h2>
          <p className="mt-1 text-xs leading-relaxed text-amber-200/80">
            Grouped export không tự đoán khi classifier không đủ chắc chắn. Bạn có thể dùng select hoặc nút di chuyển bằng bàn phím.
          </p>
        </div>
      </div>

      <div className="mt-4 space-y-3">
        {groups.map((group) => {
          const id = groupId(group);
          const status = resolvedGroups[id];
          const paths = workingOrders[id] ?? orderedPaths(group);
          const blockedByIncomplete = group.completenessStatus === "INCOMPLETE";

          return (
            <article key={id} className="rounded-xl border border-border/70 bg-background/50 p-3">
              <div className="flex flex-col gap-2 sm:flex-row sm:items-center sm:justify-between">
                <div>
                  <p className="font-semibold text-foreground">
                    {group.key.employeeRelativeDir}
                    <span className="ml-2 font-mono text-xs text-muted-foreground">
                      {group.key.year}-{String(group.key.month).padStart(2, "0")}
                    </span>
                  </p>
                  <div className="mt-1 flex flex-wrap gap-1.5">
                    {group.reasons.map((reason) => (
                      <Badge key={reason} variant="warning" className="text-[10px]">
                        {reasonLabel(reason)}
                      </Badge>
                    ))}
                  </div>
                </div>
                {status && (
                  <Badge variant={status === "resolved" ? "success" : "secondary"}>
                    {status === "resolved" ? "Đã xác nhận" : "Đã bỏ qua"}
                  </Badge>
                )}
              </div>

              <div className="mt-3 space-y-2">
                {paths.map((path, index) => {
                  const sourcePage = group.sourcePages.find((page) => page.sourceRelativePath === path);
                  return (
                    <div key={path} className="flex flex-col gap-2 rounded-lg border border-border/60 bg-card/60 p-2 sm:flex-row sm:items-center">
                      <div
                        className="flex min-w-0 flex-1 items-center gap-2"
                        aria-label={`Preview ${path}`}
                      >
                        {onPreview ? (
                          <Button
                            type="button"
                            variant="ghost"
                            size="sm"
                            aria-label={`Xem preview ${path}`}
                            onClick={() => onPreview(path)}
                            className="h-10 w-10 shrink-0 p-0"
                          >
                            <Eye className="h-4 w-4 text-primary" />
                          </Button>
                        ) : (
                          <div className="flex h-10 w-10 shrink-0 items-center justify-center rounded-md bg-primary/10">
                            <Eye className="h-4 w-4 text-primary" />
                          </div>
                        )}
                        <div className="min-w-0">
                          <p className="truncate font-mono text-xs text-foreground" title={path}>
                            {path}
                          </p>
                          <p className="text-[11px] text-muted-foreground">
                            {sourcePage?.identity.pageType ?? "UNKNOWN"}
                            {sourcePage?.identity.confidence != null
                              ? ` • confidence ${Math.round(sourcePage.identity.confidence * 100)}%`
                              : " • cần xác nhận"}
                          </p>
                        </div>
                      </div>
                      <div className="flex items-center gap-1.5">
                        <label htmlFor={`${id}-${index}`} className="sr-only">
                          Thứ tự page của {path}
                        </label>
                        <select
                          id={`${id}-${index}`}
                          value={index + 1}
                          onChange={(event) => {
                            const nextIndex = Number(event.target.value) - 1;
                            const nextPaths = [...paths];
                            const [moved] = nextPaths.splice(index, 1);
                            nextPaths.splice(nextIndex, 0, moved);
                            setWorkingOrders((current) => ({ ...current, [id]: nextPaths }));
                          }}
                          disabled={Boolean(status) || blockedByIncomplete}
                          className="min-h-[40px] rounded-lg border border-border bg-background px-2 text-xs text-foreground focus:outline-none focus:ring-2 focus:ring-ring"
                        >
                          {paths.map((_, order) => (
                            <option key={order} value={order + 1}>
                              Page {order + 1}
                            </option>
                          ))}
                        </select>
                        <Button
                          type="button"
                          variant="ghost"
                          size="sm"
                          aria-label={`Đưa ${path} lên trước`}
                          onClick={() => {
                            if (index === 0) return;
                            const nextPaths = [...paths];
                            [nextPaths[index - 1], nextPaths[index]] = [
                              nextPaths[index],
                              nextPaths[index - 1],
                            ];
                            setWorkingOrders((current) => ({ ...current, [id]: nextPaths }));
                          }}
                          disabled={Boolean(status) || blockedByIncomplete || index === 0}
                        >
                          <ArrowUp className="h-4 w-4" />
                        </Button>
                        <Button
                          type="button"
                          variant="ghost"
                          size="sm"
                          aria-label={`Đưa ${path} xuống sau`}
                          onClick={() => {
                            if (index === paths.length - 1) return;
                            const nextPaths = [...paths];
                            [nextPaths[index], nextPaths[index + 1]] = [
                              nextPaths[index + 1],
                              nextPaths[index],
                            ];
                            setWorkingOrders((current) => ({ ...current, [id]: nextPaths }));
                          }}
                          disabled={Boolean(status) || blockedByIncomplete || index === paths.length - 1}
                        >
                          <ArrowDown className="h-4 w-4" />
                        </Button>
                      </div>
                    </div>
                  );
                })}
              </div>

              <div className="mt-3 flex flex-col gap-2 sm:flex-row sm:justify-end">
                <Button
                  type="button"
                  variant="ghost"
                  size="sm"
                  onClick={() => onSkip(id)}
                  disabled={Boolean(status)}
                  className="w-full sm:w-auto"
                >
                  <SkipForward className="mr-2 h-4 w-4" />
                  Bỏ qua group này
                </Button>
                <Button
                  type="button"
                  variant="default"
                  size="sm"
                  onClick={() => onResolve(id, paths)}
                  disabled={Boolean(status) || blockedByIncomplete || paths.length < 2}
                  className="w-full sm:w-auto"
                >
                  Xác nhận thứ tự page
                </Button>
              </div>
              {blockedByIncomplete && (
                <p className="mt-2 text-right text-xs text-amber-200/80">
                  Group incomplete; hãy bỏ qua hoặc bổ sung source trước khi ghép.
                </p>
              )}
            </article>
          );
        })}
      </div>
    </section>
  );
}
