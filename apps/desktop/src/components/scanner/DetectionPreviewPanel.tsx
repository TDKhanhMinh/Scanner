import { AlertTriangle, Bug, CheckCircle2, Eye, X } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { formatDetectionReasonMessage, type FileResultItem } from "@/components/scanner/FileResultList";
import type { DetectionPreview, PreviewPoint } from "@/types/scanner";

export interface DetectionPreviewPanelProps {
  item: FileResultItem;
  onClose: () => void;
}

export function previewPointToSvgPoint(
  point: PreviewPoint,
  preview: Pick<DetectionPreview, "sourceWidth" | "sourceHeight">,
): string {
  const x = Math.max(0, Math.min(100, (point.x / preview.sourceWidth) * 100));
  const y = Math.max(0, Math.min(100, (point.y / preview.sourceHeight) * 100));
  return `${x.toFixed(3)},${y.toFixed(3)}`;
}

function polygonPoints(points: PreviewPoint[], preview: DetectionPreview): string {
  return points.map((point) => previewPointToSvgPoint(point, preview)).join(" ");
}

function confidenceLabel(confidence?: number | null): string {
  return confidence === null || confidence === undefined
    ? "Chưa có"
    : `${Math.round(confidence * 100)} / 100 điểm`;
}

export function DetectionPreviewPanel({
  item,
  onClose,
}: DetectionPreviewPanelProps) {
  const preview = item.detectionPreview;
  const hasFinalCorners = Boolean(preview?.finalCorners?.length === 4);
  const candidateCorners = preview?.candidateCorners ?? [];
  const reasonMessage = formatDetectionReasonMessage(
    item.detectionReason,
    item.detectionReasonCodes,
  );

  return (
    <Card className="border-primary/30 bg-card/80 shadow-sm">
      <CardHeader className="flex-row items-start justify-between gap-3 pb-3">
        <div className="min-w-0">
          <CardTitle className="flex items-center gap-2 text-base">
            <Eye className="h-4 w-4 text-primary" />
            Preview nhận diện: {item.sourceFile}
          </CardTitle>
          <p className="mt-1 truncate font-mono text-xs text-muted-foreground" title={item.relativePath}>
            {item.relativePath}
          </p>
        </div>
        <Button type="button" variant="ghost" size="sm" onClick={onClose} aria-label="Đóng preview">
          <X className="h-4 w-4" />
        </Button>
      </CardHeader>

      <CardContent className="space-y-4">
        {preview?.previewImageDataUrl ? (
          <div
            className="relative mx-auto w-full max-w-3xl overflow-hidden rounded-xl border border-border/70 bg-black/20"
            style={{ aspectRatio: `${preview.sourceWidth} / ${preview.sourceHeight}` }}
          >
            <img
              src={preview.previewImageDataUrl}
              alt={`Ảnh nguồn của ${item.sourceFile}`}
              className="absolute inset-0 h-full w-full object-fill"
              loading="lazy"
              decoding="async"
            />
            <svg
              aria-label="Lớp phủ các góc nhận diện"
              className="absolute inset-0 h-full w-full"
              viewBox="0 0 100 100"
              preserveAspectRatio="none"
              role="img"
            >
              {preview.maskOverlayUrl?.startsWith("data:image/") && (
                <image href={preview.maskOverlayUrl} x="0" y="0" width="100" height="100" opacity="0.24" />
              )}
              {candidateCorners.map((candidate, index) => (
                <polygon
                  key={`${candidate.source}-${index}`}
                  points={polygonPoints(candidate.corners, preview)}
                  fill="none"
                  stroke="#f59e0b"
                  strokeDasharray="2 1"
                  strokeWidth="0.8"
                  vectorEffect="non-scaling-stroke"
                />
              ))}
              {preview.refinedCorners && (
                <polygon
                  points={polygonPoints(preview.refinedCorners, preview)}
                  fill="none"
                  stroke="#38bdf8"
                  strokeWidth="1.2"
                  vectorEffect="non-scaling-stroke"
                />
              )}
              {preview.finalCorners && (
                <polygon
                  points={polygonPoints(preview.finalCorners, preview)}
                  fill="rgba(16, 185, 129, 0.12)"
                  stroke="#10b981"
                  strokeWidth="1.4"
                  vectorEffect="non-scaling-stroke"
                />
              )}
            </svg>
          </div>
        ) : (
          <div className="flex items-start gap-2 rounded-xl border border-dashed border-amber-500/30 bg-amber-500/10 p-4 text-xs text-amber-900 dark:text-amber-300">
            <Bug className="mt-0.5 h-4 w-4 shrink-0" />
            <span>File này chưa có ảnh preview bounded. Bật diagnostics rồi chạy lại nếu cần xem candidate.</span>
          </div>
        )}

        <div className="grid grid-cols-1 gap-2 text-xs sm:grid-cols-2">
          <div className="rounded-lg border border-border/60 bg-secondary/30 p-3">
            <span className="block text-muted-foreground">Trạng thái crop</span>
            <span className="mt-1 flex items-center gap-1.5 font-medium text-foreground">
              {hasFinalCorners ? (
                <><CheckCircle2 className="h-3.5 w-3.5 text-emerald-400" /> Đã chọn 4 góc cuối</>
              ) : (
                <><AlertTriangle className="h-3.5 w-3.5 text-amber-400" /> Fallback ảnh đầy đủ, không auto-crop</>
              )}
            </span>
          </div>
          <div className="rounded-lg border border-border/60 bg-secondary/30 p-3">
            <span className="block text-muted-foreground">Điểm chất lượng (không phải xác suất)</span>
            <span className="mt-1 font-mono font-medium text-foreground">
              {confidenceLabel(preview?.confidence)}
            </span>
          </div>
          {preview && (
            <>
              <div className="rounded-lg border border-border/60 bg-secondary/30 p-3">
                <span className="block text-muted-foreground">Detector</span>
                <span className="mt-1 font-mono font-medium text-foreground">
                  {preview.detectorName ?? "Không xác định"}
                  {preview.modelVersion ? ` · ${preview.modelVersion}` : ""}
                </span>
              </div>
              <div className="rounded-lg border border-border/60 bg-secondary/30 p-3">
                <span className="block text-muted-foreground">Candidate / mask</span>
                <span className="mt-1 font-medium text-foreground">
                  {candidateCorners.length} candidate
                  {preview.maskAvailable ? " · mask evidence có sẵn" : " · chưa có mask preview"}
                </span>
              </div>
            </>
          )}
        </div>

        {(preview?.fallbackUsed || reasonMessage) && (
          <div className="rounded-xl border border-amber-500/25 bg-amber-500/10 p-3 text-xs text-amber-900 dark:text-amber-300">
            {preview?.fallbackUsed && <p className="font-semibold">Hệ thống đã giữ nguyên ảnh để tránh cắt nhầm.</p>}
            {reasonMessage && <p className="mt-1">Lý do: {reasonMessage}</p>}
          </div>
        )}
      </CardContent>
    </Card>
  );
}
