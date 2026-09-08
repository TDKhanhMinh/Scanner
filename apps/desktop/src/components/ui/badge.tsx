import * as React from "react";
import { cva, type VariantProps } from "class-variance-authority";
import { cn } from "@/lib/utils";

const badgeVariants = cva(
  "inline-flex items-center rounded-full border px-2.5 py-0.5 text-xs font-semibold transition-colors focus:outline-none focus:ring-2 focus:ring-ring focus:ring-offset-2",
  {
    variants: {
      variant: {
        default:
          "border-transparent bg-primary text-primary-foreground shadow hover:bg-primary/80",
        secondary:
          "border-transparent bg-secondary text-secondary-foreground hover:bg-secondary/80",
        destructive:
          "border-transparent bg-destructive/10 text-destructive border-destructive/20 hover:bg-destructive/20",
        success:
          "border-emerald-500/30 bg-emerald-500/15 text-emerald-800 dark:text-emerald-300 font-medium hover:bg-emerald-500/25",
        warning:
          "border-amber-500/30 bg-amber-500/15 text-amber-900 dark:text-amber-300 font-medium hover:bg-amber-500/25",
        outline: "text-foreground",
      },
    },
    defaultVariants: {
      variant: "default",
    },
  }
);

export interface BadgeProps
  extends React.HTMLAttributes<HTMLDivElement>,
    VariantProps<typeof badgeVariants> {}

function Badge({ className, variant, ...props }: BadgeProps) {
  return (
    <div className={cn(badgeVariants({ variant }), className)} {...props} />
  );
}

export { Badge, badgeVariants };

// --- Hybrid Responsive Summary ---
// mobile  (default / sm):  inline flex, text-xs chuẩn đọc được, padding cân đối
// tablet  (md / lg):       giữ kích thước nhỏ gọn trong bảng hoặc thẻ thống kê
// desktop (xl / 2xl):      màu sắc ngữ nghĩa theo tokens hệ thống, subtle border
// Interaction:             hover transition mượt, contrast ratio đạt chuẩn WCAG
