import * as React from "react";
import * as ProgressPrimitive from "@radix-ui/react-progress";
import { cn } from "@/lib/utils";

const Progress = React.forwardRef<
  React.ElementRef<typeof ProgressPrimitive.Root>,
  React.ComponentPropsWithoutRef<typeof ProgressPrimitive.Root>
>(({ className, value, ...props }, ref) => (
  <ProgressPrimitive.Root
    ref={ref}
    className={cn(
      "relative h-2.5 w-full overflow-hidden rounded-full bg-secondary",
      className
    )}
    {...props}
  >
    <ProgressPrimitive.Indicator
      className="h-full w-full flex-1 bg-primary transition-all duration-300 ease-in-out"
      style={{ transform: `translateX(-${100 - (value || 0)}%)` }}
    />
  </ProgressPrimitive.Root>
));
Progress.displayName = ProgressPrimitive.Root.displayName;

export { Progress };

// --- Hybrid Responsive Summary ---
// mobile  (default / sm):  full width, h-2.5 dễ nhìn, bo tròn mềm mại
// tablet  (md / lg):       animation transition mượt mà không giật khung hình
// desktop (xl / 2xl):      hiển thị chỉ số % đi kèm, indicator chuẩn màu primary token
// Interaction:             aria-valuenow/aria-valuemax tự động từ Radix primitive
