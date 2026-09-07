import * as React from "react";
import * as SeparatorPrimitive from "@radix-ui/react-separator";
import { cn } from "@/lib/utils";

const Separator = React.forwardRef<
  React.ElementRef<typeof SeparatorPrimitive.Root>,
  React.ComponentPropsWithoutRef<typeof SeparatorPrimitive.Root>
>(
  (
    { className, orientation = "horizontal", decorative = true, ...props },
    ref
  ) => (
    <SeparatorPrimitive.Root
      ref={ref}
      decorative={decorative}
      orientation={orientation}
      className={cn(
        "shrink-0 bg-border",
        orientation === "horizontal" ? "h-[1px] w-full" : "h-full w-[1px]",
        className
      )}
      {...props}
    />
  )
);
Separator.displayName = SeparatorPrimitive.Root.displayName;

export { Separator };

// --- Hybrid Responsive Summary ---
// mobile  (default / sm):  đường kẻ tinh tế phân cách các khối nội dung
// tablet  (md / lg):       hỗ trợ cả horizontal và vertical separator giữa các cột
// desktop (xl / 2xl):      màu border nhẹ nhàng, hòa nhập với dark/light mode
// Interaction:             aria-hidden khi decorative = true
