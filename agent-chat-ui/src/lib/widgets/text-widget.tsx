"use client";

import {
  BadgeCheck,
  CircleX,
  Info,
  MessageSquareText,
  Sparkles,
  TriangleAlert,
} from "lucide-react";

import { cn } from "@/lib/utils";
import { Markdown } from "@/components/assistant-ui/markdown";

/**
 * TextWidget — a Discord-style embedded text card.
 *
 * Renders a bordered container with an optional tinted title bar and body
 * text, using Tailwind classes for styling. The `color` prop maps to a set
 * of predefined accent colors applied as a left border stripe, an icon, and
 * a tinted header background. The body is rendered as markdown with the same
 * pipeline used for regular chat text (GFM, math/KaTeX, code blocks), so
 * fenced content such as lists, tables and `` ``` ``` `` snippets work inside
 * the widget.
 *
 * ## Accepted props (from the ::: fence attributes)
 *
 * | Attribute | Type   | Default   | Description                           |
 * |-----------|--------|-----------|---------------------------------------|
 * | `title`   | string | —         | Bold header displayed at the top.     |
 * | `color`   | string | `"gray"`  | Accent color name (`blue`, `green`, `amber`, `red`, `purple`, `gray`). |
 *
 * The fence body is rendered as markdown (same as regular chat text).
 *
 * ## Example
 *
 * ```markdown
 * :::text title="Note" color="blue"
 * This is an informational note styled like a Discord embed.
 * :::
 * ```
 */
export interface TextWidgetProps {
  title?: string;
  color?: string;
}

interface ColorStyle {
  border: string;
  header: string;
  accent: string;
  icon: React.ComponentType<{ className?: string }>;
}

const COLORS: Record<string, ColorStyle> = {
  blue: {
    border: "border-l-blue-500",
    header: "bg-blue-50 dark:bg-blue-950/50",
    accent: "text-blue-600 dark:text-blue-400",
    icon: Info,
  },
  green: {
    border: "border-l-emerald-500",
    header: "bg-emerald-50 dark:bg-emerald-950/50",
    accent: "text-emerald-600 dark:text-emerald-400",
    icon: BadgeCheck,
  },
  amber: {
    border: "border-l-amber-500",
    header: "bg-amber-50 dark:bg-amber-950/50",
    accent: "text-amber-600 dark:text-amber-400",
    icon: TriangleAlert,
  },
  red: {
    border: "border-l-red-500",
    header: "bg-red-50 dark:bg-red-950/50",
    accent: "text-red-600 dark:text-red-400",
    icon: CircleX,
  },
  purple: {
    border: "border-l-purple-500",
    header: "bg-purple-50 dark:bg-purple-950/50",
    accent: "text-purple-600 dark:text-purple-400",
    icon: Sparkles,
  },
  gray: {
    border: "border-l-zinc-400",
    header: "bg-zinc-100 dark:bg-zinc-800/60",
    accent: "text-zinc-600 dark:text-zinc-400",
    icon: MessageSquareText,
  },
};

function resolveColor(color?: string): ColorStyle {
  return COLORS[color?.toLowerCase() ?? "gray"] ?? COLORS.gray;
}

export const TextWidget: React.FC<{
  props: TextWidgetProps;
  children?: string;
}> = ({ props, children }) => {
  const { title, color } = props;
  const { border, header, accent, icon: Icon } = resolveColor(color);
  const body = (children ?? "").trim();

  return (
    <div
      className={cn(
        "my-4 overflow-hidden rounded-lg border border-l-4 bg-zinc-50/70 dark:bg-zinc-900",
        border,
      )}
    >
      {title && (
        <div
          className={cn(
            "flex items-center gap-2 border-b border-black/5 px-4 py-2 dark:border-white/10",
            header,
          )}
        >
          <Icon className={cn("size-4 shrink-0", accent)} />
          <span
            className={cn("text-xs font-bold tracking-wide uppercase", accent)}
          >
            {title}
          </span>
        </div>
      )}
      {body && (
        <div className="px-4 py-3 text-sm leading-relaxed text-zinc-600 dark:text-zinc-300">
          <Markdown>{body}</Markdown>
        </div>
      )}
    </div>
  );
};
