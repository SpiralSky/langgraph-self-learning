import React from "react";
import { WidgetRegistry } from "@/lib/widget-registry";

/**
 * Props extracted from a `:::` fence block parsed by `parseWidgetFences`.
 */
interface WidgetRendererProps {
  /** The widget name (from the opening `:::` line). */
  name: string;
  /** Widget-specific attributes (fence attr names → values). */
  widgetProps: Record<string, string>;
  /** Raw text content of the fence body. */
  children: string;
}

/**
 * WidgetRenderer resolves a widget name against the WidgetRegistry and
 * renders the registered component.  If no widget is found a simple
 * fallback is shown.
 *
 * This component is used in two places:
 * 1. Directly by `MarkdownText` for every `:::` fence parsed from the
 *    markdown source (see `parseWidgetFences`).
 * 2. Behind the `LoadExternalComponent` bridge for the generative‑UI path.
 */
export const WidgetRenderer: React.FC<WidgetRendererProps> = ({
  name,
  widgetProps: rawProps,
  children,
}) => {
  const def = WidgetRegistry.get(name);

  if (!def) {
    return (
      <div className="my-4 rounded border border-dashed border-gray-300 p-3 text-sm text-gray-400">
        Unknown widget: <code>{name}</code>
      </div>
    );
  }

  const Component = def.render;
  return <Component props={rawProps as Record<string, unknown>} children={children} />;
};