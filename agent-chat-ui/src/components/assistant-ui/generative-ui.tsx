"use client";

// Side-effect import: registers the built-in widgets (`text`, `code`) with
// the WidgetRegistry at module load. The old app never imported this barrel
// in the live page tree, so the registry was silently empty; importing it
// here makes both the `:::`-fence path and the generative-UI data-part path
// resolve the same widgets. task-06 owns the fence path and should keep
// consuming this same registry.
import "@/lib/widgets";

import {
  makeAssistantDataUI,
  type DataMessagePartComponent,
} from "@assistant-ui/react";
import { WidgetRegistry } from "@/lib/widget-registry";

/**
 * Generative UI (artifact) renderers for assistant-ui's `data` message parts.
 *
 * react-langgraph converts LangGraph `ui` messages (from the `values` state
 * key and/or `custom` stream events) into `data` parts on the assistant
 * message: `{ type: "data", name: <ui.name>, data: <ui.props> }`. Each
 * registered widget/artifact name gets a `makeAssistantDataUI` renderer that
 * resolves the name against the shared `WidgetRegistry`, so a UIMessage named
 * `text` and a `:::text` fence render through the exact same component.
 *
 * The renderers are registered while `<GenerativeUIRenderers/>` is mounted
 * (render inside `AssistantRuntimeProvider`). The side panel renders the same
 * components directly via {@link DataPartView}.
 *
 * Artifact → renderer mapping (recorded in shared.md):
 * - `text` → `TextWidget` (props: `title`, `color`; body via `children`)
 * - `code` → `CodeBlockWidget` (props: `language`, `title`; body via `children`)
 * - any other registered widget name → resolved via `WidgetRegistry`
 *
 * UIMessage props become the widget `props`; a string body is read from
 * `children` / `content` / `body` in the props (the `custom`-channel UIMessage
 * format carries no separate children field).
 */

function bodyFromData(data: unknown): string | undefined {
  if (data == null || typeof data !== "object") return undefined;
  const value = data as Record<string, unknown>;
  for (const key of ["children", "content", "body"]) {
    const candidate = value[key];
    if (typeof candidate === "string") return candidate;
  }
  return undefined;
}

export function DataPartView({
  name,
  data,
}: {
  name: string;
  data: unknown;
}) {
  const def = WidgetRegistry.get(name);
  if (!def) {
    return (
      <div className="rounded-md border border-dashed border-border p-3 text-sm text-muted-foreground">
        Unknown artifact: <code>{name}</code>
      </div>
    );
  }
  const Component = def.render;
  return (
    <Component
      props={(data ?? {}) as Record<string, unknown>}
      children={bodyFromData(data)}
    />
  );
}

export const WidgetDataPartRenderer: DataMessagePartComponent = ({ name, data }) => {
  return <DataPartView name={name} data={data} />;
};

const REGISTERED_DATA_UI = WidgetRegistry.list().map((name) => ({
  name,
  DataUI: makeAssistantDataUI({ name, render: WidgetDataPartRenderer }),
}));

/**
 * Mount inside `AssistantRuntimeProvider`. While mounted, each returned
 * component registers its data-part renderer with assistant-ui, so any
 * `data` part named after a registered widget resolves to its widget.
 */
export function GenerativeUIRenderers() {
  return (
    <>
      {REGISTERED_DATA_UI.map(({ name, DataUI }) => (
        <DataUI key={name} />
      ))}
    </>
  );
}