"use client";

import { type TextMessagePartComponent } from "@assistant-ui/react";

import { Markdown } from "./index";

/**
 * WidgetTextPart — the `Text` message-part override for assistant messages.
 *
 * Integration point (task-06 selected option (a) — custom content part):
 * passed as `components.Text` to `MessagePrimitive.Parts` in
 * `components/assistant-ui/Thread.tsx`. It replaces assistant-ui's default
 * plain-text `Text` part with the widget-aware markdown pipeline
 * (`parseWidgetFences` + react-markdown + `WidgetRenderer`), so `:::text` and
 * `:::code` fences render as widgets inline in the message stream.
 *
 * Streaming: the part's `text` prop is the live accumulator of the stream
 * (react-langgraph appends each content delta to the running text part), so
 * this component re-renders with partial content on every chunk — fences
 * whose closing `:::` has not yet arrived stay visible as raw markdown until
 * they complete (same behavior as the old app). The trailing `●` indicator
 * mirrors the default web `Text` part while the part is still running.
 *
 * Note on choice: option (b) (registry renderers) already exists for
 * generative-UI `data` parts (`generative-ui.tsx`), but `:::` fences live
 * inside the raw markdown/text of an assistant message and only become
 * distinguishable segments once parsed out of the text stream — so they
 * cannot ride the `data`-part path. Both paths resolve the same
 * `WidgetRegistry` (barrel `@/lib/widgets`), so a widget registered there
 * renders through either.
 */
export const WidgetTextPart: TextMessagePartComponent = ({
  text,
  status,
}) => {
  return (
    <>
      <Markdown>{text}</Markdown>
      {status.type === "running" && (
        <span
          aria-hidden
          className="ml-0.5 inline-block text-sm leading-relaxed text-foreground/50"
        >
          {"\u25CF"}
        </span>
      )}
    </>
  );
};