"use client";

import {
  ComposerPrimitive,
  MessagePartPrimitive,
  MessagePrimitive,
  ThreadPrimitive,
  useMessagePartFile,
} from "@assistant-ui/react";
import { FileIcon, PanelRightOpen, Paperclip } from "lucide-react";

import { ThreadView } from "@/components/assistant-ui/agent-inbox";
import { useArtifactPanel } from "@/components/assistant-ui/ArtifactPanel";
import {
  AttachmentAddErrorToast,
  ComposerAttachments,
} from "@/components/assistant-ui/ComposerAttachments";
import { FeedbackAffordance } from "@/components/assistant-ui/feedback/feedback";
import { WidgetTextPart } from "@/components/assistant-ui/markdown/text-widget-part";

/**
 * File part rendered inside user messages: PDFs attached to the composer
 * arrive here as a chip (the package default `File` part renders null).
 */
function UserFilePart() {
  const file = useMessagePartFile();
  if (file.type !== "file" || !file.filename) return null;
  return (
    <span className="my-1 inline-flex max-w-full items-center gap-1.5 rounded-md bg-primary/15 px-2 py-1 text-xs">
      <FileIcon className="size-3.5 shrink-0" />
      <span className="truncate">{file.filename}</span>
    </span>
  );
}

function UserMessage() {
  return (
    <MessagePrimitive.Root className="flex max-w-[80%] justify-end">
      <div className="flex flex-col items-end gap-1 whitespace-pre-wrap rounded-2xl bg-primary px-4 py-2 text-sm text-primary-foreground">
        <MessagePrimitive.Parts
          components={{
            Image: () => (
              <MessagePartPrimitive.Image className="my-0.5 max-h-56 max-w-full rounded-md object-contain" />
            ),
            File: UserFilePart,
          }}
        />
      </div>
    </MessagePrimitive.Root>
  );
}

/**
 * Inline representation of a generative-UI data part.
 *
 * The full artifact renders in the right-side ArtifactPanel; here we show a
 * compact chip instead (override point: `MessagePrimitive.Parts` `components
 * .data` config). Registered per-name artifact renderers (makeAssistantDataUI
 * in `generative-ui.tsx`) still resolve if this override is ever removed.
 */
function InlineDataPart({ name }: { name: string }) {
  const { setOpen } = useArtifactPanel();
  return (
    <button
      type="button"
      onClick={() => setOpen(true)}
      className="my-1 flex max-w-full items-center gap-1.5 rounded-md border border-dashed border-border bg-muted px-2 py-1 text-xs text-muted-foreground hover:bg-muted/70"
    >
      <PanelRightOpen className="size-3.5 shrink-0" />
      <span className="truncate">{name}</span>
      <span aria-hidden className="hidden sm:inline text-muted-foreground/70">
        — open in artifact panel
      </span>
    </button>
  );
}

function AssistantMessage() {
  return (
    <MessagePrimitive.Root className="flex items-start gap-3">
      <div className="flex size-8 shrink-0 items-center justify-center rounded-full bg-muted text-xs font-medium">
        AI
      </div>
      <div className="min-w-0 flex-1 whitespace-pre-wrap pt-1 text-sm">
        <MessagePrimitive.Parts
          components={{
            // task-06 integration point (option a — custom content part):
            // the widget-aware markdown pipeline (:::text / :::code fences)
            // replaces the default plain-text `Text` part. User messages
            // keep the default Text (no markup).
            Text: WidgetTextPart,
            data: { Fallback: InlineDataPart },
          }}
        />
        {/* task-08 feedback affordance: per-assistant-message trigger for the
            feedback dialog (POST /api/feedback → separate feedback FastAPI). */}
        <div className="mt-1 flex items-center gap-2">
          <FeedbackAffordance />
        </div>
      </div>
    </MessagePrimitive.Root>
  );
}

const MESSAGE_COMPONENTS = {
  UserMessage,
  AssistantMessage,
};

export function Thread() {
  return (
    <ThreadPrimitive.Root className="flex h-dvh flex-col bg-background text-foreground">
      <ThreadPrimitive.Viewport className="flex-1 overflow-y-auto p-4">
        <div className="mx-auto flex h-full max-w-3xl flex-col justify-end gap-4">
          <ThreadPrimitive.Messages components={MESSAGE_COMPONENTS} />
          {/* Human-in-the-loop interrupt panel (renders null while idle).
              Reads the pending interrupt from the runtime and resumes the
              run via `useLangGraphSendCommand` on approve/deny. */}
          <ThreadView />
        </div>
      </ThreadPrimitive.Viewport>
      <ThreadPrimitive.ViewportFooter className="border-t border-border p-4">
        <div className="mx-auto max-w-3xl">
          <ComposerPrimitive.AttachmentDropzone className="rounded-xl border border-dashed border-transparent transition-colors data-[dragging=true]:border-primary data-[dragging=true]:bg-primary/5">
            <ComposerPrimitive.Root className="flex flex-col overflow-hidden rounded-xl border border-input bg-card shadow-sm">
              <ComposerAttachments />
              <div className="flex items-end gap-2 p-3">
                <ComposerPrimitive.AddAttachment
                  multiple
                  aria-label="Add attachment"
                  className="rounded-lg p-2 text-muted-foreground transition-colors hover:bg-muted hover:text-foreground"
                >
                  <Paperclip className="size-4" />
                </ComposerPrimitive.AddAttachment>
                <ComposerPrimitive.Input
                  autoFocus
                  placeholder="Message…"
                  addAttachmentOnPaste
                  className="max-h-40 flex-1 resize-none bg-transparent px-2 py-1 text-sm outline-none placeholder:text-muted-foreground"
                />
                <ThreadPrimitive.If running={false}>
                  <ComposerPrimitive.Send className="rounded-lg bg-primary px-4 py-2 text-sm font-medium text-primary-foreground transition-colors hover:opacity-90 disabled:opacity-50">
                    Send
                  </ComposerPrimitive.Send>
                </ThreadPrimitive.If>
                <ThreadPrimitive.If running>
                  <ComposerPrimitive.Cancel className="rounded-lg bg-secondary px-4 py-2 text-sm font-medium text-secondary-foreground hover:opacity-90">
                    Stop
                  </ComposerPrimitive.Cancel>
                </ThreadPrimitive.If>
              </div>
            </ComposerPrimitive.Root>
          </ComposerPrimitive.AttachmentDropzone>
          <AttachmentAddErrorToast />
        </div>
      </ThreadPrimitive.ViewportFooter>
    </ThreadPrimitive.Root>
  );
}