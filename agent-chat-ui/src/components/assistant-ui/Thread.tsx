"use client";

import {
  ComposerPrimitive,
  MessagePrimitive,
  ThreadPrimitive,
} from "@assistant-ui/react";

function UserMessage() {
  return (
    <MessagePrimitive.Root className="flex max-w-[80%] justify-end">
      <div className="whitespace-pre-wrap rounded-2xl bg-primary px-4 py-2 text-sm text-primary-foreground">
        <MessagePrimitive.Parts />
      </div>
    </MessagePrimitive.Root>
  );
}

function AssistantMessage() {
  return (
    <MessagePrimitive.Root className="flex items-start gap-3">
      <div className="flex size-8 shrink-0 items-center justify-center rounded-full bg-muted text-xs font-medium">
        AI
      </div>
      <div className="whitespace-pre-wrap pt-1 text-sm">
        <MessagePrimitive.Parts />
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
        </div>
      </ThreadPrimitive.Viewport>
      <ThreadPrimitive.ViewportFooter className="border-t border-border p-4">
        <div className="mx-auto max-w-3xl">
          <ComposerPrimitive.Root className="flex items-end gap-2 rounded-xl border border-input bg-card p-3 shadow-sm">
            <ComposerPrimitive.Input
              autoFocus
              placeholder="Message…"
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
          </ComposerPrimitive.Root>
        </div>
      </ThreadPrimitive.ViewportFooter>
    </ThreadPrimitive.Root>
  );
}