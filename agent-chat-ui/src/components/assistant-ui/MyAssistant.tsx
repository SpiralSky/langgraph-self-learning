"use client";

import { useEffect, useMemo, useState } from "react";
import { AssistantRuntimeProvider } from "@assistant-ui/react";
import {
  unstable_createLangGraphStream,
  useLangGraphRuntime,
  type LangChainMessage,
  type UIMessage,
} from "@assistant-ui/react-langgraph";
import { PanelRightClose } from "lucide-react";
import { useQueryState } from "nuqs";
import { toast } from "sonner";

import { Thread } from "@/components/assistant-ui/Thread";
import { ThreadSidebar } from "@/components/assistant-ui/ThreadSidebar";
import {
  ArtifactPanel,
  ArtifactPanelProvider,
} from "@/components/assistant-ui/ArtifactPanel";
import { GenerativeUIRenderers } from "@/components/assistant-ui/generative-ui";
import { checkGraphStatus, createClient } from "@/lib/chatApi";
import { createLangGraphThreadListAdapter } from "@/lib/langgraphThreadList";
import { multimodalAttachmentAdapter } from "@/lib/multimodalAttachmentAdapter";

const DEFAULT_ASSISTANT_ID = process.env.NEXT_PUBLIC_ASSISTANT_ID ?? "chat";

export type MyAssistantProps = {
  apiUrl: string;
  assistantId?: string;
  apiKey?: string;
  authScheme?: string;
};

export function MyAssistant({
  apiUrl,
  assistantId,
  apiKey,
  authScheme,
}: MyAssistantProps) {
  const resolvedAssistantId = assistantId || DEFAULT_ASSISTANT_ID;

  const client = useMemo(
    () => createClient({ apiUrl, apiKey, authScheme }),
    [apiUrl, apiKey, authScheme],
  );

  // The active thread id is a controlled value: the sidebar (and the URL
  // `?threadId=` param, mirroring the old app) sets it, and the runtime
  // switches threads whenever it changes. `onThreadIdChange` echoes back the
  // settled remote id when a brand-new thread is initialized on first send.
  const [threadId, setThreadId] = useQueryState("threadId");
  const [sidebarOpen, setSidebarOpen] = useState(true);

  // Artifact-panel state is keyed by the thread it was opened for, so a
  // thread switch closes the panel without an effect (old app behavior).
  const [artifactState, setArtifactState] = useState<{
    thread: string | null;
    open: boolean;
  }>({ thread: null, open: false });
  const artifactOpen = artifactState.open && artifactState.thread === threadId;
  const setArtifactOpen = (open: boolean) =>
    setArtifactState((prev) => ({ thread: threadId, open }));

  const stream = useMemo(
    () =>
      unstable_createLangGraphStream({
        client,
        assistantId: resolvedAssistantId,
        // `values` is required for the generative-UI state-snapshot path:
        // react-langgraph reads `state.values[uiStateKey]` from values
        // events. `custom` is kept so any future `ui` custom events (live
        // path) are delivered too.
        streamMode: ["messages", "values", "updates", "custom"],
      }),
    [client, resolvedAssistantId],
  );

  // Backs the runtime's thread list with the LangGraph SDK client so the
  // runtime can switch between pre-existing threads (and create new ones)
  // without assistant-cloud.
  const threadListAdapter = useMemo(
    () =>
      createLangGraphThreadListAdapter({
        client,
        assistantId: resolvedAssistantId,
        create: async () => {
          const { thread_id } = await client.threads.create();
          return { externalId: thread_id };
        },
      }),
    [client, resolvedAssistantId],
  );

  const runtime = useLangGraphRuntime({
    unstable_allowCancellation: true,
    stream,
    // Multimodal composer attachments (images → image_url content blocks,
    // PDFs → base64 file content blocks; see multimodalAttachmentAdapter.ts).
    adapters: {
      attachments: multimodalAttachmentAdapter,
    },
    unstable_threadListAdapter: threadListAdapter,
    threadId: threadId || undefined,
    onThreadIdChange: (id) => void setThreadId(id ?? null),
    // Generative UI lives under this state key on the graph (default `"ui"`,
    // matches the chat graph's pass-through channel). react-langgraph binds
    // `ui` state values to assistant messages as `data` parts.
    uiStateKey: "ui",
    load: async (externalId) => {
      const state = await client.threads.getState<{
        messages: LangChainMessage[];
        ui?: UIMessage[];
      }>(externalId);
      return {
        messages: state.values.messages,
        interrupts: state.tasks[0]?.interrupts,
        // Persisted generative UI from graph state, rendered as `data` parts
        // on the reloaded assistant message.
        uiMessages: state.values.ui,
      };
    },
  });

  useEffect(() => {
    let active = true;
    checkGraphStatus(apiUrl, apiKey, authScheme).then((ok) => {
      if (!active || ok) return;
      toast.error("Failed to connect to LangGraph server", {
        description: () => (
          <p>
            Please ensure your graph is running at <code>{apiUrl}</code> and
            your API key is correctly set (if connecting to a deployed graph).
          </p>
        ),
        duration: 10000,
        richColors: true,
        closeButton: true,
      });
    });
    return () => {
      active = false;
    };
  }, [apiKey, apiUrl, authScheme]);

  return (
    <AssistantRuntimeProvider runtime={runtime}>
      <GenerativeUIRenderers />
      <ArtifactPanelProvider
        open={artifactOpen}
        onOpenChange={setArtifactOpen}
      >
        <div className="flex h-dvh w-full overflow-hidden bg-background text-foreground">
          <ThreadSidebar
            client={client}
            assistantId={resolvedAssistantId}
            threadId={threadId}
            onSelectThread={(id) => void setThreadId(id)}
            open={sidebarOpen}
            onToggle={() => setSidebarOpen((open) => !open)}
          />
          <div className="relative min-w-0 flex-1 overflow-hidden">
            {!sidebarOpen && (
              <button
                type="button"
                aria-label="Open thread history"
                className="absolute left-2 top-2 z-20 rounded-md border bg-background p-1.5 text-muted-foreground shadow-sm hover:bg-muted"
                onClick={() => setSidebarOpen(true)}
              >
                <PanelRightClose className="size-5" />
              </button>
            )}
            <div className="flex h-full">
              <div className="min-w-0 flex-1">
                <Thread />
              </div>
              {/* Always mounted so its auto-open effect can fire; renders
                  nothing while closed. */}
              <ArtifactPanel />
            </div>
          </div>
        </div>
      </ArtifactPanelProvider>
    </AssistantRuntimeProvider>
  );
}