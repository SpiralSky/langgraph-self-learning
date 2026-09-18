"use client";

import { useMemo } from "react";
import { AssistantRuntimeProvider } from "@assistant-ui/react";
import {
  unstable_createLangGraphStream,
  useLangGraphRuntime,
  type LangChainMessage,
} from "@assistant-ui/react-langgraph";

import { createClient } from "@/lib/chatApi";
import { Thread } from "@/components/assistant-ui/Thread";

const ASSISTANT_ID = process.env.NEXT_PUBLIC_ASSISTANT_ID ?? "agent";

export function MyAssistant() {
  const client = useMemo(() => createClient(), []);

  const stream = useMemo(
    () =>
      unstable_createLangGraphStream({
        client,
        assistantId: ASSISTANT_ID,
        streamMode: ["messages", "updates", "custom"],
      }),
    [client],
  );

  const runtime = useLangGraphRuntime({
    unstable_allowCancellation: true,
    stream,
    create: async () => {
      const { thread_id } = await client.threads.create();
      return { externalId: thread_id };
    },
    load: async (externalId) => {
      const state = await client.threads.getState<{
        messages: LangChainMessage[];
      }>(externalId);
      return {
        messages: state.values.messages,
        interrupts: state.tasks[0]?.interrupts,
      };
    },
  });

  return (
    <AssistantRuntimeProvider runtime={runtime}>
      <Thread />
    </AssistantRuntimeProvider>
  );
}