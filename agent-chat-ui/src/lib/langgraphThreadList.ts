import type { RemoteThreadListAdapter } from "@assistant-ui/react";
import { Client } from "@langchain/langgraph-sdk";
import { createAssistantStream } from "assistant-stream";
import { validate } from "uuid";

/**
 * Maps a LangGraph assistant id to the thread-search metadata filter: a
 * deployed langsmith assistant id is a UUID (`assistant_id`), anything else
 * (e.g. a local graph id like `chat`) is a `graph_id`. Mirrors the old
 * ThreadProvider exactly.
 */
export function getThreadSearchMetadata(
  assistantId: string,
): { graph_id: string } | { assistant_id: string } {
  if (validate(assistantId)) {
    return { assistant_id: assistantId };
  }
  return { graph_id: assistantId };
}

/**
 * Extracts a short human label from a thread's latest state values: the text
 * content of the first message when present, otherwise falls back to the
 * thread id. Mirrors the old sidebar's label logic.
 */
export function firstMessageLabel(thread: {
  thread_id: string;
  values?: unknown;
}) {
  const values = thread.values;
  if (
    typeof values === "object" &&
    values !== null &&
    "messages" in values &&
    Array.isArray(values.messages) &&
    values.messages.length > 0
  ) {
    const first = values.messages[0] as {
      content?: string | Array<{ type?: string; text?: string }>;
    };
    const content = first?.content;
    if (typeof content === "string") {
      if (content.trim()) return content.trim();
    } else if (Array.isArray(content)) {
      const text = content
        .filter((c) => c?.type === "text" && typeof c.text === "string")
        .map((c) => c.text as string)
        .join(" ")
        .trim();
      if (text) return text;
    }
  }
  return undefined;
}

export type LangGraphThreadListAdapterOptions = {
  client: Client;
  assistantId: string;
  create: () => Promise<{ externalId: string }>;
};

/**
 * A `RemoteThreadListAdapter` backed by the LangGraph SDK client: `list()`
 * uses `client.threads.search()` (metadata-filtered by assistant/graph id),
 * `initialize()` creates a thread server-side, and `delete()` removes it.
 * The other mutation hooks (rename/archive/…) are no-ops because the old
 * frontend never exposed them, and `generateTitle` streams nothing (thread
 * labels come from the first message, see the sidebar).
 */
export function createLangGraphThreadListAdapter({
  client,
  assistantId,
  create,
}: LangGraphThreadListAdapterOptions): RemoteThreadListAdapter {
  return {
    list: async () => {
      const threads = await client.threads.search({
        metadata: {
          ...getThreadSearchMetadata(assistantId),
        },
        limit: 100,
      });

      return {
        threads: threads.map((thread) => ({
          status: "regular" as const,
          remoteId: thread.thread_id,
          externalId: thread.thread_id,
          title: firstMessageLabel(thread),
        })),
      };
    },

    initialize: async () => {
      const { externalId } = await create();
      return { remoteId: externalId, externalId };
    },

    delete: async (remoteId: string) => {
      await client.threads.delete(remoteId);
    },

    fetch: async (remoteId: string) => {
      const thread = await client.threads.get(remoteId);
      return {
        status: "regular" as const,
        remoteId: thread.thread_id,
        externalId: thread.thread_id,
        title: firstMessageLabel(thread),
      };
    },

    rename: async () => {},
    updateCustom: async () => {},
    archive: async () => {},
    unarchive: async () => {},
    generateTitle: async () => createAssistantStream(() => {}),
  };
}