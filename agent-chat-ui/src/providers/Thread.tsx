import { validate } from "uuid";
import { getApiKey } from "@/lib/api-key";
import { Thread, ThreadState } from "@langchain/langgraph-sdk";
import { useQueryState } from "nuqs";
import {
  createContext,
  useContext,
  ReactNode,
  useCallback,
  useState,
  Dispatch,
  SetStateAction,
} from "react";
import { createClient } from "./client";

interface ThreadContextType {
  getThreads: () => Promise<Thread[]>;
  deleteThread: (threadId: string) => Promise<void>;
  exportThread: (threadId: string) => Promise<{
    thread: Thread;
    states: ThreadState[];
  }>;
  threads: Thread[];
  setThreads: Dispatch<SetStateAction<Thread[]>>;
  threadsLoading: boolean;
  setThreadsLoading: Dispatch<SetStateAction<boolean>>;
}

const ThreadContext = createContext<ThreadContextType | undefined>(undefined);

function getThreadSearchMetadata(
  assistantId: string,
): { graph_id: string } | { assistant_id: string } {
  if (validate(assistantId)) {
    return { assistant_id: assistantId };
  } else {
    return { graph_id: assistantId };
  }
}

export function ThreadProvider({ children }: { children: ReactNode }) {
  const envApiUrl: string | undefined = process.env.NEXT_PUBLIC_API_URL;
  const envAssistantId: string | undefined =
    process.env.NEXT_PUBLIC_ASSISTANT_ID;
  const envAuthScheme: string | undefined = process.env.NEXT_PUBLIC_AUTH_SCHEME;

  const [apiUrl] = useQueryState("apiUrl", {
    defaultValue: envApiUrl || "",
  });
  const [assistantId] = useQueryState("assistantId");
  const [authScheme] = useQueryState("authScheme", {
    defaultValue: envAuthScheme || "",
  });
  const [threads, setThreads] = useState<Thread[]>([]);
  const [threadsLoading, setThreadsLoading] = useState(false);

  const getThreads = useCallback(async (): Promise<Thread[]> => {
    const resolvedAssistantId = assistantId || envAssistantId;
    if (!apiUrl || !resolvedAssistantId) return [];
    const client = createClient(
      apiUrl,
      getApiKey() ?? undefined,
      authScheme || undefined,
    );

    const threads = await client.threads.search({
      metadata: {
        ...getThreadSearchMetadata(resolvedAssistantId),
      },
      limit: 100,
    });

    return threads;
  }, [apiUrl, assistantId, authScheme, envAssistantId]);

  const deleteThread = useCallback(async (threadId: string): Promise<void> => {
    const client = createClient(
      apiUrl || envApiUrl || "",
      getApiKey() ?? undefined,
      (authScheme || envAuthScheme) || undefined,
    );
    await client.threads.delete(threadId);
  }, [apiUrl, envApiUrl, authScheme, envAuthScheme]);

  const exportThread = useCallback(
    async (threadId: string): Promise<{ thread: Thread; states: ThreadState[] }> => {
      const client = createClient(
        apiUrl || envApiUrl || "",
        getApiKey() ?? undefined,
        (authScheme || envAuthScheme) || undefined,
      );
      const [thread, states] = await Promise.all([
        client.threads.get(threadId),
        client.threads.getHistory(threadId),
      ]);
      return { thread, states };
    },
    [apiUrl, envApiUrl, authScheme, envAuthScheme],
  );

  const value = {
    getThreads,
    deleteThread,
    exportThread,
    threads,
    setThreads,
    threadsLoading,
    setThreadsLoading,
  };

  return (
    <ThreadContext.Provider value={value}>{children}</ThreadContext.Provider>
  );
}

export function useThreads() {
  const context = useContext(ThreadContext);
  if (context === undefined) {
    throw new Error("useThreads must be used within a ThreadProvider");
  }
  return context;
}
