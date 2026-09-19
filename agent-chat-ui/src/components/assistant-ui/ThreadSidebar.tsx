"use client";

import { Input } from "@/components/ui/input";
import { Skeleton } from "@/components/ui/skeleton";
import { cn } from "@/lib/utils";
import { Client, Thread } from "@langchain/langgraph-sdk";
import { Download, Ellipsis, PanelRightClose, PanelRightOpen, Plus, Search, Trash2 } from "lucide-react";
import { useCallback, useEffect, useRef, useState } from "react";
import { toast } from "sonner";

import {
  firstMessageLabel,
  getThreadSearchMetadata,
} from "@/lib/langgraphThreadList";

async function sleep(ms = 4000) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function ThreadItemLabel(thread: Thread) {
  return firstMessageLabel(thread) ?? thread.thread_id;
}

export type ThreadSidebarProps = {
  client: Client;
  assistantId: string;
  /** The runtime's active thread id (`null` = brand-new thread). */
  threadId: string | null;
  /** Change the active thread; `null` starts a new thread. */
  onSelectThread: (threadId: string | null) => void;
  open: boolean;
  onToggle: () => void;
};

/**
 * Thread history sidebar (list / search / delete / export / new) ported from
 * the old ThreadProvider + `components/thread/history` onto the assistant-ui
 * runtime. The list is fetched straight off the LangGraph SDK client
 * (`client.threads.search`), mirroring the old provider, while thread
 * switching goes through the runtime's controlled `threadId` (see
 * MyAssistant). Refetch-on-create is preserved: whenever the active thread id
 * changes (a fresh thread settles after its first message), the list is
 * re-fetched after a short delay, exactly like the old `onThreadId` handler.
 */
export function ThreadSidebar({
  client,
  assistantId,
  threadId,
  onSelectThread,
  open,
  onToggle,
}: ThreadSidebarProps) {
  const [threads, setThreads] = useState<Thread[]>([]);
  const [threadsLoading, setThreadsLoading] = useState(true);
  const [search, setSearch] = useState("");
  const [openMenuId, setOpenMenuId] = useState<string | null>(null);

  const getThreads = useCallback(async (): Promise<Thread[]> => {
    if (!assistantId) return [];
    return client.threads.search({
      metadata: {
        ...getThreadSearchMetadata(assistantId),
      },
      limit: 100,
    });
  }, [client, assistantId]);

  const loadThreads = useCallback(
    () =>
      getThreads()
        .then(setThreads)
        .catch(console.error)
        .finally(() => setThreadsLoading(false)),
    [getThreads],
  );

  // Initial load (the spinner is already open at mount, so no synchronous
  // setState runs inside this effect).
  useEffect(() => {
    void loadThreads();
  }, [loadThreads]);

  // Spinner-on refetch, only ever called from async callbacks (below).
  const refetch = useCallback(() => {
    setThreadsLoading(true);
    void loadThreads();
  }, [loadThreads]);

  // Refetch shortly after the active thread id changes so a freshly created
  // thread shows up in the list (old StreamProvider onThreadId behavior).
  const prevThreadIdRef = useRef(threadId);
  useEffect(() => {
    const previous = prevThreadIdRef.current;
    prevThreadIdRef.current = threadId;
    if (typeof window === "undefined" || previous === threadId) return;
    sleep().then(refetch);
  }, [threadId, refetch]);

  const handleDelete = async (t: Thread) => {
    setOpenMenuId(null);
    if (!window.confirm("Delete this thread?")) return;
    try {
      await client.threads.delete(t.thread_id);
      setThreads((prev) => prev.filter((th) => th.thread_id !== t.thread_id));
      if (threadId === t.thread_id) {
        onSelectThread(null);
      }
    } catch (error) {
      console.error("Failed to delete thread", error);
      toast.error("Failed to delete thread");
    }
  };

  const handleExport = async (t: Thread) => {
    setOpenMenuId(null);
    try {
      const [thread, states] = await Promise.all([
        client.threads.get(t.thread_id),
        client.threads.getHistory(t.thread_id),
      ]);
      await navigator.clipboard.writeText(
        JSON.stringify({ thread, states }, null, 2),
      );
      toast.success("Conversation copied to clipboard");
    } catch (error) {
      console.error("Failed to export thread", error);
      toast.error("Failed to copy conversation");
    }
  };

  useEffect(() => {
    const handleClickOutside = (e: MouseEvent) => {
      const target = e.target as Node;
      if (
        openMenuId !== null &&
        !(target as HTMLElement).closest?.('[data-thread-menu="true"]')
      ) {
        setOpenMenuId(null);
      }
    };
    if (openMenuId) {
      document.addEventListener("mousedown", handleClickOutside);
    }
    return () => document.removeEventListener("mousedown", handleClickOutside);
  }, [openMenuId]);

  const query = search.trim().toLowerCase();
  const filteredThreads = threads.filter((t) => {
    if (!query) return true;
    const label = ThreadItemLabel(t).toLowerCase();
    return label.includes(query) || t.thread_id.includes(query);
  });

  return (
    <div
      className={cn(
        "hidden h-screen w-[300px] shrink-0 flex-col items-start justify-start gap-4 border-r border-slate-300 lg:flex",
        !open && "lg:hidden",
      )}
    >
      <div className="flex w-full items-center justify-between px-4 pt-1.5">
        <button
          type="button"
          aria-label="Toggle thread history"
          className="rounded-md p-1.5 text-muted-foreground hover:bg-gray-100"
          onClick={onToggle}
        >
          {open ? (
            <PanelRightOpen className="size-5" />
          ) : (
            <PanelRightClose className="size-5" />
          )}
        </button>
        <h1 className="text-lg font-semibold tracking-tight">
          Thread History
        </h1>
        <button
          type="button"
          aria-label="New thread"
          title="New thread"
          className="rounded-md p-1.5 text-muted-foreground hover:bg-gray-100"
          onClick={() => onSelectThread(null)}
        >
          <Plus className="size-5" />
        </button>
      </div>

      <div className="flex w-full items-center gap-2 px-3">
        <Search className="size-4 shrink-0 text-muted-foreground" />
        <Input
          value={search}
          onChange={(e) => setSearch(e.target.value)}
          placeholder="Search threads..."
          className="h-8 w-full"
        />
      </div>

      {threadsLoading ? (
        <div className="flex h-full w-full flex-col items-start justify-start gap-2 overflow-y-scroll [&::-webkit-scrollbar]:w-1.5 [&::-webkit-scrollbar-thumb]:rounded-full [&::-webkit-scrollbar-thumb]:bg-gray-300 [&::-webkit-scrollbar-track]:bg-transparent">
          {Array.from({ length: 12 }).map((_, i) => (
            <Skeleton key={`skeleton-${i}`} className="h-10 w-[280px]" />
          ))}
        </div>
      ) : (
        <div className="flex h-full w-full flex-col items-start justify-start gap-2 overflow-y-scroll [&::-webkit-scrollbar]:w-1.5 [&::-webkit-scrollbar-thumb]:rounded-full [&::-webkit-scrollbar-thumb]:bg-gray-300 [&::-webkit-scrollbar-track]:bg-transparent">
          {filteredThreads.map((t) => {
            const itemText = ThreadItemLabel(t);
            return (
              <div
                key={t.thread_id}
                role="button"
                tabIndex={0}
                className={cn(
                  "group relative flex h-9 w-full items-center justify-between gap-2 rounded-md px-2 text-sm font-medium",
                  t.thread_id === threadId && "bg-gray-100",
                )}
                onClick={() => {
                  if (t.thread_id === threadId) return;
                  onSelectThread(t.thread_id);
                }}
                onKeyDown={(e) => {
                  if (e.key === "Enter" || e.key === " ") {
                    e.preventDefault();
                    if (t.thread_id === threadId) return;
                    onSelectThread(t.thread_id);
                  }
                }}
              >
                <p className="min-w-0 flex-1 truncate">{itemText}</p>
                <div className="relative flex items-center">
                  <button
                    type="button"
                    aria-haspopup="menu"
                    aria-expanded={openMenuId === t.thread_id}
                    data-thread-menu="true"
                    className="rounded-md p-1 opacity-0 group-hover:opacity-100 hover:bg-gray-100"
                    onClick={(e) => {
                      e.stopPropagation();
                      setOpenMenuId(
                        openMenuId === t.thread_id ? null : t.thread_id,
                      );
                    }}
                    onKeyDown={(e) => e.stopPropagation()}
                  >
                    <Ellipsis className="size-4" />
                  </button>
                  {openMenuId === t.thread_id && (
                    <div
                      data-thread-menu="true"
                      className="absolute right-0 top-full z-50 w-32 rounded-md border bg-popover text-popover-foreground shadow-md"
                    >
                      <button
                        type="button"
                        className="flex w-full items-center gap-2 px-3 py-2 text-sm hover:bg-gray-100"
                        onClick={(e) => {
                          e.stopPropagation();
                          handleExport(t);
                        }}
                        onKeyDown={(e) => e.stopPropagation()}
                      >
                        <Download className="size-4" />
                        Export
                      </button>
                      <button
                        type="button"
                        className="flex w-full items-center gap-2 px-3 py-2 text-sm text-red-600 hover:bg-red-50"
                        onClick={(e) => {
                          e.stopPropagation();
                          handleDelete(t);
                        }}
                        onKeyDown={(e) => e.stopPropagation()}
                      >
                        <Trash2 className="size-4" />
                        Delete
                      </button>
                    </div>
                  )}
                </div>
              </div>
            );
          })}
          {filteredThreads.length === 0 && (
            <p className="px-3 py-2 text-sm text-muted-foreground">
              No threads found.
            </p>
          )}
        </div>
      )}
    </div>
  );
}