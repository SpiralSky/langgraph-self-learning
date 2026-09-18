import { Button } from "@/components/ui/button";
import { useThreads } from "@/providers/Thread";
import { cn } from "@/lib/utils";
import { Thread } from "@langchain/langgraph-sdk";
import { useEffect, useState, useRef } from "react";

import { getContentString } from "../utils";
import { useQueryState, parseAsBoolean } from "nuqs";
import {
  Sheet,
  SheetContent,
  SheetHeader,
  SheetTitle,
} from "@/components/ui/sheet";
import { Skeleton } from "@/components/ui/skeleton";
import { Ellipsis, PanelRightOpen, PanelRightClose, Trash2, Download } from "lucide-react";
import { useMediaQuery } from "@/hooks/useMediaQuery";
import { toast } from "sonner";

function ThreadList({
  threads,
  onThreadClick,
}: {
  threads: Thread[];
  onThreadClick?: (threadId: string) => void;
}) {
  const [threadId, setThreadId] = useQueryState("threadId");
  const { deleteThread, exportThread, setThreads } = useThreads();
  const [openMenuId, setOpenMenuId] = useState<string | null>(null);
  const triggerRef = useRef<HTMLButtonElement | null>(null);
  const menuRef = useRef<HTMLDivElement | null>(null);

  const handleExport = async (t: Thread) => {
    setOpenMenuId(null);
    try {
      const data = await exportThread(t.thread_id);
      await navigator.clipboard.writeText(JSON.stringify(data, null, 2));
      toast.success("Conversation copied to clipboard");
    } catch (error) {
      console.error("Failed to export thread", error);
      toast.error("Failed to copy conversation");
    }
  };

  const handleDelete = async (t: Thread) => {
    setOpenMenuId(null);
    if (!window.confirm("Delete this thread?")) return;
    await deleteThread(t.thread_id);
    setThreads((prev) => prev.filter((th) => th.thread_id !== t.thread_id));
    if (threadId === t.thread_id) {
      setThreadId(null);
    }
  };

  useEffect(() => {
    const handleClickOutside = (e: MouseEvent) => {
      const target = e.target as Node;
      const isTrigger = triggerRef.current?.contains(target);
      if (
        !isTrigger &&
        menuRef.current &&
        !menuRef.current.contains(target)
      ) {
        setOpenMenuId(null);
      }
    };
    if (openMenuId) {
      document.addEventListener("mousedown", handleClickOutside);
    }
    return () => document.removeEventListener("mousedown", handleClickOutside);
  }, [openMenuId]);

  return (
    <div className="flex h-full w-full flex-col items-start justify-start gap-2 overflow-y-scroll [&::-webkit-scrollbar]:w-1.5 [&::-webkit-scrollbar-thumb]:rounded-full [&::-webkit-scrollbar-thumb]:bg-gray-300 [&::-webkit-scrollbar-track]:bg-transparent">
      {threads.map((t) => {
        let itemText = t.thread_id;
        if (
          typeof t.values === "object" &&
          t.values &&
          "messages" in t.values &&
          Array.isArray(t.values.messages) &&
          t.values.messages?.length > 0
        ) {
          const firstMessage = t.values.messages[0];
          itemText = getContentString(firstMessage.content);
        }
        return (
          <div
            key={t.thread_id}
            role="button"
            tabIndex={0}
            className={cn(
              "group relative flex h-9 w-full items-center justify-between gap-2 px-2 rounded-md text-sm font-medium",
              t.thread_id === threadId && "bg-gray-100",
            )}
            onClick={() => {
              onThreadClick?.(t.thread_id);
              if (t.thread_id === threadId) return;
              setThreadId(t.thread_id);
            }}
            onKeyDown={(e) => {
              if (e.key === "Enter" || e.key === " ") {
                e.preventDefault();
                onThreadClick?.(t.thread_id);
                if (t.thread_id !== threadId) setThreadId(t.thread_id);
              }
            }}
          >
            <p className="min-w-0 flex-1 truncate text-ellipsis">{itemText}</p>
            <div className="relative flex items-center">
              <button
                ref={triggerRef}
                type="button"
                aria-haspopup="menu"
                aria-expanded={openMenuId === t.thread_id}
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
                  ref={menuRef}
                  className="bg-popover text-popover-foreground absolute right-0 top-full z-50 w-32 rounded-md border shadow-md"
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
    </div>
  );
}

function ThreadHistoryLoading() {
  return (
    <div className="flex h-full w-full flex-col items-start justify-start gap-2 overflow-y-scroll [&::-webkit-scrollbar]:w-1.5 [&::-webkit-scrollbar-thumb]:rounded-full [&::-webkit-scrollbar-thumb]:bg-gray-300 [&::-webkit-scrollbar-track]:bg-transparent">
      {Array.from({ length: 30 }).map((_, i) => (
        <Skeleton
          key={`skeleton-${i}`}
          className="h-10 w-[280px]"
        />
      ))}
    </div>
  );
}

export default function ThreadHistory() {
  const isLargeScreen = useMediaQuery("(min-width: 1024px)");
  const [chatHistoryOpen, setChatHistoryOpen] = useQueryState(
    "chatHistoryOpen",
    parseAsBoolean.withDefault(false),
  );

  const { getThreads, threads, setThreads, threadsLoading, setThreadsLoading } =
    useThreads();

  useEffect(() => {
    if (typeof window === "undefined") return;
    setThreadsLoading(true);
    getThreads()
      .then(setThreads)
      .catch(console.error)
      .finally(() => setThreadsLoading(false));
  }, []);

  return (
    <>
      <div className="shadow-inner-right hidden h-screen w-[300px] shrink-0 flex-col items-start justify-start gap-6 border-r-[1px] border-slate-300 lg:flex">
        <div className="flex w-full items-center justify-between px-4 pt-1.5">
          <Button
            className="hover:bg-gray-100"
            variant="ghost"
            onClick={() => setChatHistoryOpen((p) => !p)}
          >
            {chatHistoryOpen ? (
              <PanelRightOpen className="size-5" />
            ) : (
              <PanelRightClose className="size-5" />
            )}
          </Button>
          <h1 className="text-xl font-semibold tracking-tight">
            Thread History
          </h1>
        </div>
        {threadsLoading ? (
          <ThreadHistoryLoading />
        ) : (
          <ThreadList threads={threads} />
        )}
      </div>
      <div className="lg:hidden">
        <Sheet
          open={!!chatHistoryOpen && !isLargeScreen}
          onOpenChange={(open) => {
            if (isLargeScreen) return;
            setChatHistoryOpen(open);
          }}
        >
          <SheetContent
            side="left"
            className="flex lg:hidden"
          >
            <SheetHeader>
              <SheetTitle>Thread History</SheetTitle>
            </SheetHeader>
            <ThreadList
              threads={threads}
              onThreadClick={() => setChatHistoryOpen((o) => !o)}
            />
          </SheetContent>
        </Sheet>
      </div>
    </>
  );
}
