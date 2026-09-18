import { useState, useCallback, useRef } from "react";
import { useWorkspaceFiles } from "@/hooks/useWorkspaceFiles";
import { useFileContent } from "@/hooks/useFileContent";
import { FileTree } from "./FileTree";
import { CodeViewer } from "./CodeViewer";
import { OutputView } from "./OutputView";
import { ArtifactToolbar } from "./ArtifactToolbar";

interface ArtifactPanelProps {
  open: boolean;
  onClose: () => void;
}

export function ArtifactPanel({ open, onClose }: ArtifactPanelProps) {
  const [selectedPath, setSelectedPath] = useState<string | null>(null);
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const { items, loading: filesLoading, error: filesError } = useWorkspaceFiles();
  const selectedHostPath =
    items.find((i) => i.path === selectedPath)?.host_path ?? null;
  const {
    content,
    loading: contentLoading,
    error: contentError,
    isBinary,
    isTooLarge,
  } = useFileContent(selectedPath ?? "");
  const fileTreeRef = useRef<HTMLDivElement>(null);

  const handleDownloadFile = useCallback(
    async (path: string) => {
      const res = await fetch(
        `/api/workspace/file?path=${encodeURIComponent(path)}`,
      );
      if (!res.ok) return;
      const data: { name: string; content: string } = await res.json();
      const blob = new Blob([data.content], { type: "text/plain" });
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = data.name;
      a.click();
      URL.revokeObjectURL(url);
    },
    [],
  );

  const handleOpenInVSCode = useCallback(() => {
    if (!selectedPath) return;
    const hostPath = items.find((i) => i.path === selectedPath)?.host_path;
    if (hostPath) {
      window.open(`vscode://file/${hostPath}`, "_blank");
    }
  }, [selectedPath, items]);

  if (!open) return null;

  return (
    <div className="flex h-full w-full bg-zinc-900 text-zinc-200">
      {sidebarOpen && (
        <div
          className="w-64 border-r border-zinc-700 flex flex-col min-w-0 sm:w-56 md:w-64"
          ref={fileTreeRef}
        >
          <div className="flex items-center justify-between px-3 py-2">
            <div className="text-sm font-semibold text-zinc-400 uppercase tracking-wider">
              Files
            </div>
            <button
              className="rounded p-1 text-xs text-zinc-500 hover:text-zinc-300 sm:hidden"
              onClick={() => setSidebarOpen(false)}
            >
              ✕
            </button>
          </div>
          <FileTree
            items={items}
            selectedPath={selectedPath}
            onSelect={setSelectedPath}
          />
        </div>
      )}
      <div className="flex-1 flex flex-col min-w-0">
        <ArtifactToolbar
          selectedPath={selectedPath}
          selectedHostPath={selectedHostPath}
          items={items}
          onDownloadFile={handleDownloadFile}
          onOpenInVSCode={handleOpenInVSCode}
        />
        <div className="flex-1 overflow-auto">
          {filesLoading || contentLoading ? (
            <div className="flex items-center justify-center h-full text-zinc-500">
              Loading...
            </div>
          ) : filesError ? (
            <div className="p-4 text-red-400">Error: {filesError}</div>
          ) : contentError ? (
            <div className="p-4 text-red-400">Error: {contentError}</div>
          ) : items.length === 0 && !filesLoading ? (
            <div className="flex items-center justify-center h-full text-zinc-500">
              No files in workspace
            </div>
          ) : selectedPath && content ? (
            <CodeViewer
              code={content}
              isBinary={isBinary}
              isTooLarge={isTooLarge}
            />
          ) : (
            <OutputView stdout="" stderr="" />
          )}
        </div>
      </div>
      {!sidebarOpen && (
        <button
          className="fixed left-2 top-2 z-50 rounded bg-zinc-800 p-1.5 text-zinc-400 hover:text-white shadow-lg sm:hidden"
          onClick={() => setSidebarOpen(true)}
        >
          ☰
        </button>
      )}
    </div>
  );
}