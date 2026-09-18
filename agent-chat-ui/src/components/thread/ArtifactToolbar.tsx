import { useCallback, useState } from "react";
import { Download, DownloadIcon, FolderOpen, Copy, Check } from "lucide-react";
import JSZip from "jszip";

interface ArtifactToolbarProps {
  selectedPath: string | null;
  selectedHostPath: string | null;
  items: { path: string; type: string; host_path?: string }[];
  onDownloadFile: (path: string) => void;
  onOpenInVSCode: () => void;
}

export function ArtifactToolbar({
  selectedPath,
  selectedHostPath,
  items,
  onDownloadFile,
  onOpenInVSCode,
}: ArtifactToolbarProps) {
  const [copied, setCopied] = useState(false);

  const handleCopyHostPath = useCallback(() => {
    if (!selectedHostPath) return;
    navigator.clipboard.writeText(selectedHostPath);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  }, [selectedHostPath]);

  const handleDownloadAll = useCallback(async () => {
    const zip = new JSZip();
    for (const item of items) {
      if (item.type !== "file") continue;
      try {
        const res = await fetch(
          `/api/workspace/file?path=${encodeURIComponent(item.path)}`,
        );
        if (!res.ok) continue;
        const data: { content: string } = await res.json();
        zip.file(item.path, data.content);
      } catch {
        // skip failed files
      }
    }
    const blob = await zip.generateAsync({ type: "blob" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "workspace.zip";
    a.click();
    URL.revokeObjectURL(url);
  }, [items]);

  return (
    <div className="flex items-center gap-2 border-b border-zinc-700 px-3 py-1.5">
      <button
        className="flex items-center gap-1 rounded px-2 py-1 text-xs font-medium text-zinc-300 hover:bg-zinc-700 hover:text-white"
        onClick={() => selectedPath && onDownloadFile(selectedPath)}
        disabled={!selectedPath}
      >
        <Download className="h-3.5 w-3.5" />
        Download
      </button>
      <button
        className="flex items-center gap-1 rounded px-2 py-1 text-xs font-medium text-zinc-300 hover:bg-zinc-700 hover:text-white"
        onClick={handleDownloadAll}
      >
        <DownloadIcon className="h-3.5 w-3.5" />
        Download All
      </button>
      <button
        className="flex items-center gap-1 rounded px-2 py-1 text-xs font-medium text-zinc-300 hover:bg-zinc-700 hover:text-white"
        onClick={onOpenInVSCode}
      >
        <FolderOpen className="h-3.5 w-3.5" />
        Open in VS Code
      </button>
      {selectedHostPath ? (
        <button
          className="flex items-center gap-1 rounded px-2 py-1 text-xs font-medium text-zinc-300 hover:bg-zinc-700 hover:text-white"
          onClick={handleCopyHostPath}
          title={selectedHostPath}
        >
          {copied ? (
            <Check className="h-3.5 w-3.5 text-green-400" />
          ) : (
            <Copy className="h-3.5 w-3.5" />
          )}
          <span className="max-w-32 truncate">{selectedHostPath}</span>
        </button>
      ) : null}
    </div>
  );
}