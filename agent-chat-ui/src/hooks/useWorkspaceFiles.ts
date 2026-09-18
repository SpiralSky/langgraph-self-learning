import { useState, useEffect, useCallback } from "react";

const API_BASE = process.env.NEXT_PUBLIC_API_URL ?? "";

interface WorkspaceItem {
  name: string;
  type: "file" | "dir";
  path: string;
  host_path: string;
}

interface UseWorkspaceFilesReturn {
  items: WorkspaceItem[];
  loading: boolean;
  error: string | null;
  refetch: () => void;
}

export function useWorkspaceFiles(path = "."): UseWorkspaceFilesReturn {
  const [items, setItems] = useState<WorkspaceItem[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const fetchFiles = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const params = new URLSearchParams();
      if (path) params.set("path", path);
      const res = await fetch(`${API_BASE}/api/workspace/files?${params.toString()}`);
      if (!res.ok) {
        const body = await res.json().catch(() => ({}));
        throw new Error(body.detail ?? `HTTP ${res.status}`);
      }
      const data: WorkspaceItem[] = await res.json();
      setItems(data);
    } catch (e: any) {
      setError(e?.message ?? "Failed to fetch files");
    } finally {
      setLoading(false);
    }
  }, [path]);

  useEffect(() => {
    fetchFiles();
  }, [fetchFiles]);

  return { items, loading, error, refetch: fetchFiles };
}