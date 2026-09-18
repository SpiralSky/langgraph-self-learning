import { useState, useEffect, useCallback } from "react";

const API_BASE = process.env.NEXT_PUBLIC_API_URL ?? "";
const MAX_DISPLAY_SIZE = 1 * 1024 * 1024;

interface UseFileContentReturn {
  content: string | null;
  loading: boolean;
  error: string | null;
  isBinary: boolean;
  isTooLarge: boolean;
  refetch: () => void;
}

export function useFileContent(path: string): UseFileContentReturn {
  const [content, setContent] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [isBinary, setIsBinary] = useState(false);
  const [isTooLarge, setIsTooLarge] = useState(false);

  const fetchFile = useCallback(async () => {
    if (!path) {
      setContent(null);
      setLoading(false);
      return;
    }
    setLoading(true);
    setError(null);
    setIsBinary(false);
    setIsTooLarge(false);
    try {
      const params = new URLSearchParams();
      params.set("path", path);
      const res = await fetch(`${API_BASE}/api/workspace/file?${params.toString()}`);
      if (!res.ok) {
        if (res.status === 403) {
          setIsBinary(true);
          setLoading(false);
          return;
        }
        const body = await res.json().catch(() => ({}));
        throw new Error(body.detail ?? `HTTP ${res.status}`);
      }
      const data: { name: string; content: string; size: number; host_path: string } =
        await res.json();
      if (data.size > MAX_DISPLAY_SIZE) {
        setIsTooLarge(true);
        setContent(data.content.slice(0, MAX_DISPLAY_SIZE));
      } else {
        setContent(data.content);
      }
    } catch (e: any) {
      setError(e?.message ?? "Failed to fetch file");
    } finally {
      setLoading(false);
    }
  }, [path]);

  useEffect(() => {
    fetchFile();
  }, [fetchFile]);

  return { content, loading, error, isBinary, isTooLarge, refetch: fetchFile };
}