"use client";

import { useCallback, useEffect, useState } from "react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";

type PendingRequest = {
  filename: string;
  packages?: string[];
  reason?: string;
  proposed_at?: string;
  requested_by?: string;
  [key: string]: unknown;
};

export function PendingApprovals() {
  const [pending, setPending] = useState<PendingRequest[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const refresh = useCallback(async () => {
    try {
      const res = await fetch("/api/ledger", { cache: "no-store" });
      if (!res.ok) throw new Error(`GET /api/ledger -> ${res.status}`);
      const data = (await res.json()) as { pending?: PendingRequest[] };
      setError(null);
      setPending(data.pending ?? []);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    void refresh();
  }, [refresh]);

  const decide = async (filename: string, decision: "approve" | "reject") => {
    setError(null);
    try {
      const res = await fetch("/api/ledger", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ action: decision, filename }),
      });
      if (!res.ok) throw new Error(`POST /api/ledger -> ${res.status}`);
      await refresh();
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    }
  };

  return (
    <Card className="w-full max-w-2xl">
      <CardHeader>
        <CardTitle>Pending package requests</CardTitle>
      </CardHeader>
      <CardContent className="space-y-3">
        {error && <p className="text-sm text-red-600">{error}</p>}
        {loading ? (
          <p className="text-sm text-muted-foreground">Loading…</p>
        ) : pending.length === 0 ? (
          <p className="text-sm text-muted-foreground">No pending requests.</p>
        ) : (
          pending.map((req) => (
            <div
              key={req.filename}
              className="rounded-lg border p-3"
            >
              <div className="flex items-start justify-between gap-2">
                <div>
                  <p className="font-medium">
                    {(req.packages ?? []).join(", ") || req.filename}
                  </p>
                  {typeof req.reason === "string" && (
                    <p className="text-sm text-muted-foreground">
                      {req.reason}
                    </p>
                  )}
                </div>
                <div className="flex shrink-0 gap-2">
                  <Button
                    size="sm"
                    onClick={() => decide(req.filename, "approve")}
                  >
                    Approve
                  </Button>
                  <Button
                    size="sm"
                    variant="outline"
                    onClick={() => decide(req.filename, "reject")}
                  >
                    Reject
                  </Button>
                </div>
              </div>
            </div>
          ))
        )}
      </CardContent>
    </Card>
  );
}