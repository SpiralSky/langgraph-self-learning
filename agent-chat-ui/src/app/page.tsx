"use client";

import { MyAssistant } from "@/components/assistant-ui/MyAssistant";
import { Toaster } from "@/components/ui/sonner";
import React from "react";

export default function DemoPage(): React.ReactNode {
  return (
    <React.Suspense fallback={<div>Loading (layout)...</div>}>
      <a
        href="/approvals"
        className="fixed bottom-4 right-4 z-50 rounded-md border bg-background px-3 py-1.5 text-sm text-foreground shadow-sm hover:bg-muted"
      >
        Approvals
      </a>
      <Toaster />
      <MyAssistant />
    </React.Suspense>
  );
}