"use client";

import { MyAssistant } from "@/components/assistant-ui/MyAssistant";
import { SetupForm } from "@/components/assistant-ui/SetupForm";
import { Toaster } from "@/components/ui/sonner";
import { useChatConfig } from "@/hooks/useChatConfig";
import React from "react";

function AppContent() {
  const {
    apiUrl,
    assistantId,
    apiKey,
    authScheme,
    isAgentBuilder,
    showSetupForm,
    setConfig,
  } = useChatConfig();

  if (showSetupForm) {
    return (
      <SetupForm
        apiUrl={apiUrl}
        assistantId={assistantId}
        apiKey={apiKey}
        isAgentBuilder={isAgentBuilder}
        onSave={setConfig}
      />
    );
  }

  return (
    <MyAssistant
      key={`${apiUrl}|${assistantId}`}
      apiUrl={apiUrl}
      assistantId={assistantId}
      apiKey={apiKey}
      authScheme={authScheme}
    />
  );
}

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
      <AppContent />
    </React.Suspense>
  );
}