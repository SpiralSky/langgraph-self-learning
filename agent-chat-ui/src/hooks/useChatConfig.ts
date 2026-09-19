"use client";

import { useQueryState } from "nuqs";
import { useCallback, useState } from "react";

import { getApiKey, setApiKey } from "@/lib/api-key";

const AGENT_BUILDER_AUTH_SCHEME = "langsmith-api-key";

export type ChatConfigValues = {
  apiUrl: string;
  assistantId: string;
  apiKey: string;
  isAgentBuilder: boolean;
};

export type ChatConfig = ChatConfigValues & {
  authScheme: string;
  showSetupForm: boolean;
  setConfig: (values: Partial<ChatConfigValues>) => void;
};

/**
 * Reads the chat connection settings from URL params (apiUrl/assistantId/
 * authScheme via nuqs), with env-var and localStorage fallbacks, exactly like
 * the old StreamProvider: URL params win, then env vars; the API key is kept
 * in localStorage under `lg:chat:apiKey`.
 */
export function useChatConfig(): ChatConfig {
  const envApiUrl = process.env.NEXT_PUBLIC_API_URL;
  const envAssistantId = process.env.NEXT_PUBLIC_ASSISTANT_ID;
  const envAuthScheme = process.env.NEXT_PUBLIC_AUTH_SCHEME;

  const [apiUrl, setApiUrl] = useQueryState("apiUrl", {
    defaultValue: envApiUrl ?? "",
  });
  const [assistantId, setAssistantId] = useQueryState("assistantId", {
    defaultValue: envAssistantId ?? "",
  });
  const [authScheme, setAuthScheme] = useQueryState("authScheme", {
    defaultValue: envAuthScheme ?? "",
  });
  const [isAgentBuilder, setIsAgentBuilder] = useState(
    () =>
      (authScheme || envAuthScheme || "").toLowerCase() ===
      AGENT_BUILDER_AUTH_SCHEME,
  );
  const [apiKey, setApiKeyState] = useState(() => getApiKey() ?? "");

  const setConfig = useCallback(
    (values: Partial<ChatConfigValues>) => {
      if (values.apiUrl !== undefined) void setApiUrl(values.apiUrl);
      if (values.assistantId !== undefined)
        void setAssistantId(values.assistantId);
      if (values.isAgentBuilder !== undefined) {
        setIsAgentBuilder(values.isAgentBuilder);
        void setAuthScheme(
          values.isAgentBuilder
            ? AGENT_BUILDER_AUTH_SCHEME
            : envAuthScheme ?? "",
        );
      }
      if (values.apiKey !== undefined) {
        setApiKey(values.apiKey);
        setApiKeyState(values.apiKey);
      }
    },
    [setApiUrl, setAssistantId, setAuthScheme, envAuthScheme],
  );

  const finalApiUrl = apiUrl || envApiUrl || "";
  const finalAssistantId = assistantId || envAssistantId || "";
  const finalAuthScheme = authScheme || envAuthScheme || "";

  return {
    apiUrl: finalApiUrl,
    assistantId: finalAssistantId,
    apiKey,
    authScheme: finalAuthScheme,
    isAgentBuilder,
    showSetupForm: !finalApiUrl || !finalAssistantId,
    setConfig,
  };
}