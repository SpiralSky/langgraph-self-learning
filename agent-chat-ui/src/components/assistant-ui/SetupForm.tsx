"use client";

import { ArrowRight } from "lucide-react";
import { useState } from "react";

import { LangGraphLogoSVG } from "@/components/icons/langgraph";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { PasswordInput } from "@/components/ui/password-input";
import { Switch } from "@/components/ui/switch";

import type { ChatConfigValues } from "@/hooks/useChatConfig";

const DEFAULT_API_URL = "http://localhost:2024";
const DEFAULT_ASSISTANT_ID = "chat";

export function SetupForm({
  apiUrl,
  assistantId,
  apiKey,
  isAgentBuilder,
  onSave,
}: ChatConfigValues & {
  onSave: (values: ChatConfigValues) => void;
}) {
  const [agentBuilderEnabled, setAgentBuilderEnabled] =
    useState(isAgentBuilder);

  return (
    <div className="flex min-h-screen w-full items-center justify-center p-4">
      <div className="animate-in fade-in-0 zoom-in-95 bg-background flex max-w-3xl flex-col rounded-lg border shadow-lg">
        <div className="mt-14 flex flex-col gap-2 border-b p-6">
          <div className="flex flex-col items-start gap-2">
            <LangGraphLogoSVG className="h-7" />
            <h1 className="text-xl font-semibold tracking-tight">Agent Chat</h1>
          </div>
          <p className="text-muted-foreground">
            Welcome to Agent Chat! Before you get started, you need to enter
            the URL of the deployment and the assistant / graph ID.
          </p>
        </div>
        <form
          onSubmit={(e) => {
            e.preventDefault();

            const form = e.target as HTMLFormElement;
            const formData = new FormData(form);

            onSave({
              apiUrl: formData.get("apiUrl") as string,
              assistantId: formData.get("assistantId") as string,
              apiKey: formData.get("apiKey") as string,
              isAgentBuilder: agentBuilderEnabled,
            });

            form.reset();
          }}
          className="bg-muted/50 flex flex-col gap-6 p-6"
        >
          <div className="flex flex-col gap-2">
            <Label htmlFor="apiUrl">
              Deployment URL<span className="text-rose-500">*</span>
            </Label>
            <p className="text-muted-foreground text-sm">
              This is the URL of your LangGraph deployment. Can be a local, or
              production deployment.
            </p>
            <Input
              id="apiUrl"
              name="apiUrl"
              className="bg-background"
              defaultValue={apiUrl || DEFAULT_API_URL}
              required
            />
          </div>

          <div className="flex flex-col gap-2">
            <Label htmlFor="assistantId">
              Assistant / Graph ID<span className="text-rose-500">*</span>
            </Label>
            <p className="text-muted-foreground text-sm">
              This is the ID of the graph (can be the graph name), or
              assistant to fetch threads from, and invoke when actions are
              taken.
            </p>
            <Input
              id="assistantId"
              name="assistantId"
              className="bg-background"
              defaultValue={assistantId || DEFAULT_ASSISTANT_ID}
              required
            />
          </div>

          <div className="flex flex-col gap-2">
            <Label htmlFor="apiKey">LangSmith API Key</Label>
            <p className="text-muted-foreground text-sm">
              This is <strong>NOT</strong> required if using a local LangGraph
              server. This value is stored in your browser&apos;s local storage
              and is only used to authenticate requests sent to your LangGraph
              server.
            </p>
            <PasswordInput
              id="apiKey"
              name="apiKey"
              defaultValue={apiKey ?? ""}
              className="bg-background"
              placeholder="lsv2_pt_..."
            />
          </div>

          <div className="flex flex-col gap-3">
            <div className="flex items-center justify-between gap-4">
              <div className="flex flex-col gap-1">
                <Label htmlFor="agentBuilderEnabled">
                  Built with Agent Builder
                </Label>
                <p className="text-muted-foreground text-sm">
                  Enable this for Agent Builder deployments.
                </p>
              </div>
              <Switch
                id="agentBuilderEnabled"
                checked={agentBuilderEnabled}
                onCheckedChange={setAgentBuilderEnabled}
              />
            </div>
          </div>

          <div className="mt-2 flex justify-end">
            <Button type="submit" size="lg">
              Continue
              <ArrowRight className="size-5" />
            </Button>
          </div>
        </form>
      </div>
    </div>
  );
}