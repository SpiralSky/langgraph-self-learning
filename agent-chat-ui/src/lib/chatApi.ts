import { Client } from "@langchain/langgraph-sdk";

export function createClient(): Client {
  const apiUrl =
    process.env.NEXT_PUBLIC_API_URL ??
    (typeof window !== "undefined"
      ? new URL("/api", window.location.href).href
      : "/api");
  return new Client({ apiUrl });
}