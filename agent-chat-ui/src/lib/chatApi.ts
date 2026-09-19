import { Client } from "@langchain/langgraph-sdk";

export type ChatClientOptions = {
  apiUrl?: string;
  apiKey?: string;
  authScheme?: string;
};

export function createClient(options: ChatClientOptions = {}): Client {
  const { apiUrl, apiKey, authScheme } = options;
  const resolvedApiUrl =
    apiUrl ||
    process.env.NEXT_PUBLIC_API_URL ||
    (typeof window !== "undefined"
      ? new URL("/api", window.location.href).href
      : "/api");
  return new Client({
    apiUrl: resolvedApiUrl,
    ...(apiKey ? { apiKey } : {}),
    ...(authScheme ? { defaultHeaders: { "X-Auth-Scheme": authScheme } } : {}),
  });
}

export async function checkGraphStatus(
  apiUrl: string,
  apiKey?: string | null,
  authScheme?: string,
): Promise<boolean> {
  try {
    const headers = new Headers();
    if (apiKey) headers.set("X-Api-Key", apiKey);
    if (authScheme) headers.set("X-Auth-Scheme", authScheme);

    const res = await fetch(`${apiUrl}/info`, {
      headers,
    });

    return res.ok;
  } catch (e) {
    console.error(e);
    return false;
  }
}