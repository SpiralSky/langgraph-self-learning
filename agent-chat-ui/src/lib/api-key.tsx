const API_KEY_STORAGE_KEY = "lg:chat:apiKey";

export function getApiKey(): string | null {
  try {
    if (typeof window === "undefined") return null;
    return window.localStorage.getItem(API_KEY_STORAGE_KEY) ?? null;
  } catch {
    // no-op
  }

  return null;
}

export function setApiKey(key: string): void {
  try {
    if (typeof window === "undefined") return;
    window.localStorage.setItem(API_KEY_STORAGE_KEY, key);
  } catch {
    // no-op
  }
}
