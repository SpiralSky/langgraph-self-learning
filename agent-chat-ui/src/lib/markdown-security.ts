/**
 * Link/image URL sanitization for the markdown pipeline.
 *
 * react-markdown applies a `urlTransform` to every URL-bearing attribute
 * (`href`, `src`, ...) before rendering, and custom `a` / `img` components
 * re-apply it as a second layer of defence. Only relative URLs and a small
 * allow-list of safe protocols survive; anything else is dropped so hostile
 * schemes like `javascript:` or `data:` can never reach a browser sink.
 */

const SAFE_URL_PROTOCOLS = new Set(["http:", "https:", "mailto:", "tel:"]);

function isSafeOrRelativeUrl(value: string): boolean {
  const colon = value.indexOf(":");
  const slash = value.indexOf("/");
  const questionMark = value.indexOf("?");
  const numberSign = value.indexOf("#");

  if (
    colon === -1 ||
    (slash !== -1 && colon > slash) ||
    (questionMark !== -1 && colon > questionMark) ||
    (numberSign !== -1 && colon > numberSign)
  ) {
    return true;
  }

  return SAFE_URL_PROTOCOLS.has(value.slice(0, colon + 1).toLowerCase());
}

/**
 * Return a rendering-safe version of a markdown-injected URL.
 *
 * @param value - Raw URL from the markdown source (`href`, `src`, ...).
 * @returns The URL unchanged when it is relative or uses an allowed protocol,
 *   otherwise an empty string.
 */
export function sanitizeUrl(value: string | null | undefined): string {
  if (!value) {
    return "";
  }
  const trimmed = value.trim();
  return isSafeOrRelativeUrl(trimmed) ? trimmed : "";
}