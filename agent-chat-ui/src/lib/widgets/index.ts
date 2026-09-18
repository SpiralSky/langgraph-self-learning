/**
 * Barrel file that registers all built-in widgets with the WidgetRegistry.
 *
 * Importing this module (e.g. `import "@/lib/widgets"`) at the app entry
 * point is sufficient to make all widgets available for both the markdown
 * fence path and the generative‑UI (UIMessage) path.
 *
 * ## Adding a new widget
 *
 * 1. Create `src/lib/widgets/my-widget.tsx` exporting a React component
 *    that conforms to `WidgetDefinition.render`.
 * 2. Import it here and call `WidgetRegistry.register({ name: "my-widget", render: MyWidget })`.
 * 3. Use `:::my-widget` in chat or send a UIMessage with `name: "my-widget"`.
 */

import { WidgetRegistry } from "@/lib/widget-registry";
import { TextWidget } from "./text-widget";
import { CodeBlockWidget } from "./code-block-widget";

WidgetRegistry.register({
  name: "text",
  render: TextWidget,
});

WidgetRegistry.register({
  name: "code",
  render: CodeBlockWidget,
});