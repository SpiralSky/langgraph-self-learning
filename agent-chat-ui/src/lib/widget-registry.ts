import React from "react";

/**
 * A descriptor for a custom widget.
 *
 * @typeParam TProps - The shape of the props the widget's render function accepts.
 */
export interface WidgetDefinition<TProps = Record<string, unknown>> {
  /** Unique widget name (used in the ::: fence syntax). */
  name: string;
  /**
   * React component that renders the widget.
   * Receives a `props` object containing parsed fence attributes
   * and an optional `children` string from the fence body.
   */
  render: React.FC<{ props: TProps; children?: string }>;
}

/**
 * Singleton registry of named widget renderers.
 *
 * Widgets are registered once (typically at import time via `src/lib/widgets/index.ts`)
 * and resolved by `WidgetRenderer` at render time.
 *
 * @example
 * ```ts
 * WidgetRegistry.register({
 *   name: "text",
 *   render: TextWidget,
 * });
 * ```
 */
export class WidgetRegistry {
  private static widgets = new Map<string, WidgetDefinition>();

  /**
   * Register a widget definition.
   *
   * @throws if a widget with the same name already exists
   * (use `override()` to replace an existing widget).
   */
  static register<TProps>(widget: WidgetDefinition<TProps>): void {
    const { name } = widget;
    if (WidgetRegistry.widgets.has(name)) {
      throw new Error(
        `Widget "${name}" is already registered. Use WidgetRegistry.override() to replace it.`,
      );
    }
    WidgetRegistry.widgets.set(name, widget as WidgetDefinition);
  }

  /**
   * Retrieve a registered widget by name.
   *
   * @returns The widget definition, or `undefined` if not found.
   */
  static get(name: string): WidgetDefinition | undefined {
    return WidgetRegistry.widgets.get(name);
  }

  /**
   * Check whether a widget with the given name is registered.
   */
  static has(name: string): boolean {
    return WidgetRegistry.widgets.has(name);
  }

  /**
   * List the names of all registered widgets.
   */
  static list(): string[] {
    return Array.from(WidgetRegistry.widgets.keys());
  }

  /**
   * Register or replace a widget definition.
   * Unlike `register()`, this will silently overwrite an existing entry.
   */
  static override<TProps>(widget: WidgetDefinition<TProps>): void {
    WidgetRegistry.widgets.set(widget.name, widget as WidgetDefinition);
  }
}