// Copyright 2025-2026 Lablup Inc. Licensed under the Apache License, Version 2.0.
import React from 'react';
import {
  Activity,
  Boxes,
  ChartNoAxesColumn,
  CircleQuestionMark,
  Command,
  Copy,
  FileCode,
  Info,
  KeyRound,
  List,
  type LucideIcon,
  Menu,
  MessageSquare,
  Paperclip,
  Pencil,
  RefreshCw,
  Search,
  Settings,
  SwatchBook,
  Trash,
  TriangleAlert,
  X,
} from 'lucide-react';

export type IconName = 'models' | 'chat' | 'activity' | 'settings' | 'gallery' | 'command' | 'help' | 'menu' | 'close' | 'search' | 'warning' | 'key' | 'schema' | 'list' | 'attach' | 'copy' | 'edit' | 'retry' | 'trash' | 'info' | 'details';

/** The lucide glyph behind each name: the same family, at lucide's default stroke, as the Astryx components' own icons. */
export const iconGlyphs: Readonly<Record<IconName, LucideIcon>> = {
  models: Boxes,
  chat: MessageSquare,
  activity: Activity,
  settings: Settings,
  gallery: SwatchBook,
  command: Command,
  help: CircleQuestionMark,
  menu: Menu,
  close: X,
  search: Search,
  warning: TriangleAlert,
  key: KeyRound,
  schema: FileCode,
  list: List,
  attach: Paperclip,
  copy: Copy,
  edit: Pencil,
  retry: RefreshCw,
  trash: Trash,
  info: Info,
  details: ChartNoAxesColumn,
};

export function Icon(props: { name: IconName; className?: string }): React.JSX.Element {
  const Glyph = iconGlyphs[props.name];
  return <Glyph className={props.className ?? 'ds-icon'} aria-hidden="true" />;
}

/** A product glyph as an SVG component, for slots that take an icon component (SideNavItem). */
export type GlyphComponent = React.ComponentType<React.SVGProps<SVGSVGElement>>;
const glyphs = new Map<IconName, GlyphComponent>();
export function glyph(name: IconName): GlyphComponent {
  let component = glyphs.get(name);
  if (!component) {
    const Lucide = iconGlyphs[name];
    const Glyph = (props: React.SVGProps<SVGSVGElement>): React.JSX.Element => <Lucide aria-hidden="true" {...props} />;
    Glyph.displayName = `Glyph(${name})`;
    component = Glyph;
    glyphs.set(name, component);
  }
  return component;
}
