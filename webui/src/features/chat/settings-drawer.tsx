// Copyright 2026 Lablup Inc. Licensed under Apache-2.0.
// Chat settings, one header control away: the current conversation's name and system
// prompt, the next-turn parameters and local history. The shared Drawer keeps its
// content mounted while closed, so local-history persistence keeps running.
import React, { useEffect, useId, useRef } from 'react';
import { Drawer, Field } from '../../design-system/primitives';
import { t, type Locale } from '../../i18n/catalog';
import { TITLE_LIMIT } from './conversation-list';
import type { ChatConversation } from './history';
import { MAX_PROMPT_CHARACTERS } from './stream';

export interface SettingsDrawerProps {
  readonly open: boolean;
  readonly onClose: () => void;
  readonly locale: Locale;
  /** Changes whenever the drawer should open on the next-turn parameters (the overrides badge). */
  readonly parametersRequest: number;
  readonly conversation: ChatConversation | null;
  readonly disabled: boolean;
  readonly onConversationChange: (next: ChatConversation) => void;
  readonly parameters: React.ReactNode;
  readonly history: React.ReactNode;
}

// The shared Drawer moves focus to its close button one to a few frames after opening;
// claim the parameters heading after that, retrying until the panel can take focus. Each
// request is handled once, so a later plain open lands on the close button as usual.
function useFocusParameters(open: boolean, request: number, heading: React.RefObject<HTMLHeadingElement | null>): void {
  const handled = useRef(0);
  useEffect(() => {
    if (!open || request === handled.current) return;
    let frame = 0;
    let attempts = 0;
    let held = 0;
    const settle = (): void => {
      const node = heading.current;
      if (!node) return;
      if (document.activeElement !== node) node.focus();
      if (document.activeElement === node) {
        handled.current = request;
        if (held++ === 0 && typeof node.scrollIntoView === 'function') node.scrollIntoView({ block: 'start' });
        // Hold it for two more frames in case the drawer's own hand-off lands late.
        if (held < 3) frame = requestAnimationFrame(settle);
        return;
      }
      if (attempts++ < 20) frame = requestAnimationFrame(settle);
    };
    frame = requestAnimationFrame(settle);
    return () => cancelAnimationFrame(frame);
  }, [open, request, heading]);
}

export function SettingsDrawer(props: SettingsDrawerProps): React.JSX.Element {
  const { locale, conversation } = props;
  const conversationId = useId();
  const parametersId = useId();
  const historyId = useId();
  const parametersHeading = useRef<HTMLHeadingElement>(null);
  useFocusParameters(props.open, props.parametersRequest, parametersHeading);
  const [samplingBefore, samplingAfter = ''] = t(locale, 'chat.settings.sampling').split('{link}');
  return <Drawer open={props.open} onClose={props.onClose} title={t(locale, 'chat.settings.title')} closeLabel={t(locale, 'common.close')} testId="chat-settings-drawer" width="medium" side="end">
    <div className="chat-settings">
      <section className="chat-settings-section" aria-labelledby={conversationId}>
        <h3 id={conversationId}>{t(locale, 'chat.settings.summary')}</h3>
        {conversation ? <>
          <Field label={t(locale, 'chat.settings.name')} value={conversation.title} disabled={props.disabled} onChange={(title) => props.onConversationChange({ ...conversation, title: title.slice(0, TITLE_LIMIT) })} />
          <label className="ds-field"><span>{t(locale, 'chat.settings.system_prompt')}</span><textarea value={conversation.systemPrompt} maxLength={MAX_PROMPT_CHARACTERS} disabled={props.disabled} onChange={(event) => props.onConversationChange({ ...conversation, systemPrompt: event.target.value })} /></label>
          <p>{samplingBefore}<a href="#settings">{t(locale, 'nav.settings')}</a>{samplingAfter}</p>
        </> : <p className="chat-settings-hint">{t(locale, 'chat.settings.none')}</p>}
      </section>
      <section className="chat-settings-section" aria-labelledby={parametersId}>
        <h3 id={parametersId} ref={parametersHeading} tabIndex={-1}>{t(locale, 'chat.params.summary')}</h3>
        {props.parameters}
      </section>
      <section className="chat-settings-section" aria-labelledby={historyId}>
        <h3 id={historyId}>{t(locale, 'chat.privacy.summary')}</h3>
        {props.history}
      </section>
    </div>
  </Drawer>;
}
