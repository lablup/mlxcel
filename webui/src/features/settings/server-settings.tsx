import './settings.css';
import { ContextCheck } from './context-check';
import React, { useEffect, useId, useState } from 'react';
import { t, type Locale } from '../../i18n/catalog';
import { lifecycleLabel } from '../../provider-surfaces';
import { ErrorBanner, Select } from '../../design-system/primitives';
import { formatSettingValue, type ModelProps, type SettingSpec, type SettingsResponse } from '../../api/settings';
import { LiveSettings } from './live-settings';
import { ProfileSettings } from './profile-settings';
import { recordServerSettings } from './server-defaults';
import { HelpTip, startupLabel } from './setting-control';
import { useSettingsTarget, type SettingsTarget } from './settings-target';

/** The app-wide model selection, scoped here to what Settings observes. Selecting never loads. */
function ModelSelector({ locale, target }: { locale: Locale; target: SettingsTarget }): React.JSX.Element {
  const { snapshot, actions } = target;
  return <Select locale={locale} label={t(locale, 'settings.server.model')} value={snapshot.selectedModelId ?? ''} options={[{ value: '', label: t(locale, 'settings.server.model.none') }, ...snapshot.catalog.map((entry) => ({ value: entry.identity.id, label: `${entry.identity.display_name} · ${lifecycleLabel(locale, entry.lifecycle.state)}` }))]} onChange={(value) => actions.selectModel(value === '' ? null : value)} hint={t(locale, 'settings.server.model.hint')} testId="settings-model-selector" />;
}

const unavailable = (locale: Locale): React.JSX.Element => <ErrorBanner tone="warning" title={t(locale, 'settings.server.unavailable.title')} body={t(locale, 'settings.server.unavailable.body')} />;
const noModel = (locale: Locale): React.JSX.Element => <ErrorBanner tone="info" title={t(locale, 'settings.server.no_model.title')} body={t(locale, 'settings.server.no_model.body')} />;
const liveDisabled = (locale: Locale): React.JSX.Element => <ErrorBanner tone="info" title={t(locale, 'settings.server.live_disabled.title')} body={t(locale, 'settings.server.live_disabled.body')} />;

/** Model tab: the selection, the next-load profile, then the live mutable values of a Ready model. */
export function ModelSettings({ locale }: { locale: Locale }): React.JSX.Element {
  const target = useSettingsTarget();
  if (!target.connected) return <div className="screen-stack settings-sections">{unavailable(locale)}</div>;
  const { selected, readyId, scope } = target;
  return (
    <div className="screen-stack settings-sections">
      <ModelSelector locale={locale} target={target} />
      <ProfileSettings key={selected?.identity.id ?? 'reusable'} model={selected} locale={locale} single={target.single} />
      {readyId === null ? noModel(locale) : target.liveEnabled ? <LiveSettings key={readyId} modelId={readyId} scope={scope} locale={locale} /> : liveDisabled(locale)}
    </div>
  );
}

const isRecord = (value: unknown): value is Readonly<Record<string, unknown>> => typeof value === 'object' && value !== null && !Array.isArray(value);

/** Where a nested value sits, for its catalog label; `null` inside the raw-name disclosure, where names stay raw. */
type LabelPath = readonly string[] | null;

/** A value inside a structured startup value: plain text, a list joined by commas, or further sub-rows. */
function nestedValue(locale: Locale, value: unknown, path: LabelPath): React.ReactNode {
  if (value === null || value === undefined || Array.isArray(value) && value.length === 0) return t(locale, 'settings.value.unset');
  if (isRecord(value)) return <StartupSubValues locale={locale} value={value} path={path} />;
  if (Array.isArray(value)) {
    // A list of structured items stays sub-rows per item; a list of plain values reads as one line.
    if (value.some((item) => isRecord(item) || Array.isArray(item))) return value.map((item, index) => <React.Fragment key={index}>{nestedValue(locale, item, path)}</React.Fragment>);
    return value.map(String).join(', ');
  }
  return String(value);
}

/**
 * Values the catalog has no label for (a field or sub-field newer than this UI), under their raw
 * names. Internal identifiers are not user copy, so they appear only inside this disclosure (#2334).
 */
function RawValues({ locale, rows, testId }: { locale: Locale; rows: readonly (readonly [string, unknown, React.ReactNode])[]; testId?: string }): React.JSX.Element {
  return (
    <details className="settings-disclosure" data-testid={testId}>
      <summary>{t(locale, 'settings.server.startup.other')}</summary>
      <dl className="settings-values settings-keys">{rows.map(([key, value, text]) => <div key={key} className={isRecord(value) ? 'settings-value-group' : undefined}><dt>{key}</dt><dd>{text}</dd></div>)}</dl>
    </details>
  );
}

/** A structured startup value (tensor parallel, prompt cache, CORS, ...) as labelled sub-rows rather than a JSON string. */
function StartupSubValues({ locale, value, path }: { locale: Locale; value: Readonly<Record<string, unknown>>; path: LabelPath }): React.JSX.Element {
  const rows = Object.entries(value).map(([key, item]) => ({ key, item, label: path === null ? key : startupLabel(locale, [...path, key]), childPath: path === null ? null : [...path, key] }));
  const labelled = rows.filter((row) => row.label !== null);
  const raw = rows.filter((row) => row.label === null);
  return (
    <>
      {labelled.length > 0 ? <dl className="settings-subvalues">{labelled.map(({ key, item, label, childPath }) => <div key={key} data-setting={key}><dt>{label}</dt><dd>{nestedValue(locale, item, childPath)}</dd></div>)}</dl> : null}
      {raw.length > 0 ? <RawValues locale={locale} rows={raw.map(({ key, item }) => [key, item, nestedValue(locale, item, null)] as const)} /> : null}
    </>
  );
}

/** Startup values as text: a disabled input per value would make the tab several screens tall. */
function StartupValues({ locale, specs, current }: { locale: Locale; specs: readonly SettingSpec[]; current: Readonly<Record<string, unknown>> }): React.JSX.Element {
  const titleId = useId();
  const reasons = new Map<string, string[]>();
  for (const spec of specs) {
    const reason = spec.reason ?? spec.help;
    const label = startupLabel(locale, [spec.name]);
    reasons.set(reason, [...reasons.get(reason) ?? [], label === null ? spec.name : `${label} (${spec.name})`]);
  }
  const valueOf = (spec: SettingSpec): unknown => Object.hasOwn(current, spec.name) ? current[spec.name] : spec.default;
  const text = (spec: SettingSpec, value: unknown, path: LabelPath): React.ReactNode => {
    if (value === null || value === undefined) return t(locale, 'settings.value.unset');
    return isRecord(value) ? <StartupSubValues locale={locale} value={value} path={path} /> : formatSettingValue(spec, value);
  };
  const rows = specs.map((spec) => ({ spec, value: valueOf(spec), label: startupLabel(locale, [spec.name]) }));
  const raw = rows.filter((row) => row.label === null);
  return (
    <section className="screen-stack" aria-labelledby={titleId}>
      <h2 id={titleId}>{t(locale, 'settings.live.startup')}</h2>
      <dl className="settings-values" data-testid="settings-startup-values">
        {rows.filter((row) => row.label !== null).map(({ spec, value, label }) => <div key={spec.name} data-setting={spec.name} className={isRecord(value) ? 'settings-value-group' : undefined}><dt>{label}</dt><dd>{text(spec, value, [spec.name])}</dd></div>)}
      </dl>
      {raw.length > 0 ? <RawValues locale={locale} testId="settings-startup-other" rows={raw.map(({ spec, value }) => [spec.name, value, text(spec, value, null)] as const)} /> : null}
      {/* Server reason strings are internal wording, kept inside an explicit disclosure. */}
      <details className="settings-disclosure">
        <summary>{t(locale, 'settings.server.startup.reasons')}</summary>
        <dl className="settings-reasons">{[...reasons].map(([reason, names]) => <div key={reason}><dt>{reason}</dt><dd>{names.join(', ')}</dd></div>)}</dl>
      </details>
    </section>
  );
}

/** Server tab: read-only facts about the selected Ready model's server. */
export function ServerSettings({ locale }: { locale: Locale }): React.JSX.Element {
  const target = useSettingsTarget();
  const { actions, readyId, scope, selected, liveEnabled } = target;
  const [props, setProps] = useState<ModelProps | null>(null);
  const [propsFailed, setPropsFailed] = useState(false);
  const [startup, setStartup] = useState<SettingsResponse | null>(null);
  const [startupFailed, setStartupFailed] = useState(false);
  const contextTitleId = useId();
  useEffect(() => {
    setProps(null); setPropsFailed(false);
    if (readyId === null) return;
    const controller = new AbortController();
    void actions.getModelProps(readyId, controller.signal).then((value) => { if (!controller.signal.aborted) setProps(value); }).catch(() => { if (!controller.signal.aborted) setPropsFailed(true); });
    return () => controller.abort();
  }, [actions, readyId, selected?.identity.revision]);
  useEffect(() => {
    setStartup(null); setStartupFailed(false);
    if (readyId === null || !liveEnabled) return;
    const controller = new AbortController();
    void actions.getSettings(readyId, controller.signal).then((value) => {
      if (controller.signal.aborted) return;
      setStartup(value);
      if (scope !== null) recordServerSettings(scope, value);
    }).catch(() => { if (!controller.signal.aborted) setStartupFailed(true); });
    return () => controller.abort();
  // The scope object is rebuilt on every render; its fields are the real dependency.
  }, [actions, readyId, scope?.instance, scope?.revision, liveEnabled]);
  if (!target.connected) return <div className="screen-stack settings-sections">{unavailable(locale)}</div>;
  const unknown = t(locale, 'settings.server.unknown');
  const readOnly = startup?.schema.filter((spec) => !spec.mutable) ?? [];
  return (
    <div className="screen-stack settings-sections">
      <ModelSelector locale={locale} target={target} />
      {readyId === null ? noModel(locale) : <>
        <section className="screen-stack" aria-labelledby={contextTitleId}>
          <div className="settings-heading"><h2 id={contextTitleId}>{t(locale, 'settings.server.context.title')}</h2><HelpTip label={t(locale, 'settings.help.context')} content={t(locale, 'settings.server.context.n_ctx_hint')} /></div>
          {propsFailed ? <ErrorBanner tone="info" title={t(locale, 'settings.server.props_unavailable.title')} body={t(locale, 'settings.server.props_unavailable.body')} testId="settings-props-unavailable" /> : null}
          <dl className="settings-values" data-testid="settings-context-values">
            <div><dt>{t(locale, 'settings.server.context.n_ctx')}</dt><dd data-testid="settings-context-n-ctx">{props?.nCtx?.toString() ?? unknown}</dd></div>
            <div><dt>{t(locale, 'settings.server.context.slots')}</dt><dd>{props?.totalSlots?.toString() ?? unknown}</dd></div>
            <div><dt>{t(locale, 'settings.server.context.kv_mode')}</dt><dd>{props?.kvCacheMode ?? unknown}</dd></div>
          </dl>
          <details className="settings-disclosure">
            <summary>{t(locale, 'settings.server.context.geometry')}</summary>
            <code className="settings-cli">{props === null || props.geometry === null ? t(locale, 'settings.server.context.geometry_unknown') : JSON.stringify(props.geometry)}</code>
          </details>
          <ContextCheck key={readyId} modelId={readyId} nCtx={props?.nCtx ?? null} locale={locale} />
        </section>
        {!liveEnabled ? liveDisabled(locale) : startupFailed ? <ErrorBanner tone="warning" title={t(locale, 'settings.server.startup.failed')} body={t(locale, 'settings.live.unavailable')} /> : startup === null ? null : <StartupValues locale={locale} specs={readOnly} current={startup.current} />}
      </>}
    </div>
  );
}
