import React, { useState } from 'react';
import { Button, Dialog, EmptyState, ErrorBanner, Field, Inspector, ProgressBar, Select, StatusBadge, Tabs, Tooltip } from './design-system/primitives';
import { formatBytes, formatTokensPerSecond } from './design-system/format';
import type { Locale } from './i18n/catalog';
import { t, testId } from './i18n/catalog';

export function DesignGallery(props: { locale: Locale }): React.JSX.Element {
  const [dialogOpen, setDialogOpen] = useState(false);
  const [tab, setTab] = useState('controls');
  const [fieldValue, setFieldValue] = useState('mlx-community/Meta-Llama-3.1-8B-Instruct-4bit');
  return (
    <div className="screen-stack gallery-screen">
      <section className="screen-hero">
        <p className="eyebrow">{t(props.locale, 'gallery.issue')}</p>
        <h1 data-testid={testId('gallery.title')}>{t(props.locale, 'gallery.title')}</h1>
        <p data-testid={testId('gallery.subtitle')}>{t(props.locale, 'gallery.subtitle')}</p>
      </section>
      <Tabs active={tab} onChange={setTab} label={t(props.locale, 'gallery.tab.sections')} tabs={[
        { id: 'controls', label: t(props.locale, 'gallery.tab.controls'), panel: <ControlPanel locale={props.locale} value={fieldValue} onValueChange={setFieldValue} onOpenDialog={() => setDialogOpen(true)} /> },
        { id: 'states', label: t(props.locale, 'gallery.tab.states'), panel: <StatePanel locale={props.locale} /> },
        { id: 'data', label: t(props.locale, 'gallery.tab.data'), panel: <DataPanel locale={props.locale} /> },
      ]} />
      <Dialog open={dialogOpen} title={t(props.locale, 'models.delete.confirm.title')} onClose={() => setDialogOpen(false)} testId="gallery-dialog" closeLabel={t(props.locale, 'common.close')}>
        <p data-testid={testId('models.delete.confirm.body')}>{t(props.locale, 'models.delete.confirm.body', { model: fieldValue })}</p>
        <Field label={t(props.locale, 'models.delete.confirm.token_label')} value="" placeholder={t(props.locale, 'gallery.delete_token')} />
        <div className="dialog-actions"><Button onClick={() => setDialogOpen(false)}>{t(props.locale, 'common.cancel')}</Button><Button tone="danger" data-autofocus>{t(props.locale, 'common.delete')}</Button></div>
      </Dialog>
    </div>
  );
}

function ControlPanel(props: { locale: Locale; value: string; onValueChange: (value: string) => void; onOpenDialog: () => void }): React.JSX.Element {
  return (
    <div className="gallery-grid">
      <article className="surface-card">
        <h2>{t(props.locale, 'gallery.controls.title')}</h2>
        <div className="control-row"><Button tone="primary">{t(props.locale, 'gallery.controls.primary')}</Button><Button>{t(props.locale, 'gallery.controls.secondary')}</Button><Button tone="danger">{t(props.locale, 'gallery.controls.danger')}</Button><Button busy>{t(props.locale, 'gallery.controls.busy')}</Button></div>
        <Field label={t(props.locale, 'gallery.field.repo')} value={props.value} onChange={props.onValueChange} error={t(props.locale, 'gallery.field.repo_error')} testId="gallery-field" />
        <Select label={t(props.locale, 'gallery.select.native')} value="ready" onChange={() => undefined} options={[{ value: 'ready', label: t(props.locale, 'models.status.ready') }, { value: 'unloaded', label: t(props.locale, 'models.status.unloaded') }]} />
      </article>
      <article className="surface-card">
        <h2>{t(props.locale, 'gallery.overlays.title')}</h2>
        <p>{t(props.locale, 'gallery.long_cjk')}</p>
        <div className="control-row"><Tooltip label={t(props.locale, 'gallery.tooltip')}><Button>{t(props.locale, 'gallery.hover_focus')}</Button></Tooltip><Button tone="primary" onClick={props.onOpenDialog}>{t(props.locale, 'gallery.dialog.open')}</Button></div>
      </article>
    </div>
  );
}

function StatePanel(props: { locale: Locale }): React.JSX.Element {
  return (
    <div className="gallery-grid">
      <EmptyState title={t(props.locale, 'models.empty.title')} body={t(props.locale, 'models.empty.body')} action={<Button tone="primary">{t(props.locale, 'common.add_model')}</Button>} testId="gallery-empty" />
      <ErrorBanner title={t(props.locale, 'state.unauthorized.title')} body={t(props.locale, 'state.unauthorized.body')} action={<Button>{t(props.locale, 'common.enter_key')}</Button>} testId="gallery-unauthorized" />
      <ErrorBanner tone="warning" title={t(props.locale, 'gallery.states.load_failed')} body={t(props.locale, 'gallery.states.load_failed_body')} action={<Button>{t(props.locale, 'common.retry')}</Button>} />
      <ProgressBar label={t(props.locale, 'gallery.download')} detail={t(props.locale, 'activity.progress.indeterminate', { bytes: formatBytes(64 * 1024 * 1024, props.locale) })} />
    </div>
  );
}

function DataPanel(props: { locale: Locale }): React.JSX.Element {
  return (
    <div className="gallery-grid">
      <article className="surface-card table-card">
        <h2>{t(props.locale, 'gallery.data.title')}</h2>
        <table className="ds-table"><thead><tr><th>{t(props.locale, 'gallery.data.name')}</th><th>{t(props.locale, 'gallery.data.status')}</th><th>{t(props.locale, 'gallery.data.rate')}</th></tr></thead><tbody><tr><td><span className="truncate" title={t(props.locale, 'models.long_name')}>{t(props.locale, 'models.long_name')}</span></td><td><StatusBadge state="ready">{t(props.locale, 'models.status.ready')}</StatusBadge></td><td>{formatTokensPerSecond(39.4, props.locale)}</td></tr><tr><td>granite-4.0-h-tiny-4bit</td><td><StatusBadge state="unloaded">{t(props.locale, 'models.status.unloaded')}</StatusBadge></td><td>{formatTokensPerSecond(null, props.locale)}</td></tr></tbody></table>
      </article>
      <Inspector title={t(props.locale, 'gallery.data.inspector')}><p>{t(props.locale, 'models.unsupported.reason')}</p><StatusBadge state="failed">{t(props.locale, 'models.status.failed')}</StatusBadge></Inspector>
    </div>
  );
}
