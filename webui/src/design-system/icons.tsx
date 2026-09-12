import React from 'react';

export type IconName = 'models' | 'chat' | 'activity' | 'settings' | 'gallery' | 'command' | 'help' | 'menu' | 'close' | 'search' | 'warning' | 'key' | 'schema';

const paths: Record<IconName, React.ReactNode> = {
  models: <path d="M5 6.5A1.5 1.5 0 0 1 6.5 5h4A1.5 1.5 0 0 1 12 6.5v4a1.5 1.5 0 0 1-1.5 1.5h-4A1.5 1.5 0 0 1 5 10.5zm8 0A1.5 1.5 0 0 1 14.5 5h3A1.5 1.5 0 0 1 19 6.5v11a1.5 1.5 0 0 1-1.5 1.5h-3a1.5 1.5 0 0 1-1.5-1.5zm-8 8A1.5 1.5 0 0 1 6.5 13h4a1.5 1.5 0 0 1 1.5 1.5v3a1.5 1.5 0 0 1-1.5 1.5h-4A1.5 1.5 0 0 1 5 17.5z" />,
  chat: <path d="M5.5 7.5A2.5 2.5 0 0 1 8 5h8a2.5 2.5 0 0 1 2.5 2.5v5A2.5 2.5 0 0 1 16 15h-3.7l-3.6 3.2a.7.7 0 0 1-1.2-.52V15A2.5 2.5 0 0 1 5.5 12.5z" />,
  activity: <path d="M4.5 13h3l1.8-5.5 3.2 9 2.1-6.5h4.9" />,
  settings: <path d="M12 8.2a3.8 3.8 0 1 0 0 7.6 3.8 3.8 0 0 0 0-7.6Zm0-3.2v2m0 10v2m7-7h-2M7 12H5m11.95-4.95-1.42 1.42M8.47 15.53l-1.42 1.42m9.9 0-1.42-1.42M8.47 8.47 7.05 7.05" />,
  gallery: <path d="M5 6.5A1.5 1.5 0 0 1 6.5 5h11A1.5 1.5 0 0 1 19 6.5v11a1.5 1.5 0 0 1-1.5 1.5h-11A1.5 1.5 0 0 1 5 17.5zm3 2h8m-8 3h8m-8 3h5" />,
  command: <path d="M8 8.5A2.5 2.5 0 1 1 5.5 6 2.5 2.5 0 0 1 8 8.5Zm0 7A2.5 2.5 0 1 1 5.5 13 2.5 2.5 0 0 1 8 15.5Zm8-7A2.5 2.5 0 1 1 13.5 6 2.5 2.5 0 0 1 16 8.5Zm0 7A2.5 2.5 0 1 1 13.5 13 2.5 2.5 0 0 1 16 15.5Z" />,
  help: <path d="M9.5 9a2.5 2.5 0 1 1 4.4 1.62c-.68.72-1.9 1.18-1.9 2.38v.25m0 3.25h.01M12 21a9 9 0 1 0 0-18 9 9 0 0 0 0 18Z" />,
  menu: <path d="M5 7h14M5 12h14M5 17h14" />,
  close: <path d="m7 7 10 10M17 7 7 17" />,
  search: <path d="m16.5 16.5 3 3M11 18a7 7 0 1 1 0-14 7 7 0 0 1 0 14Z" />,
  warning: <path d="M12 4.8 21 19H3zm0 5.2v4m0 2.5h.01" />,
  key: <path d="M14.5 9.5A4.5 4.5 0 1 0 11 13.9l2.1 2.1H16v2h2v2h2v-2.9l-4.6-4.6a4.5 4.5 0 0 0-.9-3Z" />,
  schema: <path d="M6 5h7l5 5v9H6zm7 0v5h5M9 13h6M9 16h6M9 10h2" />,
};

export function Icon(props: { name: IconName; className?: string }): React.JSX.Element {
  return (
    <svg className={props.className ?? 'ds-icon'} aria-hidden="true" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
      {paths[props.name]}
    </svg>
  );
}
