const base = import.meta.env.BASE_URL;

export function asset(path: string): string {
  return `${base}${path.replace(/^\//, '')}`;
}

export function route(path = ''): string {
  if (!path || path === '/') {
    return base;
  }
  return `${base}${path.replace(/^\//, '').replace(/\/$/, '')}/`;
}

export function absoluteUrl(path: string): string {
  const site = import.meta.env.SITE ?? 'https://kushals256.github.io';
  return new URL(asset(path), site).href;
}
