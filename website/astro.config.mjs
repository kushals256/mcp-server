import { defineConfig } from 'astro/config';
import tailwind from '@astrojs/tailwind';

export default defineConfig({
  site: 'https://kushals256.github.io',
  base: '/mcp-server/',
  trailingSlash: 'always',
  integrations: [tailwind()],
});
