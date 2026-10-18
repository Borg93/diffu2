import { defineConfig } from "vite";
import { svelte } from "@sveltejs/vite-plugin-svelte";
// Build to ../src/diffu_studio/webui so gr.Server can serve the compiled SPA at "/".
export default defineConfig({
  plugins: [svelte()],
  base: "/app/",
  build: { outDir: "../src/diffu_studio/webui", emptyOutDir: true },
});
