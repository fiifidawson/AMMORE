"use client";

import {
  ChakraProvider,
  createSystem,
  defaultConfig,
  defineConfig,
} from "@chakra-ui/react";

const config = defineConfig({
  theme: {
    tokens: {
      fonts: {
        heading: { value: "'Chillax', sans-serif" },
        body: { value: "'Chillax', sans-serif" },
      },
    },
  },
  globalCss: {
    body: {
      bg: "white",
      color: "black",
    },
  },
});

const system = createSystem(defaultConfig, config);

export function Provider({ children }: { children: React.ReactNode }) {
  return <ChakraProvider value={system}>{children}</ChakraProvider>;
}
