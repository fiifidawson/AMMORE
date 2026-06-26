import type { Metadata } from "next";
import { Flex, Box } from "@chakra-ui/react";
import { Provider } from "@/components/Provider";
import { Sidebar } from "@/components/Sidebar";

export const metadata: Metadata = {
  title: "AMMORE",
  description: "Agentic literature review",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" suppressHydrationWarning>
      <head>
        <link
          href="https://api.fontshare.com/v2/css?f[]=chillax@400,500,600&display=swap"
          rel="stylesheet"
        />
      </head>
      <body>
        <Provider>
          <Flex>
            <Sidebar />
            <Box flex="1" p="8" maxW="900px">
              {children}
            </Box>
          </Flex>
        </Provider>
      </body>
    </html>
  );
}
