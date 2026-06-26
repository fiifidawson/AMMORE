"use client";

import {
  Box,
  Flex,
  Heading,
  Link,
  Spinner,
  Stack,
  Text,
} from "@chakra-ui/react";
import { use, useEffect, useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { getJSON } from "@/lib/api";

const slug = (s: string) =>
  s
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/(^-|-$)/g, "");

export default function ReviewPage({
  params,
}: {
  params: Promise<{ name: string }>;
}) {
  const { name } = use(params);
  const [content, setContent] = useState<string | null>(null);
  const [error, setError] = useState("");

  useEffect(() => {
    getJSON<{ content: string }>(`/api/reviews/${name}`)
      .then((r) => setContent(r.content))
      .catch((e) => setError(String(e)));
  }, [name]);

  if (error) return <Text color="red.600">{error}</Text>;
  if (content === null) return <Spinner />;

  const lines = content.split("\n");
  const title = lines[0]?.replace(/^#\s*/, "") ?? name;
  const body = lines.slice(1).join("\n");
  const headings = body
    .split("\n")
    .filter((l) => l.startsWith("## "))
    .map((l) => l.replace(/^##\s*/, "").trim());

  return (
    <Flex gap="10" align="flex-start">
      <Box flex="1" minW="0">
        <Heading size="xl" mb="6">
          {title}
        </Heading>
        <Box
          css={{
            "& h2": {
              fontSize: "1.25rem",
              fontWeight: 600,
              marginTop: "1.5rem",
              marginBottom: "0.5rem",
              borderBottom: "1px solid black",
              scrollMarginTop: "1rem",
            },
            "& p": { marginBottom: "0.75rem", lineHeight: 1.6 },
            "& ul, & ol": { paddingLeft: "1.5rem", marginBottom: "0.75rem" },
            "& li": { marginBottom: "0.4rem" },
            "& a": { textDecoration: "underline" },
            "& code": { background: "#f0f0f0", padding: "0 0.2rem" },
          }}
        >
          <ReactMarkdown
            remarkPlugins={[remarkGfm]}
            components={{
              h2: ({ children }) => (
                <h2 id={slug(String(children))}>{children}</h2>
              ),
            }}
          >
            {body}
          </ReactMarkdown>
        </Box>
      </Box>

      {headings.length > 0 && (
        <Box
          as="nav"
          position="sticky"
          top="4"
          w="200px"
          flexShrink={0}
          fontSize="sm"
          display={{ base: "none", lg: "block" }}
        >
          <Text fontWeight="600" mb="2">
            On this page
          </Text>
          <Stack gap="1">
            {headings.map((h) => (
              <Link
                key={h}
                href={`#${slug(h)}`}
                color="gray.600"
                lineClamp={2}
                _hover={{ color: "black" }}
              >
                {h}
              </Link>
            ))}
          </Stack>
        </Box>
      )}
    </Flex>
  );
}
