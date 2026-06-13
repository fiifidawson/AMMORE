"use client";

import { Box, Heading, Spinner, Text } from "@chakra-ui/react";
import { use, useEffect, useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { getJSON } from "@/lib/api";

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

  return (
    <Box>
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
          },
          "& p": { marginBottom: "0.75rem", lineHeight: 1.6 },
          "& ul, & ol": { paddingLeft: "1.5rem", marginBottom: "0.75rem" },
          "& li": { marginBottom: "0.4rem" },
          "& a": { textDecoration: "underline" },
          "& code": { background: "#f0f0f0", padding: "0 0.2rem" },
        }}
      >
        <ReactMarkdown remarkPlugins={[remarkGfm]}>{body}</ReactMarkdown>
      </Box>
    </Box>
  );
}
