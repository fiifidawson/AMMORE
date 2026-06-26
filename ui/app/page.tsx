"use client";

import { Box, Button, Heading, Text } from "@chakra-ui/react";
import Link from "next/link";

export default function Home() {
  return (
    <Box>
      <Heading size="2xl" mb="2">
        AMMORE
      </Heading>
      <Text color="gray.600" mb="8">
        Agentic literature review. Ask a question over a corpus of documents,
        get a cited synthesis.
      </Text>
      <Link href="/create">
        <Button bg="black" color="white" borderRadius="0" size="lg">
          New review
        </Button>
      </Link>
    </Box>
  );
}
