"use client";

import { Box, Button, Heading, Stack, Text } from "@chakra-ui/react";
import Link from "next/link";
import { useEffect, useState } from "react";
import { getJSON, Review } from "@/lib/api";

export default function Home() {
  const [reviews, setReviews] = useState<Review[]>([]);

  useEffect(() => {
    getJSON<Review[]>("/api/reviews").then(setReviews).catch(() => {});
  }, []);

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
        <Button bg="black" color="white" borderRadius="0" size="lg" mb="10">
          New review
        </Button>
      </Link>

      <Heading size="md" mb="3">
        Recent reviews
      </Heading>
      <Stack gap="2">
        {reviews.slice(0, 8).map((r) => (
          <Link key={r.name} href={`/review/${r.name}`}>
            <Box border="1px solid black" p="3" _hover={{ bg: "gray.50" }}>
              <Text fontWeight="500">{r.title}</Text>
              <Text fontSize="xs" color="gray.500">
                {new Date(r.mtime * 1000).toLocaleString()}
              </Text>
            </Box>
          </Link>
        ))}
        {reviews.length === 0 && (
          <Text color="gray.500">
            No reviews yet. Make sure the API server is running:{" "}
            <code>python -m ammore.server</code>
          </Text>
        )}
      </Stack>
    </Box>
  );
}
