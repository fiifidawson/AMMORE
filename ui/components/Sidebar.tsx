"use client";

import { Box, Button, Stack, Text } from "@chakra-ui/react";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { useEffect, useState } from "react";
import { getJSON, Review } from "@/lib/api";

export function Sidebar() {
  const [reviews, setReviews] = useState<Review[]>([]);
  const pathname = usePathname();

  useEffect(() => {
    getJSON<Review[]>("/api/reviews").then(setReviews).catch(() => {});
  }, [pathname]);

  return (
    <Box
      w="260px"
      minH="100vh"
      borderRight="1px solid black"
      p="4"
      position="sticky"
      top="0"
    >
      <Link href="/">
        <Text fontWeight="600" fontSize="lg" mb="4">
          AMMORE
        </Text>
      </Link>
      <Link href="/create">
        <Button
          w="100%"
          bg="black"
          color="white"
          borderRadius="0"
          mb="6"
          _hover={{ bg: "gray.800" }}
        >
          New review
        </Button>
      </Link>
      <Text fontSize="sm" fontWeight="500" mb="2" color="gray.600">
        History
      </Text>
      <Stack gap="1">
        {reviews.map((r) => (
          <Link key={r.name} href={`/review/${r.name}`}>
            <Text
              fontSize="sm"
              lineClamp={2}
              p="1"
              _hover={{ bg: "gray.100" }}
              title={r.title}
            >
              {r.title}
            </Text>
          </Link>
        ))}
        {reviews.length === 0 && (
          <Text fontSize="sm" color="gray.500">
            No reviews yet
          </Text>
        )}
      </Stack>
    </Box>
  );
}
