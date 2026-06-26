"use client";

import { Box, Button, Flex, Stack, Text } from "@chakra-ui/react";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { useEffect, useState } from "react";
import { getJSON, Review } from "@/lib/api";

export function Sidebar() {
  const [reviews, setReviews] = useState<Review[]>([]);
  const [collapsed, setCollapsed] = useState(false);
  const pathname = usePathname();

  useEffect(() => {
    setCollapsed(localStorage.getItem("sidebarCollapsed") === "1");
  }, []);

  useEffect(() => {
    getJSON<Review[]>("/api/reviews").then(setReviews).catch(() => {});
  }, [pathname]);

  function toggle() {
    setCollapsed((c) => {
      const next = !c;
      localStorage.setItem("sidebarCollapsed", next ? "1" : "0");
      return next;
    });
  }

  // the home and create pages already show a primary "New review" call to
  // action, so the sidebar one would just duplicate it there.
  const showNewReview = pathname !== "/" && pathname !== "/create";

  if (collapsed) {
    return (
      <Box
        w="44px"
        minH="100vh"
        borderRight="1px solid black"
        p="2"
        position="sticky"
        top="0"
      >
        <Button
          aria-label="Expand sidebar"
          onClick={toggle}
          variant="ghost"
          size="sm"
          borderRadius="0"
          w="100%"
          p="0"
          title="Expand"
        >
          »
        </Button>
      </Box>
    );
  }

  return (
    <Box
      w="260px"
      minH="100vh"
      borderRight="1px solid black"
      p="4"
      position="sticky"
      top="0"
    >
      <Flex justify="space-between" align="center" mb="4">
        <Link href="/">
          <Text fontWeight="600" fontSize="lg">
            AMMORE
          </Text>
        </Link>
        <Button
          aria-label="Collapse sidebar"
          onClick={toggle}
          variant="ghost"
          size="sm"
          borderRadius="0"
          minW="auto"
          p="1"
          title="Collapse"
        >
          «
        </Button>
      </Flex>

      {showNewReview && (
        <Link href="/create">
          <Button
            w="100%"
            variant="outline"
            bg="white"
            color="black"
            borderColor="black"
            borderRadius="0"
            mb="6"
            _hover={{ bg: "gray.100" }}
          >
            New review
          </Button>
        </Link>
      )}

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
