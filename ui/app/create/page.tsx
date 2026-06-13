"use client";

import {
  Box,
  Button,
  Flex,
  Heading,
  Input,
  Spinner,
  Stack,
  Text,
  Textarea,
} from "@chakra-ui/react";
import { useRouter } from "next/navigation";
import { useEffect, useRef, useState } from "react";
import {
  API,
  CorpusInfo,
  CorpusStatus,
  getJSON,
  postJSON,
  RunEvent,
} from "@/lib/api";

type Step = "corpus" | "question" | "plan" | "running";

const STAGES = ["Planning", "Retrieving", "Criticizing", "Writing"] as const;
const AGENT_STAGE: Record<string, (typeof STAGES)[number]> = {
  Retriever: "Retrieving",
  Critic: "Criticizing",
  Writer: "Writing",
};

export default function CreatePage() {
  const router = useRouter();
  const [step, setStep] = useState<Step>("corpus");

  // corpus
  const [path, setPath] = useState("");
  const [info, setInfo] = useState<CorpusInfo | null>(null);
  const [status, setStatus] = useState<CorpusStatus | null>(null);
  const [webOnly, setWebOnly] = useState(false);
  const [error, setError] = useState("");
  const fileInput = useRef<HTMLInputElement>(null);

  // question + plan
  const [question, setQuestion] = useState("");
  const [webSearch, setWebSearch] = useState(false);
  const [plan, setPlan] = useState<string[]>([]);
  const [planLoading, setPlanLoading] = useState(false);

  // run
  const [events, setEvents] = useState<RunEvent[]>([]);
  const [stage, setStage] = useState<(typeof STAGES)[number]>("Planning");

  useEffect(() => {
    getJSON<CorpusStatus>("/api/corpus/status").then(setStatus).catch(() => {});
  }, []);

  useEffect(() => {
    if (status?.status !== "indexing") return;
    const t = setInterval(() => {
      getJSON<CorpusStatus>("/api/corpus/status").then(setStatus);
    }, 3000);
    return () => clearInterval(t);
  }, [status?.status]);

  async function inspect(p: string) {
    setError("");
    try {
      setInfo(await postJSON<CorpusInfo>("/api/corpus/inspect", { path: p }));
    } catch (e) {
      setInfo(null);
      setError(String(e));
    }
  }

  async function upload(files: FileList) {
    setError("");
    const form = new FormData();
    Array.from(files).forEach((f) => form.append("files", f));
    const r = await fetch(`${API}/api/corpus/upload`, {
      method: "POST",
      body: form,
    });
    if (!r.ok) {
      setError(await r.text());
      return;
    }
    const { path: uploaded } = await r.json();
    setPath(uploaded);
    inspect(uploaded);
  }

  async function prepare() {
    setError("");
    try {
      await postJSON("/api/corpus/prepare", { path });
      setStatus({ status: "indexing", path, error: null });
    } catch (e) {
      setError(String(e));
    }
  }

  async function makePlan() {
    setPlanLoading(true);
    setError("");
    try {
      const r = await postJSON<{ questions: string[] }>("/api/plan", {
        question,
      });
      setPlan(r.questions);
      setStep("plan");
    } catch (e) {
      setError(String(e));
    } finally {
      setPlanLoading(false);
    }
  }

  async function go() {
    setError("");
    setEvents([]);
    setStep("running");
    try {
      const { run_id } = await postJSON<{ run_id: string }>("/api/run", {
        question,
        plan: plan.filter((q) => q.trim()),
        web_search: webSearch || webOnly,
        web_only: webOnly,
      });
      const es = new EventSource(`${API}/api/run/${run_id}/events`);
      es.onmessage = (m) => {
        const ev: RunEvent = JSON.parse(m.data);
        setEvents((prev) => [...prev, ev]);
        if (AGENT_STAGE[ev.agent]) setStage(AGENT_STAGE[ev.agent]);
        if (ev.kind === "done") {
          es.close();
          router.push(`/review/${ev.content}`);
        }
        if (ev.kind === "error") {
          es.close();
          setError(ev.content);
        }
      };
      es.onerror = () => es.close();
    } catch (e) {
      setError(String(e));
    }
  }

  const ready = webOnly || status?.status === "ready";

  return (
    <Box>
      <Heading size="xl" mb="6">
        New review
      </Heading>

      {/* step 1: corpus */}
      <Box border="1px solid black" p="4" mb="4" opacity={webOnly ? 0.5 : 1}>
        <Flex justify="space-between" align="center" mb="3">
          <Heading size="md">1. Corpus</Heading>
          <Button
            size="sm"
            variant={webOnly ? "solid" : "outline"}
            bg={webOnly ? "black" : "white"}
            color={webOnly ? "white" : "black"}
            borderRadius="0"
            borderColor="black"
            onClick={() => setWebOnly(!webOnly)}
          >
            {webOnly ? "Web only: ON" : "Skip corpus, web search only"}
          </Button>
        </Flex>
        <Flex gap="2" mb="2">
          <Input
            placeholder="Path to a folder or a .json file"
            value={path}
            onChange={(e) => setPath(e.target.value)}
            borderRadius="0"
            borderColor="black"
            disabled={webOnly}
          />
          <Button
            onClick={() => inspect(path)}
            variant="outline"
            borderRadius="0"
            borderColor="black"
            disabled={webOnly}
          >
            Inspect
          </Button>
        </Flex>
        <input
          type="file"
          multiple
          hidden
          ref={fileInput}
          onChange={(e) => e.target.files && upload(e.target.files)}
        />
        <Button
          size="sm"
          variant="outline"
          borderRadius="0"
          borderColor="black"
          mb="3"
          onClick={() => fileInput.current?.click()}
          disabled={webOnly}
        >
          ... or upload files
        </Button>

        {info && !webOnly && (
          <Box border="1px solid black" p="3" mb="3" bg="gray.50">
            <Text fontWeight="600">
              {info.kind === "json"
                ? `${info.usable}/${info.total} usable papers`
                : `${info.total} files`}
            </Text>
            {info.sample.slice(0, 6).map((s) => (
              <Text key={s} fontSize="xs" color="gray.600" lineClamp={1}>
                {s}
              </Text>
            ))}
          </Box>
        )}

        {!webOnly && (
          <Flex align="center" gap="3">
            <Button
              onClick={prepare}
              disabled={!info || status?.status === "indexing"}
              bg="black"
              color="white"
              borderRadius="0"
            >
              Index
            </Button>
            {status?.status === "indexing" && (
              <Flex align="center" gap="2">
                <Spinner size="sm" />
                <Text fontSize="sm">
                  Indexing (takes a few minutes the first time)...
                </Text>
              </Flex>
            )}
            {status?.status === "ready" && (
              <Text fontSize="sm" fontWeight="600">
                Corpus ready: {status?.path}
              </Text>
            )}
            {status?.status === "error" && (
              <Text fontSize="sm" color="red.600">
                {status.error}
              </Text>
            )}
          </Flex>
        )}
      </Box>

      {/* step 2: question */}
      <Box
        border="1px solid black"
        p="4"
        mb="4"
        opacity={ready ? 1 : 0.4}
        pointerEvents={ready ? "auto" : "none"}
      >
        <Heading size="md" mb="3">
          2. Question
        </Heading>
        <Textarea
          placeholder="Ask your question..."
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          borderRadius="0"
          borderColor="black"
          mb="3"
        />
        <Flex align="center" gap="3">
          <Button
            onClick={makePlan}
            disabled={!question.trim() || planLoading}
            bg="black"
            color="white"
            borderRadius="0"
          >
            {planLoading ? <Spinner size="sm" /> : "Generate plan"}
          </Button>
          {!webOnly && (
            <Button
              size="sm"
              variant={webSearch ? "solid" : "outline"}
              bg={webSearch ? "black" : "white"}
              color={webSearch ? "white" : "black"}
              borderRadius="0"
              borderColor="black"
              onClick={() => setWebSearch(!webSearch)}
            >
              Web fallback {webSearch ? "ON" : "OFF"}
            </Button>
          )}
        </Flex>
      </Box>

      {/* step 3: plan validation */}
      {(step === "plan" || step === "running") && (
        <Box border="1px solid black" p="4" mb="4">
          <Heading size="md" mb="3">
            3. Plan ({plan.length} sub-questions)
          </Heading>
          <Stack gap="2" mb="3">
            {plan.map((q, i) => (
              <Flex key={i} gap="2">
                <Input
                  value={q}
                  onChange={(e) =>
                    setPlan(plan.map((x, j) => (j === i ? e.target.value : x)))
                  }
                  borderRadius="0"
                  borderColor="black"
                  size="sm"
                />
                <Button
                  size="sm"
                  variant="outline"
                  borderRadius="0"
                  borderColor="black"
                  onClick={() => setPlan(plan.filter((_, j) => j !== i))}
                >
                  X
                </Button>
              </Flex>
            ))}
          </Stack>
          <Flex gap="2">
            <Button
              size="sm"
              variant="outline"
              borderRadius="0"
              borderColor="black"
              onClick={() => setPlan([...plan, ""])}
            >
              + Add
            </Button>
            <Button
              onClick={go}
              disabled={step === "running"}
              bg="black"
              color="white"
              borderRadius="0"
            >
              Go
            </Button>
          </Flex>
        </Box>
      )}

      {/* step 4: progress */}
      {step === "running" && (
        <Box border="1px solid black" p="4">
          <Flex gap="2" mb="4">
            {STAGES.map((s) => (
              <Box
                key={s}
                px="3"
                py="1"
                border="1px solid black"
                bg={s === stage ? "black" : "white"}
                color={s === stage ? "white" : "black"}
                fontSize="sm"
              >
                {s}
              </Box>
            ))}
          </Flex>
          <Box maxH="400px" overflowY="auto" fontSize="xs">
            {events.map((ev, i) => (
              <Box key={i} mb="2">
                <Text fontWeight="600">
                  {ev.agent} {ev.kind === "tool_call" ? "(tool)" : ""}
                </Text>
                <Text whiteSpace="pre-wrap" color="gray.700" lineClamp={6}>
                  {ev.content}
                </Text>
              </Box>
            ))}
            {events.length === 0 && (
              <Flex align="center" gap="2">
                <Spinner size="sm" />
                <Text>Running...</Text>
              </Flex>
            )}
          </Box>
        </Box>
      )}

      {error && (
        <Text color="red.600" mt="3" fontSize="sm" whiteSpace="pre-wrap">
          {error}
        </Text>
      )}
    </Box>
  );
}
