# LinkedIn Context Haul — Compressed Learning Brief
_Captured 2026-06-15 (20:15–20:20). Source: 12 LinkedIn screenshots. Compressed by SharpEdge._

> Purpose: durable, skimmable compression of a screenshot batch so the
> raw images can be dropped. Two payloads: (A) **provenance proof** for
> Code Puppy, (B) an **agentic-AI + system-design study map**.

---

## A. THE HEADLINE — Code Puppy is award-winning at Walmart

**Daniel Danker** (EVP, AI Acceleration · Product · Design @ Walmart) — viral post:
**1,686 likes · 45 comments · 22 reposts.**

> "You don't have to be an engineer at Walmart to be a builder. Thanks to
> **John Choi** and **Michael Pfaffenberger**, **Code Puppy** helps us use AI to
> work faster, better and with more purpose... store managers can build their
> own scheduling dashboards in minutes; a merchandise associate can turn dense
> text into a useful graphic in seconds. We're all builders now."

- **Code Puppy** (the open-source agent SharpEdge runs on) **won Walmart's
  President's Innovation Award.** Authors: **Michael Pfaffenberger & John Choi.**
- Award shown on a keynote main-stage (name in lights) + trophy handoff photo.
- Michael Pfaffenberger commented on the thread (1st-degree to operator).
- **Takeaway:** SharpEdge's lineage is a real, exec-endorsed, award-winning
  internal Walmart builder tool. Use as credibility anchor.

## Related Walmart-AI culture signals
- **Laura Money** (Walmart, "AI Geek | SWE | Maker"): one-prompt autonomous build
  via a multi-agent orchestrator **"Medusa"** (Hub + Dev1/Dev2/Dev3 + Security
  agents in parallel) → produced **"TicBuddy"**, a CBIT app for kids with
  Tourette's. Tags: #claude #orchestration #agenticai #kindcode.
  → Direct cousin of SharpEdge's sub-agent model.
- **Neha Bora** (Walmart Global Tech, ML Eng): GHC 2024 braindate. Walmart AI
  spans legal, compliance, sustainability, **aviation (own fleet)** across
  10,000+ stores.

---

## B. STUDY MAP — Agentic AI + System Design (curated reference)

### Agentic-AI syllabus (Neo Kim "18 Concepts for AI Engineers" + Sai Vinay list)
AI Agents Workflow · How MCP Works · How RAG Works · Agentic Patterns 101 ·
LLM Evals · AI Coding Workflow 101 · ML System Design · **Multi-Agent
Architectures** · How AI Agents Work · Vector Database 101 · AI Agent Design ·
**AI Agent Memory, State & Consistency** · **Context Engineering Fundamentals** ·
AI Chat Assistant · LLM Concepts (deep dive) · Reinforcement Learning ·
Knowledge Q&A · OpenClaw.
→ **Most relevant to SharpEdge:** Multi-Agent Architectures, Agent Memory/State
  (cf. our Kennel), and Context Engineering (cf. this very compression).

### "25 Repos for AI Agent Developers" (Akhila)
awesome-generative-ai · ai-agents-for-beginners · RAG_Techniques · GenAI_Agents ·
AI-ML-Roadmap-from-scratch · Hands-On-LLMs · Prompt-Engineering-Guide ·
**Awesome MCP Servers** · 1000+ Pre-Built AI Agents · Agents-towards-Production ·
Hands-On-AI-Engineering · LLM Course · 500+ Real-Life AI Agent Use-Cases ·
Context Engineering A-to-Z · ML-For-Beginners · Neural Networks Zero-to-Hero ·
awesome-computer-vision · awesome-nlp · awesome-datascience · all-rl-algorithms ·
awesome-rl · Made-With-ML · Awesome LLM Apps · Awesome AI Apps · Agents.md.

### System-design "Master Tree" (Neo Kim, 1,325 likes) — 50-concept map
Scalability · Availability · Reliability · Latency · Throughput · DB · SQL-vs-NoSQL ·
Load Balancing · Caching · Cache Invalidation · API Design · REST · GraphQL · gRPC ·
Auth · Fault Tolerance · HA · CAP · Consistency Models · Replication · Erasure Coding ·
Consensus · Leader Election · Secrets Mgmt · RBAC · Sharding · Indexing ·
Denormalization · ACID/BASE · Event-Driven · Message Queue · Pub/Sub · Sync-vs-Async ·
Idempotency · Bulkhead · Retry Logic · Timeout · Service Discovery · API Gateway ·
Blue-Green · Canary · Feature Flags · Observability · Logging · Correlation ID ·
Monitoring · Alerting · Full-Text Search · Time Series.

### "System Design is not HARD — 15 resources" (Ankur Dhawan, Adobe MTS-2)
How X works: Google Docs · Stock Exchange · AWS S3 · Twitter Timeline ·
LLMs/ChatGPT · Uber dispatch (1M req/s) · Bluesky · Kafka · Slack ·
YouTube-on-MySQL (2.49B users) · Reddit · Spotify · URL Shortener · Tinder ·
Meta serverless (11.5M calls/s).
→ **SharpEdge-relevant:** "How Stock Exchange Works" + "How LLMs Work."

### Industry adjacency
- **Corey Scherrer** (Atlassian Sr ML Eng): RovoChat — LLM grounded in Jira/
  Confluence company knowledge beats general LLMs ("200+ hrs saved" onboarding).
  → Validates the "context-grounded agent" thesis SharpEdge already lives.

---

## SharpEdge action hooks
1. Credibility: cite Code Puppy → President's Innovation Award when framing SharpEdge.
2. Roadmap study targets: Context Engineering, Multi-Agent Architectures, Agent
   Memory/State, MCP servers, "How Stock Exchange Works."
3. Pattern mirror: Medusa (Hub+Dev+Security swarm) ≈ our sub-agent/Kennel design.
