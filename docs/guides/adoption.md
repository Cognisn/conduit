# Adoption notes

These notes are for the first Cognisn products to move onto conduit, each replacing an AI layer of its own: VOR in its 0.2.0 release, SPARK in 0.4.0 and Termograph in 0.8.0. No product adopts conduit before conduit 0.1.0 is released; VOR's adoption is the first consumer test, and may lead to 0.1.x fixes.

The same division of labour applies to every product: **conduit supplies mechanisms, the product keeps its policy.** Prompts, guardrail rules, task-class declarations, approvals, credit and billing ledgers, conversation management and routing decisions stay in the product, above conduit. What moves into conduit is the plumbing underneath them: provider clients, the catalogue and residency checks, streaming, caching, usage, the agent loop and its budgets, compaction, skills and MCP.

## A checklist for any product

1. **Declare task classes and floors.** List the kinds of work the product asks of a model, and the least each needs (tools, caching, context window, labels). These replace any hard-coded model choice.
2. **Write the settings.** Providers, one assignment per task class, the residency policy the deployment must keep, and the capture policy (nothing by default). Move every credential behind a `secret://` reference and a resolver.
3. **Overlay the catalogue** with every model the product uses that conduit does not ship, and label entries for the floors.
4. **Supply the backends** the product's layers need: an event sink, a transcript store, an approval handler, a token store, a skill source. Run the conformance suites against each storage backend.
5. **Start once, at start-up**, so every refusal (an undeclared model, a floor, residency, a plain-text credential) stops the product before it serves anyone.
6. **Route every model call through the runtime** (`for_task`, the agent runner or the compaction and skills layers), so each is metered, held to its assignment and recorded.
7. **Remove the old layer's direct SDK use.** A product-side tripwire that refuses SDK imports outside conduit keeps it that way.

## VOR 0.2.0

VOR's AI layer is the seed conduit was generalised from, so its concepts carry across nearly one for one.

- **Replace `vor.ai` with conduit**: its providers, the agent loop and its budgets come from conduit's providers and `AgentRunner`.
- **VOR keeps its policy.** Its seven prompt layers, its rule files, its guardrails and its task-class declarations stay in VOR: the prompt layers become the `system` blocks VOR builds for each `AgentTask` (with a cache breakpoint after the stable layers), the guardrails stay in VOR's tool executor, approval handler and output filter, and the task classes are declared to `start`.
- **The notebook becomes two backends.** VOR's notebook writer becomes an event sink (the record of runs, calls and tool calls) and a transcript store (the conversation itself), so VOR keeps one record of each investigation while conduit stays storage-agnostic.
- **Skills and MCP.** VOR 0.1.0's own skills and MCP code is replaced by conduit's skills layer and MCP client.

## SPARK 0.4.0

SPARK's orchestrators are synchronous, on worker threads, and stay so: they call conduit through the sync facade.

- **Replace the LLM layer, the MCP manager, skills loading and the context compactor** with conduit's providers, MCP client, skills layer and compaction layer.
- **Use the sync facade correctly.** Keep one `LoopThread` for the process (or use `start_sync`), start conduit on it, and wrap every async object conduit hands back on that same loop. Never call the facade from code already running on conduit's loop (an event sink or a tool executor): await there instead. See [the sync facade](integration.md#the-sync-facade).
- **SPARK's repositories back the transcript store.** Implement `TranscriptStore` over SPARK's own database layer and run `TranscriptStoreSuite` against it. A database driver that needs a particular event loop (psycopg's async driver needs a selector loop on Windows) sets the suite's `loop_factory`.
- **Per-user budgets read the usage events.** Every model call's `model_call_finished` event carries the four token counts apart, the task class, the run and the correlation id; SPARK's per-user budgets sum those in its event sink. Within one run, `RunBudget` bounds what the agent loop may spend.
- **Approvals and conversations stay in SPARK.** SPARK's approval interface implements `ApprovalHandler`, and its conversation manager decides what `history` each run receives.

## Termograph 0.8.0

- **Route every Bedrock call through conduit's Bedrock provider**, with residency enforced: a `ResidencyPolicy` allowing only the scopes and Regions the deployment permits, so any assignment to a cross-Region or global inference profile is refused at start-up. Use the catalogue's in-Region model ids, or the bedrock-mantle adapter for single-Region Claude where bedrock-runtime offers only profiles.
- **Deny bearer-token calls where API keys are not used.** In the AWS accounts Termograph deploys to, deny `bedrock:CallWithBearerToken` in IAM, so no Bedrock API key minted from those credentials can be used, and configure the provider with the credential chain or access keys.
- **The credit ledger becomes an event sink.** conduit's provider registry is the single funnel every model call passes through, and emits exactly one `model_call_finished` event per call however it ends, with the final usage. The ledger records those events; the correlation id, run id and task class tie each to the work that caused it.
- **Routing stays in Termograph.** conduit does no routing of its own. Termograph's routing classifier stays in Termograph, above conduit, and selects among the catalogue entries Termograph declares; conduit enforces the floors and residency of whatever it selects.
