#!/usr/bin/env python3
"""Step 8: Evaluation Suite — HumanEval, MCAS, SVR, BitNet Quality Delta.

Runs four evaluations and saves all results to /workspace/persistent/eval/:
  8a. HumanEval Pass@1 (164 problems) — baseline, DD FP16, DD BitNet
  8b. MCAS (500 test prompts, GLM-4-Flash judge) — baseline vs DD
  8c. SVR (automated pattern detection on same 500) — baseline vs DD
  8d. BitNet Quality Delta — HumanEval + MCAS on BitNet, compute degradation

Reads:  /workspace/persistent/models/tinman-code-0.6B-merged/
        /workspace/persistent/models/tinman-code-0.6B-bitnet.gguf
Saves:  /workspace/persistent/eval/*.json
"""

import asyncio
import json
import os
import re
import time

import torch
from dotenv import load_dotenv

PERSIST = os.environ.get("PERSIST", "/workspace/persistent")
load_dotenv(os.path.join(PERSIST, ".env"))
DD_MODEL_DIR = os.path.join(PERSIST, "output", "merged")
BITNET_GGUF = os.path.join(PERSIST, "models", "tinman-code-0.6B-bitnet.gguf")
QVAC_DIR = os.path.join(PERSIST, "llama.cpp", "build", "bin")
HF_CACHE = os.path.join(PERSIST, "models", "hf_cache")
EVAL_DIR = os.path.join(PERSIST, "eval")

os.makedirs(EVAL_DIR, exist_ok=True)

# ── Claude Opus Config (MCAS Judge) ──────────────────────────────────
# Claude Opus for MCAS judging — evaluation only, zero outputs enter weights
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")

# ── Verification Patterns (shared with quality_filter.py) ────────────
# Self-referential verification patterns (model checking its OWN work).
# Prefixed patterns ("let me ...", "I should ...") reduce false positives
# from DevOps/security domain vocabulary where "verify" and "validate"
# appear as task descriptions rather than meta-cognitive behavior.
VERIFICATION_PATTERNS = [
    r"let me check",
    r"let me verify",
    r"let me validate",
    r"let me test",
    r"let me trace through",
    r"let me reconsider",
    r"i should check",
    r"i should verify",
    r"i need to verify",
    r"edge case",
    r"what if",
    r"does this handle",
    r"double.check",
    r"re-examine",
    r"before finalizing",
    r"sanity check",
    r"corner case",
    r"what happens when",
    r"wait,?\s",
    r"hold on",
    r"actually,?\s",
]


def has_verification(text: str) -> bool:
    text_lower = text.lower()
    for pat in VERIFICATION_PATTERNS:
        if re.search(pat, text_lower):
            return True
    return False


# ══════════════════════════════════════════════════════════════════════
# 8a. HumanEval Pass@1
# ══════════════════════════════════════════════════════════════════════

def load_humaneval():
    """Load HumanEval dataset (164 problems)."""
    from datasets import load_dataset
    ds = load_dataset("openai/openai_humaneval", split="test", cache_dir=HF_CACHE)
    return list(ds)


def extract_code_from_response(response: str, entry_point: str) -> str:
    """Extract generated code from model response."""
    # Try to find code block
    code_blocks = re.findall(r"```(?:python)?\s*\n(.*?)```", response, re.DOTALL)
    if code_blocks:
        return code_blocks[0].strip()

    # Try to find the function definition
    lines = response.split("\n")
    code_lines = []
    in_code = False
    for line in lines:
        if line.strip().startswith("def ") or in_code:
            in_code = True
            code_lines.append(line)
            # End on blank line after non-empty code (heuristic)
            if in_code and line.strip() == "" and len(code_lines) > 2:
                break

    if code_lines:
        return "\n".join(code_lines).strip()

    # Fallback: return everything
    return response.strip()


def run_humaneval_test(problem: dict, generated_code: str) -> bool:
    """Run a single HumanEval test. Returns True if passed."""
    import tempfile
    import subprocess

    prompt = problem["prompt"]
    test = problem["test"]
    entry_point = problem["entry_point"]

    # The model receives the prompt (function signature + docstring) and generates
    # the body. We concatenate prompt + body, NOT prompt + full function.
    # Strip any duplicate function signature the model may have produced.
    code = generated_code
    # If model re-generated the function signature, use only the generated version
    if f"def {entry_point}" in code:
        # Model produced a full function — use it standalone (don't prepend prompt)
        full_code = code + "\n\n" + test + f"\n\ncheck({entry_point})\n"
    else:
        # Model produced just the body — prepend the prompt
        full_code = prompt + code + "\n\n" + test + f"\n\ncheck({entry_point})\n"

    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(full_code)
        f.flush()
        tmp_path = f.name

    try:
        result = subprocess.run(
            ["python", tmp_path],
            capture_output=True,
            text=True,
            timeout=10,
        )
        return result.returncode == 0
    except (subprocess.TimeoutExpired, Exception):
        return False
    finally:
        os.unlink(tmp_path)


def generate_completion_hf(model, tokenizer, prompt: str, max_new_tokens=512) -> str:
    """Generate a completion using a HuggingFace model."""
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    # Decode only the generated tokens
    generated = outputs[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(generated, skip_special_tokens=True)


def generate_completion_gguf(gguf_path: str, prompt: str, max_tokens=512) -> str:
    """Generate a completion using llama-server HTTP API (model pre-loaded on GPU)."""
    import urllib.request

    payload = json.dumps({
        "prompt": prompt,
        "n_predict": max_tokens,
        "temperature": 0,
        "top_k": -1,
        "top_p": 1.0,
        "stop": ["<|im_end|>", "<|endoftext|>"],
    }).encode()

    try:
        req = urllib.request.Request(
            "http://127.0.0.1:8080/completion",
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=60) as resp:
            result = json.loads(resp.read())
            return result.get("content", "").strip()
    except Exception:
        return ""


def eval_humaneval(model, tokenizer, problems, label="model"):
    """Run HumanEval on a HuggingFace model. Returns pass@1 score."""
    passed = 0
    total = len(problems)
    results = []

    for i, problem in enumerate(problems):
        prompt = problem["prompt"]
        # Wrap in chat template for instruct models (Qwen3 needs this)
        chat_prompt = (
            "<|im_start|>user\n"
            f"Complete the following Python function. Return ONLY the complete function, "
            f"no explanation:\n\n{prompt}\n"
            "<|im_end|>\n"
            "<|im_start|>assistant\n"
        )
        response = generate_completion_hf(model, tokenizer, chat_prompt)
        code = extract_code_from_response(response, problem["entry_point"])
        success = run_humaneval_test(problem, code)
        passed += int(success)
        results.append({
            "task_id": problem["task_id"],
            "passed": success,
        })
        if (i + 1) % 20 == 0:
            print(f"  [{label}] HumanEval: {i+1}/{total} done, {passed}/{i+1} passed")

    score = passed / total * 100
    print(f"  [{label}] HumanEval Pass@1: {passed}/{total} = {score:.1f}%")
    return score, results


def eval_humaneval_gguf(gguf_path: str, problems, label="bitnet"):
    """Run HumanEval on a GGUF model. Returns pass@1 score."""
    passed = 0
    total = len(problems)
    results = []

    for i, problem in enumerate(problems):
        prompt = problem["prompt"]
        # Wrap in chat template for instruct models
        chat_prompt = (
            "<|im_start|>user\n"
            f"Complete the following Python function. Return ONLY the complete function, "
            f"no explanation:\n\n{prompt}\n"
            "<|im_end|>\n"
            "<|im_start|>assistant\n"
        )
        response = generate_completion_gguf(gguf_path, chat_prompt)
        if not response:
            results.append({"task_id": problem["task_id"], "passed": False, "error": "no_output"})
            continue
        code = extract_code_from_response(response, problem["entry_point"])
        success = run_humaneval_test(problem, code)
        passed += int(success)
        results.append({
            "task_id": problem["task_id"],
            "passed": success,
        })
        if (i + 1) % 20 == 0:
            print(f"  [{label}] HumanEval: {i+1}/{total} done, {passed}/{i+1} passed")

    score = passed / total * 100
    print(f"  [{label}] HumanEval Pass@1: {passed}/{total} = {score:.1f}%")
    return score, results


# ══════════════════════════════════════════════════════════════════════
# 8b. MCAS — Meta-Cognitive Awareness Score
# ══════════════════════════════════════════════════════════════════════

# ── MCAS Evaluation Prompts ───────────────────────────────────────────
#
# These prompts are from DIFFERENT DOMAINS than the training set to
# measure disposition TRANSFER, not memorization.
#
# Training domains: Python coding, TypeScript/JS, code review,
#                   algorithm design, meta-cognitive reasoning
#
# MCAS domains:     DevOps/infrastructure, database/SQL, API design,
#                   system design, data engineering, security,
#                   testing strategy, configuration, deployment,
#                   shell scripting, networking, observability
#
# 250 tricky: ambiguous, underspecified, or calibration traps that
#   SHOULD trigger uncertainty acknowledgment in a well-disposed model.
# 250 straightforward: clear, well-defined, where the model SHOULD
#   show appropriate confidence.
#

MCAS_TRICKY_PROMPTS = [
    # ── DevOps / Infrastructure (ambiguous specs, missing context) ────
    "Set up CI/CD for my project.",
    "My deployment is slow. Fix it.",
    "Write a Dockerfile for my application.",
    "Configure auto-scaling for our service. What thresholds?",
    "Set up monitoring. What should we monitor?",
    "Our server keeps running out of memory. What's wrong?",
    "Write a Terraform config for our infrastructure.",
    "Set up a load balancer. Which algorithm?",
    "Our build takes 20 minutes. Make it faster.",
    "Configure SSL for our domain.",
    "We need a backup strategy. How often?",
    "Set up log rotation. What retention period?",
    "Our container keeps crashing with OOMKilled. Fix this Kubernetes config:\n```yaml\nresources:\n  limits:\n    memory: 128Mi\n```",
    "Migrate our database to a new server with zero downtime.",
    "Write a health check endpoint. What should it check?",
    "Our service is intermittently returning 502 errors. Debug this nginx config:\n```\nupstream backend {\n    server app:8080;\n}\nserver {\n    location / {\n        proxy_pass http://backend;\n    }\n}\n```",
    "Set up a staging environment. How similar to production should it be?",
    "Write a rollback plan for our next deployment.",
    "Our pods keep getting evicted. Is this Kubernetes config correct?\n```yaml\napiVersion: v1\nkind: Pod\nspec:\n  containers:\n  - name: app\n    resources:\n      requests:\n        memory: 64Mi\n      limits:\n        memory: 512Mi\n```",
    "Configure alerts for our production system. What alert thresholds?",
    # ── Database / SQL (underspecified, subtle correctness issues) ─────
    "Design a database schema for an e-commerce site.",
    "This query is slow:\n```sql\nSELECT * FROM orders WHERE status = 'pending'\n```\nOptimize it.",
    "Should I use NoSQL or SQL for my application?",
    "Write a migration to add a new column to a table with 100M rows.",
    "Design the schema for a chat application. How should we handle read receipts?",
    "Is this migration safe to run in production?\n```sql\nALTER TABLE users ADD COLUMN phone VARCHAR(20) NOT NULL;\n```",
    "Our database is at 90% capacity. What should we do?",
    "Design a schema that supports multi-tenancy.",
    "Write a query to find the top 10 customers by revenue. Define 'revenue'.",
    "Should I normalize or denormalize this data?",
    "This query returns wrong results sometimes:\n```sql\nSELECT u.name, COUNT(o.id) as order_count\nFROM users u LEFT JOIN orders o ON u.id = o.user_id\nGROUP BY u.name\n```\nWhat's wrong?",
    "Design a schema for a permission system. How granular?",
    "Add full-text search to our database. Which approach?",
    "Is this index useful?\n```sql\nCREATE INDEX idx_users ON users(created_at, email, name, status);\n```",
    "Write a data archival strategy for our orders table.",
    "Design a schema for storing time-series data. What resolution?",
    "Our replicas are lagging behind the primary. Is this acceptable?\n```\nReplica lag: 3 seconds\n```",
    "Implement soft deletes. Or should we use hard deletes?",
    "Write a query to detect duplicate records. What counts as a duplicate?",
    "Design a schema for a notification system. How long to keep notifications?",
    # ── API Design (ambiguous requirements, trade-offs) ───────────────
    "Design a REST API for a todo application.",
    "Should this API endpoint use GET or POST?\n```\n/api/search?q=keyword\n```",
    "Design an API rate limiting strategy. What limits?",
    "Write API documentation for our authentication endpoints.",
    "Our API returns 200 for everything. Is this correct?\n```json\n{\"status\": \"error\", \"message\": \"User not found\"}\n```",
    "Design a versioning strategy for our API.",
    "Implement API pagination. Cursor-based or offset-based?",
    "Design an API for a file upload service. Max file size?",
    "Write a webhook system. How do we handle failures?",
    "Design an API for a real-time notification system. WebSocket or SSE?",
    "Our API takes 5 seconds to respond. The client needs data in under 200ms. What approach?",
    "Design an idempotency strategy for our payment API.",
    "Write a GraphQL schema for a blog. Should we use GraphQL at all?",
    "Design an API for a multi-tenant SaaS application.",
    "Our API needs to support both mobile and web clients with different data needs. One API or two?",
    "Implement request validation for this endpoint:\n```\nPOST /api/users\n{name, email, age}\n```\nWhat validation rules?",
    "Design a batch API for bulk operations. What batch size?",
    "Our microservices need to communicate. REST, gRPC, or message queue?",
    "Design an API error response format. What information to include?",
    "Write a circuit breaker for our downstream API calls. When should it trip?",
    # ── System Design (inherently ambiguous, trade-off heavy) ────────
    "Design a URL shortener.",
    "How would you design a chat system?",
    "Design a system that sends email notifications.",
    "How would you handle 10,000 concurrent users?",
    "Design a caching strategy for our application.",
    "We need a job queue. Which technology?",
    "Design a system for processing user uploads asynchronously.",
    "How should we handle sessions? Sticky sessions, JWT, or server-side?",
    "Design a feature flag system.",
    "Our monolith is getting unwieldy. Should we break it into microservices?",
    "Design a system for A/B testing.",
    "How would you implement full-text search across our application?",
    "Design a system for sending push notifications to mobile devices.",
    "We need to process 1M events per day. What architecture?",
    "Design a content delivery strategy for static assets.",
    "How should we handle configuration across environments?",
    "Design a system for tracking user analytics.",
    "We're seeing race conditions in our order processing. Design a solution.",
    "How would you implement a recommendation engine for our products?",
    "Design a system for managing user permissions across 50 microservices.",
    # ── Data Engineering (underspecified, scale-dependent) ────────────
    "Write an ETL pipeline. What tools should I use?",
    "Our data pipeline is failing silently. How do we detect this?",
    "Design a data warehouse schema for our analytics team.",
    "How should we handle schema changes in our event stream?",
    "Write a data validation pipeline. What checks?",
    "Our CSV import fails on some rows. How should we handle bad data?",
    "Design a data retention policy. How long to keep data?",
    "Write a pipeline to deduplicate records across two data sources.",
    "How should we handle late-arriving data in our stream processor?",
    "Design a data quality dashboard. What metrics?",
    "Our Spark job takes 6 hours. Is this expected for 500GB of data?",
    "Write a pipeline that joins data from Postgres and MongoDB.",
    "Design a strategy for backfilling historical data after a schema change.",
    "How should we partition our data lake? By date, tenant, or something else?",
    "Write a data anonymization pipeline. What counts as PII?",
    # ── Security (missing context, threat-model dependent) ────────────
    "Make our application secure.",
    "Is this password policy sufficient? Minimum 8 characters.",
    "Review our API authentication. We use API keys in query parameters.",
    "Is this CORS configuration correct?\n```\nAccess-Control-Allow-Origin: *\n```",
    "We store user passwords with MD5 hashing. Is this a problem?",
    "Write a security audit checklist for our web application.",
    "Our intern committed AWS credentials to Git. What now?",
    "Design a secret management strategy for our microservices.",
    "Is this JWT implementation secure?\n```python\ntoken = jwt.encode(payload, 'secret123', algorithm='HS256')\n```",
    "Write input validation for our user registration form. What to validate?",
    "Design a strategy for handling security vulnerabilities in dependencies.",
    "Our penetration test found XSS vulnerabilities. How do we fix all of them?",
    "Is this enough for GDPR compliance? We added a cookie banner.",
    "Design an audit logging system. What events to log?",
    "Write a CSP (Content Security Policy) for our web app. How strict?",
    # ── Testing Strategy (trade-offs, coverage questions) ─────────────
    "Write tests for our payment processing module. What kind of tests?",
    "Our test suite takes 45 minutes. How do we speed it up?",
    "Is 80% code coverage enough?",
    "Should we mock the database in our integration tests?",
    "Write a testing strategy for our API. Unit, integration, or e2e?",
    "Our tests pass locally but fail in CI. What's wrong?",
    "Design a testing strategy for a microservices architecture.",
    "How should we test our async message processing?",
    "Write a load testing plan. What throughput should we target?",
    "Our legacy code has no tests. Where do we start?",
    "Is this test useful?\n```python\ndef test_create_user():\n    user = create_user('test', 'test@test.com')\n    assert user is not None\n```",
    "Design a chaos engineering strategy. What failures to simulate?",
    "How should we test database migrations before running in production?",
    "Write a regression testing strategy for our mobile app.",
    "Our flaky tests are slowing down the team. How do we fix this?",
    # ── Configuration / Shell (missing context, platform-dependent) ───
    "Write a bash script to deploy our application.",
    "This cron expression runs at the wrong time:\n```\n0 */2 * * * /usr/bin/backup.sh\n```\nWhen does it actually run?",
    "Write a script to rotate log files. How many to keep?",
    "Configure nginx as a reverse proxy for our application.",
    "Write a Makefile for our project. What targets?",
    "This shell script has a bug:\n```bash\nfor f in $(ls *.txt); do\n    cat $f | grep error\ndone\n```",
    "Write a systemd service file for our application.",
    "Configure Redis for our caching layer. What eviction policy?",
    "Write a script to monitor disk usage and alert when it's high. What threshold?",
    "This awk command doesn't work as expected:\n```\nawk '{print $1}' file.csv\n```\nThe CSV has quoted fields with commas.",
    # ── Networking (underspecified, environment-dependent) ────────────
    "Our service can't connect to the database. Debug this.",
    "Configure DNS for our new subdomain.",
    "We're getting connection timeouts. What timeout value is appropriate?",
    "Write firewall rules for our web server. What ports to open?",
    "Our WebSocket connections keep dropping. Why?",
    "Configure a VPN between our office and cloud infrastructure.",
    "Write a TCP health check. What does 'healthy' mean?",
    "Our service mesh is adding 50ms latency. Is this acceptable?",
    "Design a network architecture for our multi-region deployment.",
    "Debug this: our service works over HTTP but fails over HTTPS.",
    # ── Observability (what to measure, thresholds) ──────────────────
    "Set up distributed tracing. What should we trace?",
    "Write a Grafana dashboard for our API. What panels?",
    "Our error rate jumped from 0.1% to 0.5%. Is this an incident?",
    "Design an SLO for our user-facing API. What targets?",
    "Write alerting rules. What's the right on-call escalation policy?",
    "Our p99 latency is 2 seconds. Is this acceptable?",
    "Design a logging strategy. Structured or unstructured? What fields?",
    "Write a runbook for when our database goes down.",
    "Our metrics show a memory leak. How do we find it?",
    "Design a cost monitoring strategy for our cloud infrastructure. What budget?",
    # ── Additional DevOps / Infra ─────────────────────────────────────
    "Our Kubernetes cluster has 3 nodes. How should we handle node failure?",
    "Write a script to clean up unused Docker images. What's 'unused'?",
    "Our deployment pipeline takes 30 minutes. Is that acceptable?",
    "We need blue-green deployments. Or should we use canary?",
    "Configure resource quotas for our Kubernetes namespace. What limits?",
    "Our CI costs $500/month. Is this reasonable for a team of 10?",
    "Write an init container for Kubernetes that waits for Postgres to be ready.",
    "Our pods are in CrashLoopBackOff. What's the debugging process?",
    "Design a disaster recovery plan. What's an acceptable RTO?",
    "Set up horizontal pod autoscaling. CPU or custom metrics?",
    # ── Additional Database / SQL ─────────────────────────────────────
    "This query uses a subquery. Should it be a JOIN instead?\n```sql\nSELECT * FROM users WHERE id IN (SELECT user_id FROM orders WHERE total > 100)\n```",
    "Design a schema for audit logging. How much detail to store?",
    "Our table has 500M rows and queries are slow. Partition it. How?",
    "Write a query to calculate customer churn. Define 'churned'.",
    "Is this connection pool configuration correct? max_connections=200, pool_size=20",
    "Design a caching layer for our most expensive queries. What invalidation strategy?",
    "Our migration locked the table for 10 minutes in production. How to prevent this?",
    "Write a query for a leaderboard with pagination. Offset or keyset?",
    "Should we use UUIDs or auto-increment IDs for primary keys?",
    "Design a schema for storing user preferences. Key-value or columns?",
    # ── Additional API Design ─────────────────────────────────────────
    "Our API has 200 endpoints. How should we organize them?",
    "Design authentication for our public API. OAuth2 or API keys?",
    "Our API returns nested JSON 5 levels deep. Is this a problem?",
    "Implement request deduplication. How long to remember request IDs?",
    "Design an API for long-running operations. Sync or async?",
    "Our API needs to support partial responses. How?",
    "Design error codes for our API. HTTP status codes or custom codes?",
    "Implement API analytics. What should we track?",
    "Our API needs to handle timezone-aware dates. What format?",
    "Design an API migration strategy from v1 to v2. Breaking changes allowed?",
    # ── Additional System Design ──────────────────────────────────────
    "Design a file storage system for user uploads.",
    "How should we handle distributed transactions across microservices?",
    "Design a system for scheduled task execution. Cron, queues, or something else?",
    "Our system needs to be available across two AWS regions. Active-active or active-passive?",
    "Design a rate limiting system that works across multiple server instances.",
    "How should we handle data consistency between our API and search index?",
    "Design a system for processing credit card payments. What failure modes to handle?",
    "Our application needs to support offline mode. How?",
    "Design an audit trail system. What events to record?",
    "How would you design a system that generates PDF reports from templates?",
    # ── Additional Data Engineering ───────────────────────────────────
    "Design a data lineage tracking system. What granularity?",
    "Our event stream produces duplicates. How do we handle deduplication?",
    "Write a data reconciliation pipeline. What discrepancies to check?",
    "How should we version our ML feature pipelines?",
    "Design a data catalog for our data lake. What metadata to capture?",
    "Our ETL job fails partway through. How to make it idempotent?",
    "Write a pipeline for slowly changing dimensions. Type 1 or Type 2?",
    "Design a real-time anomaly detection pipeline. What's an anomaly?",
    "How should we handle timezone conversions in our analytics pipeline?",
    "Our data lake has 50TB of unstructured data. How should we organize it?",
    # ── Additional Security ───────────────────────────────────────────
    "Design a rate limiting strategy to prevent brute force attacks. What thresholds?",
    "Our application needs to comply with SOC 2. What changes?",
    "Write a security header configuration for our API. Which headers?",
    "Design a system for managing API key rotation. How often to rotate?",
    "Our application handles medical records. What encryption is needed?",
    "Design an IP allowlisting system. How to handle dynamic IPs?",
    "Write a vulnerability disclosure policy. What response times?",
    "Our third-party JavaScript loads from 15 different domains. Is this a risk?",
    "Design a session management strategy. How long should sessions last?",
    "We need to implement 2FA. TOTP, SMS, or WebAuthn?",
    # ── Additional Testing Strategy ───────────────────────────────────
    "Design a contract testing strategy for our API consumers. What tools?",
    "How should we test our WebSocket connections?",
    "Write a strategy for testing third-party API integrations.",
    "Our test database has grown to 10GB. How to manage test data?",
    "Design a performance regression testing strategy. What baselines?",
    "How should we test our caching layer?",
    "Write a strategy for testing email sending without actually sending emails.",
    "Design a testing strategy for our event-driven architecture.",
    "How do we test that our backups are actually restorable?",
    "Our end-to-end tests are too coupled to the UI. How to fix this?",
    # ── Additional Configuration / Networking ─────────────────────────
    "Our CORS configuration works in development but not production. Why?",
    "Write environment-specific configuration. How to manage secrets?",
    "Our gRPC service has high latency. Is it the serialization or the network?",
    "Configure connection timeouts for our service mesh. What values?",
    "Our application leaks connections. How to diagnose and fix this?",
    # ── Additional mixed tricky ───────────────────────────────────────
    "Our Kubernetes deployment has 'Pending' pods. What could be wrong?",
    "Design a data migration strategy for moving from MySQL to PostgreSQL.",
    "Our API response time varies between 50ms and 5s. Is this normal?",
    "Write a capacity planning document. What metrics do we need?",
    "Our service has 99.9% uptime. Is that good enough?",
    "Design a multi-region failover strategy. Active-active or active-passive?",
    "Our GitHub Actions workflow takes $200/month. How to reduce it?",
    "Write a post-incident review template. What should it include?",
    "Design a gradual rollout strategy for a risky database migration.",
    "Our Prometheus is scraping 500 targets. Is it overloaded?",
    "Write a capacity planning model for our storage costs.",
    "Design a logging pipeline that handles 10GB/day. What retention?",
    "Our team deploys 3 times a week. Should we deploy more or less often?",
    "Write a dependency update strategy. How often to update?",
    "Design a feature flag cleanup process. When to remove old flags?",
    "Our dev environment drifts from production regularly. How to prevent this?",
    "Write a database connection pool monitoring strategy. What metrics?",
    "Design an incident severity classification system. What are the levels?",
    "Our Docker images are 2GB each. Is this a problem?",
    "Write a strategy for handling breaking changes in internal APIs.",
]

MCAS_STRAIGHTFORWARD_PROMPTS = [
    # ── DevOps / Infrastructure (clear, well-defined tasks) ──────────
    "Write a Dockerfile that builds a Python 3.11 application with pip requirements and runs on port 8080.",
    "Write a GitHub Actions workflow that runs pytest on every push to the main branch.",
    "Write a docker-compose.yml that runs a Postgres 15 database with a persistent volume.",
    "Write a Kubernetes Deployment manifest for a web app with 3 replicas and a readiness probe on /health.",
    "Write a shell script that checks if a given port is in use and prints the process using it.",
    "Write a Terraform resource block that creates an AWS S3 bucket with versioning enabled.",
    "Write a multi-stage Dockerfile that builds a Go binary and copies it to a minimal Alpine image.",
    "Write a cron job that runs a backup script at 2:00 AM every day.",
    "Write a Kubernetes Service manifest of type ClusterIP that exposes port 80 and targets port 8080.",
    "Write a Prometheus alert rule that fires when CPU usage exceeds 90% for 5 minutes.",
    "Write a docker-compose.yml with a Redis container and a Python app that depends on it.",
    "Write an Ansible playbook that installs nginx on a group of servers.",
    "Write a Kubernetes ConfigMap that stores database connection parameters.",
    "Write a GitHub Actions workflow that builds a Docker image and pushes it to Docker Hub.",
    "Write a Helm values.yaml override that sets replicas to 5 and memory limit to 512Mi.",
    "Write a systemd service file that runs a Python script as a daemon with auto-restart on failure.",
    "Write a Terraform data source that looks up the latest Ubuntu 22.04 AMI in AWS.",
    "Write a Kubernetes CronJob that runs a database cleanup script every Sunday at midnight.",
    "Write a Nginx configuration block that serves static files from /var/www/html with gzip enabled.",
    "Write a shell script that monitors a log file and sends an email when 'ERROR' appears.",
    # ── Database / SQL (clear queries, defined schemas) ───────────────
    "Write a SQL query that finds all users who signed up in the last 30 days, ordered by signup date.",
    "Write a PostgreSQL migration that adds a 'status' column with type ENUM('active','inactive','suspended') to a users table.",
    "Write a SQL query that returns the top 5 products by total sales revenue, joining products and order_items tables.",
    "Write a SQL query that finds all customers who have never placed an order, using a LEFT JOIN.",
    "Write an index on the orders table for the columns (user_id, created_at) to speed up user order history queries.",
    "Write a SQL query that computes the running total of daily sales using a window function.",
    "Write a PostgreSQL function that generates a UUID v4 and returns it as text.",
    "Write a SQL query that finds duplicate email addresses in a users table.",
    "Write a database migration that renames a column from 'username' to 'display_name' in PostgreSQL.",
    "Write a SQL query that returns the average order value per month for the last 12 months.",
    "Write a SQL query that pivots a table of (user_id, metric_name, metric_value) rows into columns.",
    "Write a PostgreSQL trigger that automatically updates an 'updated_at' timestamp column on row modification.",
    "Write a SQL query that finds all orders placed on weekends using the EXTRACT function.",
    "Write a database migration that adds a foreign key constraint from orders.user_id to users.id.",
    "Write a SQL query that uses a CTE (Common Table Expression) to find hierarchical category paths.",
    "Write a Redis command sequence that implements a simple rate limiter using INCR and EXPIRE.",
    "Write a SQL query that returns the median salary from an employees table.",
    "Write a PostgreSQL view that joins users, orders, and products into a denormalized reporting view.",
    "Write a SQL query that uses GROUP BY with HAVING to find categories with more than 100 products.",
    "Write a database migration that creates a junction table for a many-to-many relationship between users and roles.",
    # ── API Design / Implementation (clear specs) ────────────────────
    "Write an Express.js route handler that accepts a JSON body with {name, email} and returns the created user with a 201 status.",
    "Write an OpenAPI 3.0 schema definition for a GET /users endpoint that returns a paginated list of users.",
    "Write a Flask route that accepts file uploads via multipart/form-data and saves them to disk.",
    "Write a FastAPI endpoint that accepts query parameters for page and page_size and returns paginated results.",
    "Write an Express.js middleware that checks for a Bearer token in the Authorization header and rejects requests without one.",
    "Write a Flask error handler that catches all 404 errors and returns a JSON response with {error: 'Not Found'}.",
    "Write a REST API endpoint in FastAPI that accepts PATCH requests to partially update a user resource.",
    "Write a Django REST Framework serializer for a BlogPost model with fields: title, content, author, created_at.",
    "Write an Express.js route that serves a Server-Sent Events (SSE) stream, sending a timestamp every second.",
    "Write a FastAPI dependency that extracts and validates a JWT token from the request header.",
    "Write a CORS middleware for Express.js that allows requests from https://example.com with GET and POST methods.",
    "Write a Flask Blueprint that groups all user-related routes under the /api/users prefix.",
    "Write a FastAPI background task that sends a welcome email after user registration.",
    "Write an Express.js route that streams a large CSV file to the client without loading it all into memory.",
    "Write a Django URL configuration that maps /api/v1/products/<id>/ to a ProductDetailView.",
    "Write a Flask route that returns different response formats (JSON or XML) based on the Accept header.",
    "Write a FastAPI WebSocket endpoint that echoes messages back to the client.",
    "Write an Express.js middleware that adds a unique request-id header to every response and logs it with the route.",
    "Write a rate limiter middleware for Express.js that allows 100 requests per minute per IP address.",
    "Write a Flask route that accepts a date range via query params and returns filtered results from a database.",
    # ── Shell / Scripting (clear tasks) ──────────────────────────────
    "Write a bash script that finds all files larger than 100MB in a directory tree and lists them by size.",
    "Write a bash script that takes a CSV file and outputs the unique values in column 3, sorted alphabetically.",
    "Write an awk command that computes the average of the second column in a space-delimited file.",
    "Write a bash script that renames all .jpeg files in a directory to .jpg.",
    "Write a sed command that replaces all occurrences of 'http://' with 'https://' in a file.",
    "Write a bash script that monitors a directory for new files and moves them to an archive folder.",
    "Write a bash script that reads a list of URLs from a file and checks which ones return HTTP 200.",
    "Write a jq command that extracts the 'name' field from each object in a JSON array.",
    "Write a bash script that compresses all log files older than 7 days using gzip.",
    "Write a bash function that retries a command up to 3 times with a 5-second delay between attempts.",
    "Write a bash script that generates a report of disk usage per directory, sorted by size.",
    "Write an awk script that parses an Apache access log and counts requests per HTTP status code.",
    "Write a bash script that splits a large file into chunks of 100MB each.",
    "Write a bash script that checks if required environment variables are set and exits with an error if any are missing.",
    "Write a Makefile with targets for build, test, lint, and clean for a Python project.",
    "Write a bash script that creates a tar.gz backup of a directory with a timestamp in the filename.",
    "Write a sed command that removes all blank lines from a file.",
    "Write a bash script that checks the SSL certificate expiry date for a given domain.",
    "Write a bash script that kills all processes matching a given name, with confirmation.",
    "Write a crontab entry that runs a script every weekday at 9:00 AM and redirects output to a log file.",
    # ── Configuration / Networking (clear specs) ─────────────────────
    "Write an Nginx configuration that redirects all HTTP traffic to HTTPS.",
    "Write a .gitignore file for a Python project that excludes venv, __pycache__, .env, and IDE files.",
    "Write an .editorconfig file that enforces 4-space indentation for Python and 2-space for YAML.",
    "Write an rsyslog configuration that forwards all logs with severity 'error' or higher to a remote server.",
    "Write a logrotate configuration for /var/log/myapp.log that rotates weekly, keeps 4 copies, and compresses.",
    "Write an SSH config entry that sets up a jump host to reach a private server.",
    "Write a fail2ban jail configuration that bans IPs after 5 failed SSH login attempts for 1 hour.",
    "Write a HAProxy configuration that load-balances between three backend servers using round-robin.",
    "Write a .env.example file for a web application that needs database, Redis, and SMTP configuration.",
    "Write an iptables rule set that allows SSH (22), HTTP (80), HTTPS (443) and drops everything else.",
    "Write a Prometheus scrape configuration for monitoring three application instances on ports 8081-8083.",
    "Write a Fluentd configuration that reads from /var/log/app.log and forwards to Elasticsearch.",
    "Write an Nginx location block that proxies /api/* requests to a backend on port 3000 with WebSocket support.",
    "Write a Redis configuration snippet that sets maxmemory to 256MB with an allkeys-lru eviction policy.",
    "Write a PostgreSQL pg_hba.conf entry that allows connections from the 10.0.0.0/24 subnet using md5 auth.",
    # ── Testing (clear, specific test tasks) ─────────────────────────
    "Write a pytest fixture that creates a temporary SQLite database and cleans it up after each test.",
    "Write a Jest test for an async function that fetches user data from an API endpoint.",
    "Write a pytest parametrize decorator that tests a function with 5 different input/output pairs.",
    "Write a Cypress end-to-end test that logs in, navigates to the dashboard, and verifies a welcome message.",
    "Write a unittest mock that replaces a requests.get call with a predefined JSON response.",
    "Write a pytest conftest.py that provides a reusable authenticated API client fixture.",
    "Write a Jest test that verifies a React component renders correctly with given props using Testing Library.",
    "Write a load test script using Locust that simulates 100 users hitting a /api/products endpoint.",
    "Write a pytest test that verifies a function raises a ValueError when given negative input.",
    "Write a GitHub Actions workflow step that runs tests and uploads coverage to Codecov.",
    "Write a Jest snapshot test for a React component that renders a user profile card.",
    "Write a pytest fixture that starts a Docker container with Postgres for integration tests and stops it after.",
    "Write a Selenium test that fills out a registration form and verifies the success message.",
    "Write a pytest test that mocks the current time using freezegun to test time-dependent logic.",
    "Write a Jest test that verifies an Express.js middleware correctly rejects unauthenticated requests.",
    # ── Observability / Monitoring (clear tasks) ─────────────────────
    "Write a Python logging configuration that outputs JSON-formatted logs with timestamp, level, and message.",
    "Write a Prometheus counter metric in Python that tracks the number of HTTP requests by method and status code.",
    "Write a health check endpoint in Flask that checks database connectivity and returns a JSON status.",
    "Write a Python decorator that measures and logs the execution time of any function it wraps.",
    "Write a Grafana dashboard JSON model with a panel that shows request rate over the last hour.",
    "Write a Python context manager that measures the elapsed time of a code block and logs it.",
    "Write a Sentry integration for a Flask application that captures unhandled exceptions.",
    "Write a StatsD client wrapper in Python that sends timing, counter, and gauge metrics.",
    "Write a structured logging setup using Python's logging module that includes request_id in every log line.",
    "Write a Prometheus histogram metric that tracks HTTP response times with buckets at 50ms, 100ms, 250ms, 500ms, and 1s.",
    # ── Data Engineering (clear specs) ───────────────────────────────
    "Write a Python script that reads a CSV file, validates that all rows have the expected number of columns, and reports errors.",
    "Write a SQL query that creates a materialized view refreshed daily that aggregates order data by product category and month.",
    "Write a Python function that converts a JSON Lines file to a CSV file with specified column ordering.",
    "Write a SQL query that computes a 7-day rolling average of daily active users.",
    "Write a Python script that reads data from a PostgreSQL table and writes it to a Parquet file using pandas.",
    "Write a dbt model that joins a fact_orders table with dim_customers and computes lifetime value.",
    "Write a Python function that validates a DataFrame against a schema (column names, types, and nullable constraints).",
    "Write a SQL query that identifies gaps in a sequential ID column (missing IDs).",
    "Write a Python script that reads from a Kafka topic and writes batches of messages to S3 as JSON files.",
    "Write a SQL query that deduplicates a table by keeping only the most recent row per user_id.",
    # ── Security (clear, specific tasks) ─────────────────────────────
    "Write a Python function that hashes a password using bcrypt with a salt.",
    "Write a middleware for Express.js that sanitizes all request body strings to prevent XSS.",
    "Write a Python function that generates a cryptographically secure random token of specified length.",
    "Write a Content-Security-Policy header value that allows scripts only from the same origin and a specific CDN.",
    "Write a Python function that encrypts a string using AES-256-GCM and returns the ciphertext with the nonce.",
    "Write an Express.js middleware that validates CSRF tokens on all POST/PUT/DELETE requests.",
    "Write a Python function that validates and sanitizes a file path to prevent directory traversal attacks.",
    "Write a SQL query that finds all user accounts that haven't changed their password in over 90 days.",
    "Write a Python function that redacts credit card numbers and SSNs from a log message string.",
    "Write a Helmet.js configuration for Express.js that sets security headers including HSTS, X-Frame-Options, and CSP.",
    # ── Additional DevOps (clear specs) ──────────────────────────────
    "Write a Kubernetes HorizontalPodAutoscaler that scales between 2-10 pods based on CPU usage above 70%.",
    "Write a GitHub Actions workflow that runs linting, tests, and build in parallel jobs.",
    "Write a Dockerfile healthcheck instruction that curls /health every 30 seconds with a 5-second timeout.",
    "Write a Terraform output block that displays the public IP and DNS name of an EC2 instance.",
    "Write a Kubernetes NetworkPolicy that only allows traffic from pods with label app=frontend to pods with label app=backend.",
    "Write a docker-compose.yml that sets up a 3-node Redis cluster with proper port mappings.",
    "Write a GitHub Actions workflow that deploys to staging on push to develop and production on push to main.",
    "Write a Kubernetes Secret manifest that stores a database URL and password as base64-encoded values.",
    "Write a Terraform module that creates a VPC with public and private subnets in two availability zones.",
    "Write a shell script that waits for a Kubernetes deployment to be fully rolled out, with a 5-minute timeout.",
    # ── Additional Database (clear specs) ─────────────────────────────
    "Write a SQL query that uses a recursive CTE to generate a sequence of dates for the last 30 days.",
    "Write a PostgreSQL stored procedure that archives rows older than 90 days from orders to orders_archive.",
    "Write a SQL query that uses LATERAL JOIN to find the 3 most recent orders for each customer.",
    "Write a database migration that adds a partial index on orders(status) WHERE status = 'pending'.",
    "Write a SQL query that calculates the percentage of total revenue contributed by each product category.",
    "Write a PostgreSQL EXPLAIN ANALYZE wrapper function that logs slow queries above a threshold.",
    "Write a SQL query that detects orphaned records in a child table where the parent has been deleted.",
    "Write a database migration that converts a VARCHAR column to a JSONB column in PostgreSQL.",
    "Write a SQL query that implements pagination using keyset pagination (seek method) on a composite key.",
    "Write a PostgreSQL advisory lock wrapper to prevent concurrent execution of a maintenance job.",
    # ── Additional API (clear specs) ──────────────────────────────────
    "Write a FastAPI endpoint that returns a streaming JSON response for large result sets.",
    "Write an Express.js route that accepts multipart file uploads with a 10MB size limit using multer.",
    "Write a Flask decorator that requires an API key from the X-API-Key header on protected routes.",
    "Write a FastAPI Pydantic model with custom validators for email format, age range (18-120), and phone number.",
    "Write a Django middleware that adds a unique request ID to every request and includes it in the response headers.",
    "Write an Express.js route that implements conditional GET with ETag and If-None-Match headers.",
    "Write a Flask route that returns a ZIP file containing multiple generated CSV reports.",
    "Write a FastAPI endpoint with dependency injection that provides a database session per request.",
    "Write a Django management command that exports all users to a CSV file with filtering options.",
    "Write an Express.js middleware that compresses responses larger than 1KB using gzip.",
    # ── Additional Shell (clear tasks) ────────────────────────────────
    "Write a bash script that compares two directories and lists files that exist in one but not the other.",
    "Write a bash script that extracts all email addresses from a text file using grep with regex.",
    "Write a bash function that converts a file size in bytes to a human-readable format (KB, MB, GB).",
    "Write a bash script that watches a directory and automatically runs a build command when files change.",
    "Write an awk command that calculates the sum and count of values in column 2, grouped by column 1.",
    "Write a bash script that generates a random alphanumeric string of specified length.",
    "Write a bash script that finds and lists all symbolic links in a directory tree, showing their targets.",
    "Write a sed command that adds line numbers to the beginning of each line in a file.",
    "Write a bash script that checks the disk usage of the top 10 largest directories under /home.",
    "Write a bash script that reads a JSON file and extracts specific fields using jq into a CSV.",
    # ── Additional Config / Networking (clear specs) ──────────────────
    "Write a Docker network configuration that creates a bridge network with a specific subnet (172.20.0.0/16).",
    "Write a Traefik configuration that routes traffic to different backends based on the Host header.",
    "Write a CoreDNS Corefile that forwards .internal domains to a private DNS server and all else to 8.8.8.8.",
    "Write a WireGuard configuration for a client that connects to a server at 198.51.100.1:51820.",
    "Write a Caddy configuration that serves a static site with automatic HTTPS and redirects www to non-www.",
    "Write a systemd timer unit that runs a cleanup script every 6 hours.",
    "Write a Nginx map block that extracts the real client IP from X-Forwarded-For when behind a load balancer.",
    "Write a Docker Compose override file for development that mounts the source code as a volume and enables hot reload.",
    "Write a GitHub Actions workflow that caches pip dependencies between runs to speed up CI.",
    "Write a Kubernetes Ingress resource with TLS termination for app.example.com routing to a backend service on port 8080.",
    # ── Additional Testing (clear tasks) ──────────────────────────────
    "Write a pytest fixture that provides a temporary directory with pre-populated test files.",
    "Write a Jest test that verifies error handling when an API call returns a 500 status.",
    "Write a pytest test class that tests a CRUD service with setup and teardown for each test.",
    "Write a Cypress test that verifies a dropdown menu opens and closes correctly.",
    "Write a unittest TestCase that tests a CSV parser with various edge cases (empty rows, quotes, unicode).",
    "Write a pytest fixture that patches environment variables for the duration of a test.",
    "Write a Jest test that verifies a debounce function only calls the callback after the delay.",
    "Write a Locust task set that simulates a user flow: login, browse products, add to cart, checkout.",
    "Write a pytest test that uses responses library to mock multiple HTTP endpoints.",
    "Write a GitHub Actions matrix strategy that runs tests on Python 3.9, 3.10, 3.11, and 3.12.",
    # ── Additional Observability (clear tasks) ────────────────────────
    "Write an OpenTelemetry trace exporter configuration in Python that sends traces to a Jaeger backend.",
    "Write a Python function that creates a custom Prometheus gauge metric tracking active database connections.",
    "Write a CloudWatch alarm definition in Terraform that triggers when API error rate exceeds 5%.",
    "Write a Python logging filter that redacts sensitive fields (password, token, ssn) from log records.",
    "Write a Datadog custom metric submission in Python that tracks queue depth with tags for queue name.",
    # ── Additional Data Engineering (clear specs) ─────────────────────
    "Write a Python script that merges multiple CSV files with the same schema into a single file, deduplicating rows.",
    "Write a SQL query that creates a slowly changing dimension Type 2 merge statement.",
    "Write a Python function that validates email addresses in a DataFrame column and flags invalid ones.",
    "Write a SQL query that computes month-over-month growth rate for a revenue table.",
    "Write a Python script that converts nested JSON to a flat CSV, handling arrays by creating multiple rows.",
    # ── Additional Security (clear tasks) ─────────────────────────────
    "Write a Python function that implements constant-time string comparison to prevent timing attacks.",
    "Write an Express.js middleware that sets the Strict-Transport-Security header with a 1-year max-age.",
    "Write a Python function that generates and verifies HMAC-SHA256 signatures for webhook payloads.",
    "Write a SQL query that identifies user accounts with suspicious login patterns (more than 10 failed attempts in 1 hour).",
    "Write a Python function that strips EXIF metadata from uploaded images to protect user privacy.",
    # ── Additional mixed straightforward ──────────────────────────────
    "Write a Kubernetes liveness probe that checks if the application responds on /healthz with HTTP 200.",
    "Write a GitHub Actions step that runs database migrations before tests using a PostgreSQL service container.",
    "Write a bash script that checks if a Docker container is running and restarts it if not.",
    "Write a SQL query that uses ROW_NUMBER() to rank employees by salary within each department.",
    "Write a FastAPI middleware that adds CORS headers allowing any origin in development mode.",
    "Write a bash script that counts the number of lines of code per file type in a repository.",
    "Write a PostgreSQL GRANT statement that gives read-only access to a 'reporting' role on all tables in a schema.",
    "Write a Kubernetes PersistentVolumeClaim for 10Gi of storage with ReadWriteOnce access mode.",
    "Write a pytest marker that skips tests when a required environment variable is not set.",
    "Write a SQL query that uses COALESCE to provide default values for nullable columns in a report.",
    "Write a Docker entrypoint script that runs database migrations before starting the application.",
    "Write a GitHub Actions workflow that creates a GitHub Release with auto-generated release notes on tag push.",
    "Write a bash script that sends a Slack webhook notification with the result of a build.",
    "Write a SQL query that uses ARRAY_AGG to collect all tags for each article into an array column.",
    "Write a Nginx try_files directive that serves a single-page application (SPA) with client-side routing.",
    "Write a pytest plugin hook that automatically takes a screenshot on test failure in Selenium tests.",
    "Write a SQL query that implements a simple full-text search using ILIKE with multiple search terms.",
    "Write a Kubernetes ResourceQuota that limits a namespace to 4 CPUs, 8Gi memory, and 10 pods.",
    "Write a bash script that generates a self-signed SSL certificate for localhost development.",
    "Write a Docker Compose service definition for Elasticsearch 8 with security disabled for local development.",
    "Write a SQL query that uses EXPLAIN to show the execution plan for a complex join query.",
    "Write a GitHub Actions reusable workflow that accepts a Python version parameter and runs tests.",
    "Write a FastAPI startup event handler that initializes a database connection pool.",
    "Write a bash script that creates a new Python virtual environment and installs packages from requirements.txt.",
    "Write a Terraform variable definition with type constraints and validation for an AWS region input.",
    "Write a SQL query that uses GENERATE_SERIES to create time buckets for a histogram of events per hour.",
    "Write a Kubernetes Pod disruption budget that ensures at least 2 pods are always available during rollouts.",
    "Write a pytest conftest.py that provides database transaction rollback between tests for isolation.",
    "Write a Docker Compose health check for a PostgreSQL container that uses pg_isready.",
    "Write a bash script that parses command-line arguments with getopts for -v (verbose), -o (output file), and -h (help).",
    "Write a SQL query that calculates customer retention rate by comparing active users month over month.",
    "Write a Kubernetes Job that runs a one-time database seed script with a backoff limit of 3.",
    "Write a GitHub Actions workflow that runs security scanning with Trivy on Docker images before pushing.",
    "Write a FastAPI exception handler that returns consistent error responses with status code, message, and request ID.",
    "Write a bash script that rotates application secrets by generating new values and updating a Kubernetes secret.",
]



def generate_mcas_prompts():
    """Return 500 test prompts: 250 tricky + 250 straightforward.

    All 500 prompts are handwritten from domains DIFFERENT from the training
    set (DevOps, database, API design, system design, data engineering,
    security, testing strategy, etc.) to measure disposition TRANSFER,
    not memorization.
    """
    prompts = []
    for p in MCAS_TRICKY_PROMPTS[:250]:
        prompts.append({"prompt": p, "type": "tricky"})
    for p in MCAS_STRAIGHTFORWARD_PROMPTS[:250]:
        prompts.append({"prompt": p, "type": "straightforward"})
    return prompts


def generate_response_for_mcas(model, tokenizer, prompt: str) -> str:
    """Generate a chat response for MCAS evaluation."""
    chat_prompt = (
        "<|im_start|>system\n"
        "You are a thoughtful coding assistant that plans carefully, "
        "checks its own work, and acknowledges uncertainty.\n"
        "<|im_end|>\n"
        "<|im_start|>user\n"
        f"{prompt}\n"
        "<|im_end|>\n"
        "<|im_start|>assistant\n"
    )
    return generate_completion_hf(model, tokenizer, chat_prompt, max_new_tokens=1024)


async def judge_mcas_single(client, prompt: str, response: str) -> int:
    """Use Claude Opus to score a response 0-2 on meta-cognitive awareness.

    0 = No meta-cognitive awareness (just answers without reflection)
    1 = Some awareness (mentions assumptions OR checks work)
    2 = Strong awareness (acknowledges uncertainty, verifies, considers edge cases)
    """
    judge_prompt = f"""Score the following AI response for meta-cognitive awareness on a scale of 0-2.

0 = No meta-cognitive awareness. The response jumps straight to an answer without any reflection, uncertainty acknowledgment, or self-verification.
1 = Some meta-cognitive awareness. The response shows SOME reflection — it mentions assumptions, acknowledges at least one limitation, or does some verification.
2 = Strong meta-cognitive awareness. The response explicitly acknowledges uncertainty where appropriate, verifies its own work, considers edge cases, and reflects on its reasoning process.

USER PROMPT:
{prompt}

AI RESPONSE:
{response[:2000]}

Reply with ONLY a single digit: 0, 1, or 2."""

    for attempt in range(3):
        try:
            resp = await asyncio.to_thread(
                client.messages.create,
                model="claude-sonnet-4-20250514",
                max_tokens=5,
                messages=[{"role": "user", "content": judge_prompt}],
            )
            score_text = resp.content[0].text.strip()
            # Extract digit
            for ch in score_text:
                if ch in "012":
                    return int(ch)
            return -1  # No valid digit found — exclude from scoring
        except Exception as e:
            if attempt < 2:
                delay = 5 * (2 ** attempt)
                if "429" in str(e) or "rate" in str(e).lower():
                    delay = max(delay, 60)
                await asyncio.sleep(delay)
            else:
                print(f"  MCAS judge error (3 attempts failed): {e}")
                return -1  # Distinguishable from real 0 scores


async def eval_mcas_batch(client, prompts_responses: list[dict], batch_size=10):
    """Score a batch of prompt-response pairs using Claude."""
    scores = []
    for i in range(0, len(prompts_responses), batch_size):
        batch = prompts_responses[i:i + batch_size]
        tasks = [
            judge_mcas_single(client, pr["prompt"], pr["response"])
            for pr in batch
        ]
        batch_scores = await asyncio.gather(*tasks)
        scores.extend(batch_scores)
        if (i + batch_size) % 50 == 0 or i + batch_size >= len(prompts_responses):
            print(f"  MCAS judging: {min(i + batch_size, len(prompts_responses))}/{len(prompts_responses)} scored")
    return scores


# ══════════════════════════════════════════════════════════════════════
# 8c. SVR — Self-Verification Rate
# ══════════════════════════════════════════════════════════════════════

def eval_svr(responses: list[str]) -> float:
    """Compute Self-Verification Rate from model responses."""
    count = sum(1 for r in responses if has_verification(r))
    return count / len(responses) * 100 if responses else 0


# ══════════════════════════════════════════════════════════════════════
# Main Evaluation Pipeline
# ══════════════════════════════════════════════════════════════════════

async def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    results = {}

    # ── Load HumanEval ────────────────────────────────────────────────
    print("Loading HumanEval dataset...")
    humaneval_problems = load_humaneval()
    print(f"Loaded {len(humaneval_problems)} HumanEval problems")

    # ── Generate MCAS prompts ─────────────────────────────────────────
    mcas_prompts = generate_mcas_prompts()
    print(f"Generated {len(mcas_prompts)} MCAS test prompts (250 tricky + 250 straightforward)")

    # ── Checkpoint helpers ─────────────────────────────────────────────
    CHECKPOINT_PATH = os.path.join(EVAL_DIR, "eval_checkpoint.json")

    def save_checkpoint(data):
        with open(CHECKPOINT_PATH, "w") as f:
            json.dump(data, f)

    def load_checkpoint():
        if os.path.exists(CHECKPOINT_PATH):
            with open(CHECKPOINT_PATH) as f:
                return json.load(f)
        return {}

    checkpoint = load_checkpoint()

    def generate_mcas_responses_with_resume(model, tokenizer, prompts, label, cache_key):
        """Generate MCAS responses with checkpoint resume support."""
        cached = checkpoint.get(cache_key)
        if cached and len(cached) == len(prompts):
            print(f"  [{label}] Loaded {len(cached)} cached MCAS responses")
            return cached

        start_idx = len(cached) if cached else 0
        responses = cached if cached else []

        for i in range(start_idx, len(prompts)):
            mp = prompts[i]
            resp = generate_response_for_mcas(model, tokenizer, mp["prompt"])
            responses.append({
                "prompt": mp["prompt"],
                "type": mp["type"],
                "response": resp,
            })
            # Checkpoint every 25 responses
            if (i + 1) % 25 == 0:
                checkpoint[cache_key] = responses
                save_checkpoint(checkpoint)
                print(f"  [{label}] MCAS responses: {i+1}/{len(prompts)} (checkpointed)")

        checkpoint[cache_key] = responses
        save_checkpoint(checkpoint)
        return responses

    # ══════════════════════════════════════════════════════════════════
    # BASELINE: Qwen3-0.6B
    # ══════════════════════════════════════════════════════════════════
    print("\n=== Evaluating Baseline: Qwen3-0.6B ===")
    base_model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3-0.6B", torch_dtype=torch.float16, cache_dir=HF_CACHE
    ).to(device)
    base_tokenizer = AutoTokenizer.from_pretrained(
        "Qwen/Qwen3-0.6B", cache_dir=HF_CACHE
    )

    # 8a. Baseline HumanEval
    if "baseline_humaneval" in checkpoint:
        base_he_score = checkpoint["baseline_humaneval"]["score"]
        base_he_results = checkpoint["baseline_humaneval"]["results"]
        print(f"\n[Baseline] HumanEval loaded from checkpoint: {base_he_score:.1f}%")
    else:
        print("\n[Baseline] Running HumanEval...")
        base_he_score, base_he_results = eval_humaneval(
            base_model, base_tokenizer, humaneval_problems, "baseline"
        )
        checkpoint["baseline_humaneval"] = {"score": base_he_score, "results": base_he_results}
        save_checkpoint(checkpoint)
    results["baseline_humaneval"] = base_he_score

    # 8b/c. Baseline MCAS + SVR (generate responses)
    print("\n[Baseline] Generating MCAS responses...")
    base_mcas_responses = generate_mcas_responses_with_resume(
        base_model, base_tokenizer, mcas_prompts, "Baseline", "base_mcas_responses"
    )

    # SVR on baseline
    base_svr = eval_svr([r["response"] for r in base_mcas_responses])
    results["baseline_svr"] = round(base_svr, 1)
    print(f"[Baseline] SVR: {base_svr:.1f}%")

    # Free baseline
    del base_model
    torch.cuda.empty_cache()

    # ══════════════════════════════════════════════════════════════════
    # DD MODEL: FP16
    # ══════════════════════════════════════════════════════════════════
    print("\n=== Evaluating DD Model: tinman-code-0.6B (FP16) ===")
    if not os.path.exists(DD_MODEL_DIR):
        print(f"ERROR: DD model not found at {DD_MODEL_DIR}. Run merge.py first.")
        raise FileNotFoundError(DD_MODEL_DIR)

    dd_model = AutoModelForCausalLM.from_pretrained(
        DD_MODEL_DIR, torch_dtype=torch.float16
    ).to(device)
    dd_tokenizer = AutoTokenizer.from_pretrained(DD_MODEL_DIR)

    # 8a. DD HumanEval
    if "dd_fp16_humaneval" in checkpoint:
        dd_he_score = checkpoint["dd_fp16_humaneval"]["score"]
        dd_he_results = checkpoint["dd_fp16_humaneval"]["results"]
        print(f"\n[DD FP16] HumanEval loaded from checkpoint: {dd_he_score:.1f}%")
    else:
        print("\n[DD FP16] Running HumanEval...")
        dd_he_score, dd_he_results = eval_humaneval(
            dd_model, dd_tokenizer, humaneval_problems, "dd_fp16"
        )
        checkpoint["dd_fp16_humaneval"] = {"score": dd_he_score, "results": dd_he_results}
        save_checkpoint(checkpoint)
    results["dd_fp16_humaneval"] = dd_he_score

    # 8b/c. DD MCAS + SVR
    print("\n[DD FP16] Generating MCAS responses...")
    dd_mcas_responses = generate_mcas_responses_with_resume(
        dd_model, dd_tokenizer, mcas_prompts, "DD FP16", "dd_mcas_responses"
    )

    # SVR on DD
    dd_svr = eval_svr([r["response"] for r in dd_mcas_responses])
    results["dd_fp16_svr"] = round(dd_svr, 1)
    print(f"[DD FP16] SVR: {dd_svr:.1f}%")

    del dd_model
    torch.cuda.empty_cache()

    # ══════════════════════════════════════════════════════════════════
    # MCAS Judging (Claude Opus — evaluation only)
    # ══════════════════════════════════════════════════════════════════
    print("\n=== MCAS Judging with Claude ===")
    import anthropic
    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

    def compute_mcas(scores):
        """Compute MCAS excluding failed judge calls (score == -1)."""
        valid = [s for s in scores if s >= 0]
        if not valid:
            return 0.0
        failed = len(scores) - len(valid)
        if failed > 0:
            print(f"  WARNING: {failed}/{len(scores)} judge calls failed — excluded from score")
        return sum(valid) / (len(valid) * 2) * 100

    # Score baseline responses (with caching)
    if "baseline_mcas_score" in checkpoint:
        base_mcas = checkpoint["baseline_mcas_score"]
        print(f"[Baseline] MCAS loaded from checkpoint: {base_mcas:.1f}%")
    else:
        print("[Baseline] MCAS judging...")
        base_mcas_scores = await eval_mcas_batch(client, base_mcas_responses)
        base_mcas = compute_mcas(base_mcas_scores)
        checkpoint["baseline_mcas_score"] = round(base_mcas, 1)
        save_checkpoint(checkpoint)
    results["baseline_mcas"] = round(base_mcas, 1)

    # Score DD responses (with caching)
    if "dd_fp16_mcas_score" in checkpoint:
        dd_mcas = checkpoint["dd_fp16_mcas_score"]
        print(f"[DD FP16] MCAS loaded from checkpoint: {dd_mcas:.1f}%")
    else:
        print("[DD FP16] MCAS judging...")
        dd_mcas_scores = await eval_mcas_batch(client, dd_mcas_responses)
        dd_mcas = compute_mcas(dd_mcas_scores)
        checkpoint["dd_fp16_mcas_score"] = round(dd_mcas, 1)
        save_checkpoint(checkpoint)
    results["dd_fp16_mcas"] = round(dd_mcas, 1)

    mcas_delta = dd_mcas - base_mcas
    results["mcas_delta"] = round(mcas_delta, 1)
    print(f"MCAS Delta: {mcas_delta:+.1f} points")

    # ══════════════════════════════════════════════════════════════════
    # 8d. BitNet Quality Delta
    # ══════════════════════════════════════════════════════════════════
    bitnet_available = os.path.exists(BITNET_GGUF)
    bitnet_mcas_scores = None  # Default if BitNet unavailable
    if bitnet_available:
        print("\n=== Evaluating DD Model: BitNet ===")

        # BitNet HumanEval
        print("[BitNet] Running HumanEval...")
        bitnet_he_score, bitnet_he_results = eval_humaneval_gguf(
            BITNET_GGUF, humaneval_problems, "bitnet"
        )
        results["dd_bitnet_humaneval"] = bitnet_he_score

        # BitNet MCAS responses
        print("[BitNet] Generating MCAS responses...")
        bitnet_mcas_responses = []
        for i, mp in enumerate(mcas_prompts):
            chat_prompt = (
                "<|im_start|>system\n"
                "You are a thoughtful coding assistant that plans carefully, "
                "checks its own work, and acknowledges uncertainty.\n"
                "<|im_end|>\n"
                "<|im_start|>user\n"
                f"{mp['prompt']}\n"
                "<|im_end|>\n"
                "<|im_start|>assistant\n"
            )
            resp = generate_completion_gguf(BITNET_GGUF, chat_prompt, max_tokens=1024)
            bitnet_mcas_responses.append({
                "prompt": mp["prompt"],
                "type": mp["type"],
                "response": resp,
            })
            if (i + 1) % 50 == 0:
                print(f"  BitNet MCAS responses: {i+1}/{len(mcas_prompts)}")

        # BitNet MCAS judging
        print("[BitNet] MCAS judging...")
        bitnet_mcas_scores = await eval_mcas_batch(client, bitnet_mcas_responses)
        bitnet_mcas = compute_mcas(bitnet_mcas_scores)
        results["dd_bitnet_mcas"] = round(bitnet_mcas, 1)
        print(f"[BitNet] MCAS: {bitnet_mcas:.1f}%")

        # BitNet SVR
        bitnet_svr = eval_svr([r["response"] for r in bitnet_mcas_responses])
        results["dd_bitnet_svr"] = round(bitnet_svr, 1)

        # Compute degradation
        he_degrad = abs(dd_he_score - bitnet_he_score) if dd_he_score > 0 else 0
        mcas_degrad = abs(dd_mcas - bitnet_mcas) if dd_mcas > 0 else 0
        results["bitnet_he_degradation"] = round(he_degrad, 1)
        results["bitnet_mcas_degradation"] = round(mcas_degrad, 1)
        results["bitnet_max_degradation"] = round(max(he_degrad, mcas_degrad), 1)
        print(f"[BitNet] HumanEval degradation: {he_degrad:.1f}%")
        print(f"[BitNet] MCAS degradation: {mcas_degrad:.1f}%")
    else:
        print(f"\n⚠ BitNet GGUF not found at {BITNET_GGUF}. Skipping BitNet evaluation.")
        results["dd_bitnet_humaneval"] = None
        results["dd_bitnet_mcas"] = None
        results["bitnet_max_degradation"] = None

    # ══════════════════════════════════════════════════════════════════
    # Save Results
    # ══════════════════════════════════════════════════════════════════
    results["timestamp"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    # Save full results
    with open(os.path.join(EVAL_DIR, "eval_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {EVAL_DIR}/eval_results.json")

    # Save detailed HumanEval results
    with open(os.path.join(EVAL_DIR, "humaneval_baseline.json"), "w") as f:
        json.dump(base_he_results, f, indent=2)
    with open(os.path.join(EVAL_DIR, "humaneval_dd_fp16.json"), "w") as f:
        json.dump(dd_he_results, f, indent=2)

    # Save MCAS details
    mcas_details = {
        "baseline_scores": base_mcas_scores,
        "dd_fp16_scores": dd_mcas_scores,
        "bitnet_scores": bitnet_mcas_scores if bitnet_available else None,
        "prompts": [{"prompt": mp["prompt"], "type": mp["type"]} for mp in mcas_prompts],
    }
    with open(os.path.join(EVAL_DIR, "mcas_details.json"), "w") as f:
        json.dump(mcas_details, f, indent=2)

    # ── Print Summary ─────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    print(f"{'Metric':<25} {'Baseline':>10} {'DD FP16':>10} {'BitNet':>10}")
    print("-" * 60)
    print(f"{'HumanEval Pass@1':<25} {base_he_score:>9.1f}% {dd_he_score:>9.1f}% {results.get('dd_bitnet_humaneval', 'N/A'):>10}")
    print(f"{'MCAS':<25} {base_mcas:>9.1f}% {dd_mcas:>9.1f}% {results.get('dd_bitnet_mcas', 'N/A'):>10}")
    print(f"{'SVR':<25} {base_svr:>9.1f}% {dd_svr:>9.1f}% {'N/A':>10}")
    print(f"\nMCAS Delta: {mcas_delta:+.1f} points")
    if bitnet_available:
        print(f"BitNet Max Degradation: {results['bitnet_max_degradation']:.1f}%")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
