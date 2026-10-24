const $ = (id) => document.getElementById(id);
const state = { result: null, submissionId: crypto.randomUUID(), lifecycleRequestId: crypto.randomUUID(), activeExchangeId: null, libraryPage: 1, libraryTotalPages: 0, libraryLoaded: false, sourcePage: 1, sourceTotalPages: 0, sourcesLoaded: false };
const sourceCache = new Map();

function workspaceId() { return $("workspace-select")?.value || "default"; }

function requestedWorkspaceId() {
  return new URLSearchParams(window.location.search).get("workspace_id")
    || window.localStorage.getItem("mcr.workspace_id")
    || "default";
}

function rememberWorkspace() {
  const value = workspaceId();
  window.localStorage.setItem("mcr.workspace_id", value);
  const url = new URL(window.location.href);
  url.searchParams.set("workspace_id", value);
  window.history.replaceState({}, "", url);
}

function showToast(message) {
  const toast = $("toast");
  toast.textContent = message;
  toast.classList.add("show");
  setTimeout(() => toast.classList.remove("show"), 4200);
}

function filesToParts(input) {
  return Promise.all([...input.files].map((file) => new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve({ type: "image", data_url: reader.result, media_type: file.type, detail: "auto" });
    reader.onerror = reject;
    reader.readAsDataURL(file);
  })));
}

function filesToImageSources(input) {
  return Promise.all([...input.files].map((file) => new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve({ data_url: reader.result });
    reader.onerror = reject;
    reader.readAsDataURL(file);
  })));
}

async function currentContent(prefix) {
  const text = $(`${prefix}-text`).value.trim();
  const images = await filesToParts($(`${prefix}-images`));
  const content = [];
  if (text) content.push({ type: "text", text });
  content.push(...images);
  return content.length ? content : null;
}

async function buildSessionPayload() {
  const sessionId = $("session-id").value.trim();
  if (!sessionId) throw new Error("请输入 Session ID");
  const userContent = await currentContent("user");
  if (!userContent) throw new Error("User 消息不能为空");
  const assistantContent = await currentContent("assistant");
  return {
    workspace_id: workspaceId(),
    session_id: sessionId,
    user_content: userContent,
    assistant_content: assistantContent,
    submission_id: state.submissionId,
  };
}

function pretty(value) { return JSON.stringify(value, null, 2); }

function responseError(body) {
  const detail = body.detail;
  const error = new Error(typeof detail === "string" ? detail : (detail?.error || pretty(detail)));
  error.detail = detail;
  return error;
}

async function readResponseBody(response) {
  const text = await response.text();
  if (!text) return { detail: `HTTP ${response.status} ${response.statusText}` };
  try { return JSON.parse(text); }
  catch { return { detail: text }; }
}

function renderResponseError(error) {
  const detail = error.detail;
  if (detail && typeof detail === "object") {
    $("tab-raw").textContent = detail.raw_response || "";
    $("run-meta").textContent = pretty({ error: detail.error, issues: detail.issues || [] });
    activateTab("raw");
  }
  showToast(error.message);
}

function renderRawAttempts(trace) {
  const attempts = trace?.response_attempts || [];
  if (attempts.length <= 1) return trace?.raw_response || "";
  return attempts.map((value, index) => `--- attempt ${index + 1} ---\n${value}`).join("\n\n");
}

function activateTab(name) {
  document.querySelectorAll(".tab").forEach((tab) => tab.classList.toggle("active", tab.dataset.tab === name));
  document.querySelectorAll(".tab-content").forEach((panel) => panel.classList.toggle("active", panel.id === `tab-${name}`));
}

function renderCandidates(candidates) {
  $("empty-state").style.display = candidates.length ? "none" : "grid";
  $("candidate-list").innerHTML = "";
  if (!candidates.length && state.result) {
    $("empty-state").querySelector("h3").textContent = "本轮没有候选记忆";
    $("empty-state").querySelector("p").textContent = "模型返回了有效的空 memories 列表。";
  }
  candidates.forEach((candidate, index) => {
    const node = document.createElement("article");
    node.className = "candidate";
    const head = document.createElement("div");
    head.className = "candidate-head";
    const kind = document.createElement("span");
    kind.className = "kind";
    kind.textContent = `${String(index + 1).padStart(2, "0")} · ${candidate.x.kind}`;
    const source = document.createElement("span");
    source.className = "source";
    source.textContent = candidate.meta.source_message_ids.join(" + ");
    head.append(kind, source);
    const content = document.createElement("p");
    content.textContent = candidate.x.content;
    const details = document.createElement("details");
    const summary = document.createElement("summary");
    summary.textContent = "查看完整 x + meta";
    const pre = document.createElement("pre");
    pre.textContent = pretty(candidate);
    details.append(summary, pre);
    node.append(head, content, details);
    $("candidate-list").append(node);
  });
}

function sourceMessageCard(message, includeMemories = true) {
  const node = document.createElement("article");
  node.className = `source-message-card ${message.role}`;
  const header = document.createElement("header");
  const title = document.createElement("strong");
  title.textContent = `${message.role.toUpperCase()} · turn ${message.turn_id}`;
  const meta = document.createElement("span");
  meta.textContent = `${message.session_id} · ${message.id}`;
  header.append(title, meta);
  node.append(header);
  (message.parts || []).forEach((part) => {
    const partMeta = document.createElement("div");
    partMeta.className = "source-part-meta";
    partMeta.textContent = `${part.part_type} · part ${part.part_index}`;
    node.append(partMeta);
    if (part.part_type === "text") {
      const text = document.createElement("pre");
      text.className = "source-text";
      text.textContent = part.text_content || "";
      node.append(text);
    } else {
      const image = document.createElement("img");
      image.className = "source-image";
      image.alt = `原始图片 part ${part.part_index}`;
      image.src = `/api/source-messages/${encodeURIComponent(message.id)}/parts/${part.part_index}?workspace_id=${encodeURIComponent(workspaceId())}`;
      node.append(image);
    }
  });
  if (includeMemories) {
    const linkedTitle = document.createElement("div");
    linkedTitle.className = "linked-memory-title";
    linkedTitle.textContent = `关联记忆 · ${(message.memories || []).length}`;
    const linked = document.createElement("div");
    linked.className = "linked-memory-list";
    (message.memories || []).forEach((memory) => linked.append(memoryCard(memory, null, false)));
    if (!message.memories?.length) {
      linked.textContent = "这条原始消息没有生成持久化记忆。";
      linked.classList.add("source-empty");
    }
    node.append(linkedTitle, linked);
  }
  return node;
}

async function fetchSourceMessage(messageId) {
  const key = `${workspaceId()}:${messageId}`;
  if (!sourceCache.has(key)) {
    sourceCache.set(key, fetch(`/api/source-messages/${encodeURIComponent(messageId)}?workspace_id=${encodeURIComponent(workspaceId())}`)
      .then(async (response) => {
        const body = await readResponseBody(response);
        if (!response.ok) throw responseError(body);
        return body;
      }));
  }
  return sourceCache.get(key);
}

async function attachMemorySources(container, memory) {
  const ids = [...new Set((memory.relations || []).map((relation) => relation.source_id))];
  if (!ids.length) {
    container.textContent = "该记忆没有原始消息关系。";
    return;
  }
  try {
    const messages = await Promise.all(ids.map(fetchSourceMessage));
    container.textContent = "";
    messages.forEach((message) => container.append(sourceMessageCard(message, false)));
  } catch (error) {
    container.textContent = `原始消息加载失败：${error.message}`;
  }
}

function memoryCard(memory, retrieval = null, includeSources = true) {
  const node = document.createElement("article");
  node.className = "memory-card";
  const header = document.createElement("header");
  const titleWrap = document.createElement("div");
  const title = document.createElement("h3");
  title.textContent = `${memory.x.kind.toUpperCase()} · ${memory.meta.session_id}`;
  const id = document.createElement("div");
  id.className = "memory-id";
  id.textContent = memory.id;
  titleWrap.append(title, id);
  header.append(titleWrap);
  if (retrieval) {
    const score = document.createElement("span");
    score.className = "score";
    score.textContent = `#${retrieval.rank} · ${retrieval.score.toFixed(4)}`;
    header.append(score);
  }
  const content = document.createElement("p");
  content.className = "memory-content";
  content.textContent = memory.x.content;
  const chips = document.createElement("div");
  chips.className = "chips";
  const descriptionCounts = (memory.descriptions || []).reduce((counts, item) => {
    counts[item.facet_type] = (counts[item.facet_type] || 0) + 1;
    return counts;
  }, {});
  Object.entries(descriptionCounts).forEach(([facetType, count]) => {
    const chip = document.createElement("span");
    chip.className = "chip";
    chip.textContent = `${facetType} · ${count}`;
    chips.append(chip);
  });
  if (memory.links.length) {
    const chip = document.createElement("span");
    chip.className = "chip";
    chip.textContent = `links · ${memory.links.length}`;
    chips.append(chip);
  }
  node.append(header, content, chips);
  if (includeSources) {
    const sourceDetails = document.createElement("details");
    sourceDetails.className = "memory-sources";
    const sourceSummary = document.createElement("summary");
    sourceSummary.textContent = `原始消息 · ${new Set((memory.relations || []).map((relation) => relation.source_id)).size}`;
    const sourceList = document.createElement("div");
    sourceList.className = "memory-source-list";
    sourceList.textContent = "正在加载原始消息…";
    sourceDetails.append(sourceSummary, sourceList);
    node.append(sourceDetails);
    attachMemorySources(sourceList, memory);
  }
  if (retrieval) {
    const routeDetails = document.createElement("details");
    const routeSummary = document.createElement("summary");
    routeSummary.textContent = `匹配路径 · ${retrieval.evidence.length}`;
    const routePre = document.createElement("pre");
    routePre.textContent = pretty({
      item_type: retrieval.item_type,
      modality_subtotals: retrieval.modality_subtotals,
      evidence: retrieval.evidence,
    });
    routeDetails.append(routeSummary, routePre);
    node.append(routeDetails);
  }
  const details = document.createElement("details");
  const summary = document.createElement("summary");
  summary.textContent = "完整 x / meta / D / r / links";
  const pre = document.createElement("pre");
  pre.textContent = pretty(memory);
  details.append(summary, pre);
  node.append(details);
  return node;
}

function activateWorkspace(name) {
  document.querySelectorAll(".mode-button").forEach((button) => {
    button.classList.toggle("active", button.dataset.workspace === name);
  });
  for (const value of ["extraction", "search", "library", "sources", "summary"]) {
    $(`workspace-${value}`).classList.toggle("hidden", value !== name);
  }
}

function sourceParams() {
  const params = new URLSearchParams({
    workspace_id: workspaceId(),
    page: String(state.sourcePage),
    page_size: $("source-page-size").value || "20",
  });
  const sessionId = $("source-session-id").value.trim();
  if (sessionId) params.set("session_id", sessionId);
  return params;
}

async function loadSources(resetPage = false) {
  const button = $("source-refresh");
  if (resetPage) state.sourcePage = 1;
  button.disabled = true;
  try {
    const response = await fetch(`/api/source-messages?${sourceParams()}`);
    const body = await readResponseBody(response);
    if (!response.ok) throw responseError(body);
    state.sourcesLoaded = true;
    state.sourcePage = body.page;
    state.sourceTotalPages = body.total_pages;
    $("source-results").textContent = "";
    $("source-empty").style.display = body.items.length ? "none" : "grid";
    body.items.forEach((message) => $("source-results").append(sourceMessageCard(message)));
    $("source-meta").textContent = `${body.total} 条原始消息 · 每页 ${body.page_size}`;
    $("source-page-label").textContent = `第 ${body.page} / ${body.total_pages || 1} 页`;
    $("source-prev").disabled = body.page <= 1;
    $("source-next").disabled = body.total_pages === 0 || body.page >= body.total_pages;
  } catch (error) {
    renderResponseError(error);
  } finally {
    button.disabled = false;
  }
}

async function callIngestionApi(button) {
  button.disabled = true;
  try {
    const response = await fetch("/api/session-ingestions", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(await buildSessionPayload()),
    });
    const body = await readResponseBody(response);
    if (response.status === 202) throw responseError(body);
    if (!response.ok) throw responseError(body);
    return body;
  } finally { button.disabled = false; }
}

async function ingest() {
  try {
    const body = await callIngestionApi($("ingest-button"));
    const sessionResult = body.result;
    const result = sessionResult.result;
    state.result = result;
    const candidates = result.trace.extraction?.candidates || [];
    renderCandidates(candidates);
    $("tab-retrieval").textContent = pretty(result.trace.retrieved);
    $("tab-judgment").textContent = pretty({
      mode: result.mode,
      outcomes: result.outcomes,
      requests: result.trace.judgment_requests,
      responses: result.trace.judgment_responses,
    });
    $("tab-persistence").textContent = pretty({
      operation_id: result.operation_id,
      submission_id: sessionResult.submission_id,
      assigned_messages: sessionResult.assigned_messages,
      context: sessionResult.context,
      message_ids: result.message_ids,
      steps: result.trace.persistence_steps,
    });
    $("tab-prompt").textContent = pretty(result.trace.extraction?.trace?.messages || []);
    $("tab-request").textContent = pretty(result.trace.extraction?.trace?.provider_request || null);
    $("tab-raw").textContent = renderRawAttempts(result.trace.extraction?.trace);
    const turns = sessionResult.assigned_messages.map((item) => `${item.role}:${item.turn_id}`).join(" · ");
    $("run-meta").textContent = `${turns} · recent ${sessionResult.context.recent_message_ids.length} · ${result.outcomes.length} candidates`;
    state.submissionId = crypto.randomUUID();
    activateTab("persistence");
  } catch (error) { renderResponseError(error); }
}

async function beforeGeneration() {
  const button = $("before-generation-button");
  button.disabled = true;
  try {
    const sessionId = $("session-id").value.trim();
    const userContent = await currentContent("user");
    if (!sessionId) throw new Error("请输入 Session ID");
    if (!userContent) throw new Error("User 消息不能为空");
    const response = await fetch("/api/before-generation", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        workspace_id: workspaceId(), session_id: sessionId,
        user_content: userContent, top_k: 20,
        request_id: state.lifecycleRequestId,
      }),
    });
    const body = await readResponseBody(response);
    if (!response.ok) throw responseError(body);
    state.activeExchangeId = body.result.exchange_id;
    $("after-generation-button").disabled = false;
    $("lifecycle-output").textContent = pretty(body.result);
    showToast(`User 已保存，召回 ${body.result.memories.length} 条记忆`);
  } catch (error) { renderResponseError(error); }
  finally { button.disabled = false; }
}

async function afterGeneration() {
  if (!state.activeExchangeId) return showToast("请先调用 before_generation");
  const button = $("after-generation-button");
  button.disabled = true;
  try {
    const assistantContent = await currentContent("assistant");
    const response = await fetch("/api/after-generation", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        workspace_id: workspaceId(),
        exchange_id: state.activeExchangeId,
        assistant_content: assistantContent,
      }),
    });
    const body = await readResponseBody(response);
    if (!response.ok) throw responseError(body);
    $("lifecycle-output").textContent = pretty(body.result);
    state.activeExchangeId = null;
    state.lifecycleRequestId = crypto.randomUUID();
    showToast("exchange 已完成并持久化记忆");
  } catch (error) { renderResponseError(error); }
  finally { button.disabled = !state.activeExchangeId; }
}

async function searchMemories() {
  const button = $("search-button");
  button.disabled = true;
  try {
    const text = $("search-text").value.trim();
    const images = await filesToImageSources($("search-images"));
    if (!text && !images.length) throw new Error("请输入查询文字或添加查询图片");
    const response = await fetch("/api/search", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        workspace_id: workspaceId(),
        query: { texts: text ? [text] : [], images },
        top_k: Number($("search-top-k").value || 20),
      }),
    });
    const body = await readResponseBody(response);
    if (!response.ok) throw responseError(body);
    $("search-results").innerHTML = "";
    $("search-empty").style.display = body.results.length ? "none" : "grid";
    if (!body.results.length) {
      $("search-empty").querySelector("h3").textContent = "没有命中记忆";
      $("search-empty").querySelector("p").textContent = "可以换一种描述或加入图片再试。";
    }
    body.results.forEach((result) => {
      $("search-results").append(memoryCard(result.memory, result));
    });
    $("search-meta").textContent = `${body.results.length} 条结果 · ${text ? "text" : ""}${text && images.length ? " + " : ""}${images.length ? "image" : ""}`;
  } catch (error) {
    renderResponseError(error);
  } finally {
    button.disabled = false;
  }
}

function libraryParams() {
  const params = new URLSearchParams({
    workspace_id: workspaceId(),
    page: String(state.libraryPage),
    page_size: $("library-page-size").value || "20",
  });
  const fields = {
    memory_id: $("library-memory-id").value.trim(),
    session_id: $("library-session-id").value.trim(),
    kind: $("library-kind").value.trim(),
    facet_type: $("library-facet-type").value,
  };
  Object.entries(fields).forEach(([key, value]) => { if (value) params.set(key, value); });
  return params;
}

async function loadLibrary(resetPage = false) {
  const button = $("library-refresh");
  if (resetPage) state.libraryPage = 1;
  button.disabled = true;
  try {
    const response = await fetch(`/api/memories?${libraryParams()}`);
    const body = await readResponseBody(response);
    if (!response.ok) throw responseError(body);
    state.libraryLoaded = true;
    state.libraryPage = body.page;
    state.libraryTotalPages = body.total_pages;
    $("library-results").innerHTML = "";
    $("library-empty").style.display = body.items.length ? "none" : "grid";
    if (!body.items.length) {
      $("library-empty").querySelector("h3").textContent = "没有符合条件的记忆";
      $("library-empty").querySelector("p").textContent = "调整筛选条件或先持久化一条记忆。";
    }
    body.items.forEach((item) => $("library-results").append(memoryCard(item)));
    $("library-meta").textContent = `${body.total} 条逻辑记忆 · 每页 ${body.page_size}`;
    $("library-page-label").textContent = `第 ${body.page} / ${body.total_pages || 1} 页`;
    $("library-prev").disabled = body.page <= 1;
    $("library-next").disabled = body.total_pages === 0 || body.page >= body.total_pages;
  } catch (error) {
    renderResponseError(error);
  } finally {
    button.disabled = false;
  }
}

async function loadSummary() {
  const button = $("summary-refresh");
  const sessionId = $("summary-session-id").value.trim();
  if (!sessionId) return showToast("请输入 Session ID");
  button.disabled = true;
  try {
    const response = await fetch(`/api/summaries/${encodeURIComponent(sessionId)}?workspace_id=${encodeURIComponent(workspaceId())}`);
    const body = await readResponseBody(response);
    if (!response.ok) throw responseError(body);
    const status = body.status;
    $("summary-empty").classList.add("hidden");
    $("summary-card").classList.remove("hidden");
    $("summary-text").textContent = status.summary?.text || "尚无已完成摘要";
    $("summary-status-json").textContent = pretty(status);
    const version = status.summary?.version ?? 0;
    const covered = status.summary?.summarized_through_sequence ?? 0;
    const job = status.active_job ? ` · ${status.active_job.status} ${status.active_job.from_sequence}-${status.active_job.through_sequence}` : "";
    $("summary-meta").textContent = `v${version} · 已覆盖 ${covered}/${status.submission_count} · 待处理 ${status.pending_submission_count}${job}`;
  } catch (error) {
    renderResponseError(error);
  } finally {
    button.disabled = false;
  }
}

async function retrySummary() {
  const button = $("summary-retry");
  const sessionId = $("summary-session-id").value.trim();
  if (!sessionId) return showToast("请输入 Session ID");
  button.disabled = true;
  try {
    const response = await fetch(`/api/summaries/${encodeURIComponent(sessionId)}/retry?workspace_id=${encodeURIComponent(workspaceId())}`, { method: "POST" });
    const body = await readResponseBody(response);
    if (!response.ok) throw responseError(body);
    showToast(body.job ? "摘要任务已重新排队" : "该 session 没有失败任务");
    await loadSummary();
  } catch (error) {
    renderResponseError(error);
  } finally {
    button.disabled = false;
  }
}

async function health() {
  try {
    const response = await fetch("/api/health");
    const body = await readResponseBody(response);
    const status = $("status");
    status.classList.add(
      body.live_extraction_configured || body.memory_tools_configured ? "ready" : "offline"
    );
    status.querySelector("strong").textContent = body.live_extraction_configured
      ? `${body.provider} / ${body.model}${body.live_ingestion_configured ? " · persistence ready" : " · extraction only"}${body.memory_tools_configured ? " · search ready" : ""}${body.summarization_configured ? " · summaries ready" : ""}`
      : (body.memory_tools_configured ? "Memory search and library ready" : "Prompt preview only");
  } catch {
    $("status").classList.add("offline");
    $("status").querySelector("strong").textContent = "服务不可用";
  }
}

async function loadWorkspaces() {
  try {
    const response = await fetch("/api/workspaces");
    const body = await readResponseBody(response);
    if (!response.ok || !body.workspaces?.length) return;
    const selected = requestedWorkspaceId();
    $("workspace-select").innerHTML = "";
    body.workspaces.forEach((workspace) => {
      const option = document.createElement("option");
      option.value = workspace.id;
      option.textContent = workspace.id;
      $("workspace-select").append(option);
    });
    $("workspace-select").value = body.workspaces.some((item) => item.id === selected) ? selected : "default";
    rememberWorkspace();
  } catch (error) { showToast(error.message); }
}

async function createWorkspace() {
  const value = window.prompt("新 Workspace ID（字母、数字、点、下划线或短横线）");
  if (!value) return;
  try {
    const response = await fetch("/api/workspaces", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ workspace_id: value.trim() }),
    });
    const body = await readResponseBody(response);
    if (!response.ok) throw responseError(body);
    await loadWorkspaces();
    $("workspace-select").value = body.workspace.id;
    resetWorkspaceState();
    showToast(`已切换到 ${body.workspace.id}`);
  } catch (error) { showToast(error.message); }
}

function resetWorkspaceState() {
  rememberWorkspace();
  state.submissionId = crypto.randomUUID();
  state.lifecycleRequestId = crypto.randomUUID();
  state.activeExchangeId = null;
  state.libraryLoaded = false;
  state.sourcesLoaded = false;
  sourceCache.clear();
  $("after-generation-button").disabled = true;
  $("lifecycle-output").textContent = "尚未调用";
  if (!$("workspace-library").classList.contains("hidden")) loadLibrary(true);
}

document.querySelectorAll(".tab").forEach((tab) => tab.addEventListener("click", () => activateTab(tab.dataset.tab)));
$("ingest-button").addEventListener("click", ingest);
$("before-generation-button").addEventListener("click", beforeGeneration);
$("after-generation-button").addEventListener("click", afterGeneration);
$("workspace-create").addEventListener("click", createWorkspace);
$("workspace-select").addEventListener("change", resetWorkspaceState);
$("search-button").addEventListener("click", searchMemories);
$("library-refresh").addEventListener("click", () => loadLibrary(true));
$("source-refresh").addEventListener("click", () => loadSources(true));
$("summary-refresh").addEventListener("click", loadSummary);
$("summary-retry").addEventListener("click", retrySummary);
$("library-prev").addEventListener("click", () => {
  if (state.libraryPage > 1) { state.libraryPage -= 1; loadLibrary(); }
});
$("library-next").addEventListener("click", () => {
  if (state.libraryPage < state.libraryTotalPages) { state.libraryPage += 1; loadLibrary(); }
});
$("source-prev").addEventListener("click", () => {
  if (state.sourcePage > 1) { state.sourcePage -= 1; loadSources(); }
});
$("source-next").addEventListener("click", () => {
  if (state.sourcePage < state.sourceTotalPages) { state.sourcePage += 1; loadSources(); }
});
document.querySelectorAll(".mode-button").forEach((button) => {
  button.addEventListener("click", () => {
    activateWorkspace(button.dataset.workspace);
    if (button.dataset.workspace === "library" && !state.libraryLoaded) loadLibrary();
    if (button.dataset.workspace === "sources" && !state.sourcesLoaded) loadSources();
    if (button.dataset.workspace === "summary") {
      $("summary-session-id").value = $("session-id").value.trim() || "debug-session";
      loadSummary();
    }
  });
});
document.addEventListener("keydown", (event) => {
  if ((event.metaKey || event.ctrlKey) && event.key === "Enter") ingest();
});
for (const prefix of ["user", "assistant"]) {
  $(`${prefix}-images`).addEventListener("change", (event) => {
    $(`${prefix}-file-list`).textContent = [...event.target.files].map((file) => file.name).join(" · ");
  });
}
$("search-images").addEventListener("change", (event) => {
  $("search-file-list").textContent = [...event.target.files].map((file) => file.name).join(" · ");
});
for (const id of ["session-id", "user-text", "assistant-text", "user-images", "assistant-images"]) {
  $(id).addEventListener("input", () => {
    state.submissionId = crypto.randomUUID();
    state.lifecycleRequestId = crypto.randomUUID();
    state.activeExchangeId = null;
    $("after-generation-button").disabled = true;
  });
}
loadWorkspaces();
health();

