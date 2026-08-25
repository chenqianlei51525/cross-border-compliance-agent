// 应小合规 - 前端逻辑
// 1. 启动时拉工具箱 → 侧栏展示
// 2. 点击 "发送" / "Agent 轨迹" 分别走：同步请求 / SSE 流式

const sessionId = (crypto.randomUUID && crypto.randomUUID()) ||
                  ("sess-" + Math.random().toString(36).slice(2));
let es = null;

const $messages = () => document.getElementById("messages");
const $trace = () => document.getElementById("trace");
const $tools = () => document.getElementById("tool-list");
const $citations = () => document.getElementById("citations");

function appendMessage(role, text) {
  const el = document.createElement("div");
  el.className = "msg " + role;
  el.innerHTML = `<div class="bubble"></div>`;
  el.querySelector(".bubble").textContent = text;
  $messages().appendChild(el);
  $messages().scrollTop = $messages().scrollHeight;
  return el;
}

function appendToolStep(step) {
  const trace = $trace();
  if (!step) return;
  const node = document.createElement("div");
  node.className = "step";
  node.innerHTML = `
    <div class="thought">💭 ${step.thought || ""}</div>
    <div class="action">⚡ ${step.action || ""} ${step.action_input
      ? "· " + JSON.stringify(step.action_input).slice(0, 80) : ""}</div>
    <div class="obs">📄 ${(step.observation || "").slice(0, 200)}${
      (step.observation || "").length > 200 ? "…" : ""}</div>
  `;
  trace.prepend(node);
}

function appendCitations(citations) {
  if (!citations || !citations.length) return;
  citations.forEach((c) => {
    const node = document.createElement("div");
    node.className = "citation-item";
    node.textContent = `[${c.id}] ${c.title}`;
    $citations().prepend(node);
  });
}

async function loadTools() {
  try {
    const resp = await fetch("/api/agent/tools");
    const data = await resp.json();
    $tools().innerHTML = (data.tools || [])
      .map((t) => `<li><code>${t.name}</code><br/>${t.description}</li>`)
      .join("");
  } catch (e) {
    $tools().innerHTML = `<li style="color:#c00">工具列表加载失败：${e}</li>`;
  }
}

async function askSync(question, sourceFilter) {
  appendMessage("user", question);
  const placeholder = appendMessage("assistant", "思考中…");

  const resp = await fetch("/api/agent/chat", {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify({query: question, source_filter: sourceFilter, session_id: sessionId}),
  });
  const data = await resp.json();
  placeholder.querySelector(".bubble").textContent = data.answer || "（无答案）";

  // 把轨迹/引用渲染
  if (data.trace && data.trace.steps) {
    data.trace.steps.slice().reverse().forEach(appendToolStep);
  }
  // 从最终答案文本里抽 [kg_x] [hit_x] 这种编号，提取引用
  const refs = (data.answer || "").match(/\[\d+\]/g) || [];
  refs.forEach((r) =>
    appendCitations([{id: r.replace(/[\[\]]/g, ""), title: data.trace ? "引用片段" : ""}])
  );
}

function askStream(question, sourceFilter) {
  appendMessage("user", question);
  const placeholder = appendMessage("assistant", "");
  $trace().innerHTML = "";
  $citations().innerHTML = "";

  es = new EventSource(
    `/api/agent/stream?query=${encodeURIComponent(question)}` +
    `&source_filter=${encodeURIComponent(sourceFilter || "")}` +
    `&session_id=${encodeURIComponent(sessionId)}`
  );

  // 由于 EventSource 不支持 POST + body，改成 fetch + ReadableStream
  es.close();
  fetchStream(question, sourceFilter, placeholder);
}

async function fetchStream(question, sourceFilter, placeholder) {
  const resp = await fetch("/api/agent/stream", {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify({query: question, source_filter: sourceFilter, session_id: sessionId}),
  });
  if (!resp.ok) {
    placeholder.querySelector(".bubble").textContent = "请求失败";
    return;
  }
  const reader = resp.body.getReader();
  const decoder = new TextDecoder();
  let buf = "";
  let finalAnswer = "";
  while (true) {
    const {value, done} = await reader.read();
    if (done) break;
    buf += decoder.decode(value, {stream: true});
    buf = buf.replace(/\r\n/g, "\n");
    const events = buf.split("\n\n");
    buf = events.pop() || "";
    for (const ev of events) {
      if (!ev.startsWith("data:")) continue;
      const payload = ev.slice(5).trim();
      if (!payload) continue;
      let data;
      try { data = JSON.parse(payload); } catch (e) { continue; }
      if (data.type === "step") appendToolStep(data.step);
      else if (data.type === "final") {
        finalAnswer = data.answer || finalAnswer;
        placeholder.querySelector(".bubble").textContent = finalAnswer;
      }
    }
  }
}

function bind() {
  document.getElementById("send-btn").onclick = () => {
    const q = document.getElementById("user-input").value.trim();
    if (!q) return;
    const sf = document.getElementById("source-filter").value;
    askSync(q, sf);
    document.getElementById("user-input").value = "";
  };
  document.getElementById("stream-btn").onclick = () => {
    const q = document.getElementById("user-input").value.trim();
    if (!q) return;
    const sf = document.getElementById("source-filter").value;
    askStream(q, sf);
    document.getElementById("user-input").value = "";
  };
  document.getElementById("user-input").addEventListener("keydown", (e) => {
    if (e.key === "Enter") document.getElementById("send-btn").click();
  });
}

bind();
loadTools();
