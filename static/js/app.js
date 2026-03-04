/**
 * Dashboard JavaScript — handles chat interaction, API calls,
 * and UI updates for the Knowledge Graph Query System.
 */

(function () {
    "use strict";

    // DOM references
    const chatMessages = document.getElementById("chat-messages");
    const questionForm = document.getElementById("question-form");
    const questionInput = document.getElementById("question-input");
    const btnSend = document.getElementById("btn-send");
    const btnClear = document.getElementById("btn-clear");
    const btnSchema = document.getElementById("btn-schema");
    const btnHealth = document.getElementById("btn-health");

    // Detail panel elements
    const detailsPanel = document.getElementById("details-panel");
    const detailType = document.getElementById("detail-type");
    const detailScore = document.getElementById("detail-score");
    const detailCached = document.getElementById("detail-cached");
    const detailCypher = document.getElementById("detail-cypher");
    const cypherRow = document.getElementById("cypher-row");
    const detailIssues = document.getElementById("detail-issues");
    const issuesRow = document.getElementById("issues-row");
    const detailSteps = document.getElementById("detail-steps");
    const stepsRow = document.getElementById("steps-row");

    // Modals
    const schemaModal = document.getElementById("schema-modal");
    const healthModal = document.getElementById("health-modal");

    let isProcessing = false;

    // ===== Chat Functions =====

    function addMessage(role, content) {
        // Remove welcome message if present
        const welcome = chatMessages.querySelector(".welcome-message");
        if (welcome) welcome.remove();

        const div = document.createElement("div");
        div.className = `message ${role}`;
        div.setAttribute("role", "article");

        const bubble = document.createElement("div");
        bubble.className = "message-bubble";

        if (role === "assistant") {
            bubble.innerHTML = renderMarkdown(content);
        } else {
            bubble.textContent = content;
        }

        div.appendChild(bubble);
        chatMessages.appendChild(div);
        chatMessages.scrollTop = chatMessages.scrollHeight;
        return div;
    }

    function addTypingIndicator() {
        const div = document.createElement("div");
        div.className = "message assistant";
        div.id = "typing";
        div.innerHTML = `
            <div class="typing-indicator">
                <span></span><span></span><span></span>
            </div>
        `;
        chatMessages.appendChild(div);
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }

    function removeTypingIndicator() {
        const el = document.getElementById("typing");
        if (el) el.remove();
    }

    // ===== Detail Panel =====

    function updateDetails(data) {
        detailsPanel.hidden = false;

        // Type badge
        const qtype = data.query_type || "graph";
        detailType.textContent = qtype;
        detailType.className = `badge badge-${qtype}`;

        // Confidence score
        if (data.validation && data.validation.score != null) {
            const score = data.validation.score;
            const pct = (score * 100).toFixed(0) + "%";
            let cls = "score-high";
            if (score < 0.4) cls = "score-low";
            else if (score < 0.7) cls = "score-med";
            detailScore.textContent = pct;
            detailScore.className = cls;
        } else {
            detailScore.textContent = "N/A";
            detailScore.className = "";
        }

        // Cached
        detailCached.textContent = data.cached ? "Yes ⚡" : "No";

        // Cypher
        if (data.cypher) {
            cypherRow.hidden = false;
            detailCypher.textContent = data.cypher;
        } else {
            cypherRow.hidden = true;
        }

        // Issues
        if (data.validation && data.validation.issues && data.validation.issues.length > 0) {
            issuesRow.hidden = false;
            detailIssues.innerHTML = data.validation.issues
                .map(i => `<li>${escapeHtml(i)}</li>`)
                .join("");
        } else {
            issuesRow.hidden = true;
        }

        // Agent steps
        if (data.steps && data.steps.length > 0) {
            stepsRow.hidden = false;
            detailSteps.innerHTML = data.steps.map((s, i) => `
                <div class="step">
                    <div class="step-tool">Step ${i + 1}: ${escapeHtml(s.tool)}</div>
                    <div>${escapeHtml(s.input || "")}</div>
                    <div class="step-result">${escapeHtml(s.result || "")}</div>
                </div>
            `).join("");
        } else {
            stepsRow.hidden = true;
        }
    }

    // ===== API Calls =====

    async function askQuestion(question) {
        if (isProcessing) return;
        isProcessing = true;
        btnSend.disabled = true;

        addMessage("user", question);
        addTypingIndicator();

        try {
            const res = await fetch("/api/ask", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ question }),
            });

            removeTypingIndicator();

            if (!res.ok) {
                const err = await res.json();
                addMessage("assistant", `Error: ${err.error || "Something went wrong."}`);
                return;
            }

            const data = await res.json();
            addMessage("assistant", data.answer || "No answer received.");
            updateDetails(data);
            updateCacheStats();
        } catch (err) {
            removeTypingIndicator();
            addMessage("assistant", `Connection error: ${err.message}`);
        } finally {
            isProcessing = false;
            btnSend.disabled = false;
            questionInput.focus();
        }
    }

    async function clearConversation() {
        try {
            await fetch("/api/clear", { method: "POST" });
            chatMessages.innerHTML = `
                <div class="welcome-message">
                    <h2>Welcome! Ask me anything about Stack Overflow.</h2>
                    <p>I can answer questions using graph traversal, semantic similarity, or multi-step reasoning.</p>
                </div>
            `;
            detailsPanel.hidden = true;
        } catch (err) {
            console.error("Failed to clear conversation:", err);
        }
    }

    async function updateCacheStats() {
        try {
            const res = await fetch("/api/cache/stats");
            const data = await res.json();
            const el = document.getElementById("cache-stats");
            el.innerHTML = `
                <div class="detail-row"><span class="detail-label">Size</span> ${data.size}</div>
                <div class="detail-row"><span class="detail-label">Hits</span> ${data.hits}</div>
                <div class="detail-row"><span class="detail-label">Misses</span> ${data.misses}</div>
                <div class="detail-row"><span class="detail-label">Hit Rate</span> ${data.hit_rate}</div>
            `;
        } catch (err) {
            // ignore
        }
    }

    async function showSchema() {
        openModal(schemaModal);
        try {
            const res = await fetch("/api/schema");
            const data = await res.json();
            document.getElementById("schema-body").textContent = data.schema || "No schema available.";
        } catch (err) {
            document.getElementById("schema-body").textContent = "Failed to load schema.";
        }
    }

    async function showHealth() {
        openModal(healthModal);
        const body = document.getElementById("health-body");
        body.innerHTML = "<p>Checking...</p>";
        try {
            const res = await fetch("/api/health");
            const data = await res.json();
            const statusColor = data.status === "healthy" ? "var(--success)" : "var(--danger)";
            body.innerHTML = `
                <div class="detail-row">
                    <span class="detail-label">Status</span>
                    <span style="color:${statusColor}; font-weight:600">${data.status}</span>
                </div>
                <div class="detail-row">
                    <span class="detail-label">Database</span>
                    ${data.database || "unknown"}
                </div>
            `;
        } catch (err) {
            body.innerHTML = `<p style="color:var(--danger)">Health check failed: ${err.message}</p>`;
        }
    }

    // ===== Modal Helpers =====

    function openModal(modal) {
        modal.classList.add("open");
        modal.setAttribute("aria-hidden", "false");
        const closeBtn = modal.querySelector(".modal-close");
        if (closeBtn) closeBtn.focus();
    }

    function closeModal(modal) {
        modal.classList.remove("open");
        modal.setAttribute("aria-hidden", "true");
    }

    // ===== Utilities =====

    function escapeHtml(text) {
        const div = document.createElement("div");
        div.textContent = text;
        return div.innerHTML;
    }

    /**
     * Lightweight markdown-to-HTML renderer for assistant messages.
     * Handles: bold, italic, inline code, code blocks, links,
     *          numbered lists, bullet lists, headings, and paragraphs.
     */
    function renderMarkdown(text) {
        let html = escapeHtml(text);

        // Fenced code blocks: ```lang\n...\n```
        html = html.replace(/```(\w*)\n([\s\S]*?)```/g, function (_m, lang, code) {
            return '<pre class="md-code-block"><code>' + code.trim() + '</code></pre>';
        });

        // Inline code: `code`
        html = html.replace(/`([^`]+)`/g, '<code class="md-inline-code">$1</code>');

        // Bold: **text**
        html = html.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');

        // Italic: *text*  (but not inside words like file_names)
        html = html.replace(/(^|[\s(])\*([^*]+?)\*(?=[\s).,;:!?]|$)/gm, '$1<em>$2</em>');

        // Links: [text](url)
        html = html.replace(/\[([^\]]+)\]\(([^)]+)\)/g,
            '<a href="$2" target="_blank" rel="noopener">$1</a>');

        // Split into lines for block-level processing
        const lines = html.split('\n');
        const output = [];
        let inList = false;
        let listType = null; // 'ol' or 'ul'

        for (let i = 0; i < lines.length; i++) {
            const line = lines[i];
            const olMatch = line.match(/^(\d+)\.\s+(.+)/);
            const ulMatch = line.match(/^[-*]\s+(.+)/);
            const headingMatch = line.match(/^(#{1,3})\s+(.+)/);

            if (olMatch) {
                if (!inList || listType !== 'ol') {
                    if (inList) output.push(listType === 'ol' ? '</ol>' : '</ul>');
                    output.push('<ol class="md-list">');
                    inList = true;
                    listType = 'ol';
                }
                output.push('<li>' + olMatch[2] + '</li>');
            } else if (ulMatch) {
                if (!inList || listType !== 'ul') {
                    if (inList) output.push(listType === 'ol' ? '</ol>' : '</ul>');
                    output.push('<ul class="md-list">');
                    inList = true;
                    listType = 'ul';
                }
                output.push('<li>' + ulMatch[1] + '</li>');
            } else {
                if (inList) {
                    output.push(listType === 'ol' ? '</ol>' : '</ul>');
                    inList = false;
                    listType = null;
                }
                if (headingMatch) {
                    const level = headingMatch[1].length;
                    const tag = 'h' + (level + 2); // h3-h5 range
                    output.push('<' + tag + ' class="md-heading">' + headingMatch[2] + '</' + tag + '>');
                } else if (line.trim() === '') {
                    output.push('');
                } else {
                    output.push('<p class="md-para">' + line + '</p>');
                }
            }
        }
        if (inList) {
            output.push(listType === 'ol' ? '</ol>' : '</ul>');
        }

        // Collapse consecutive empty entries
        return output.filter((l, i) => !(l === '' && output[i - 1] === '')).join('\n');
    }

    // ===== Event Listeners =====

    questionForm.addEventListener("submit", (e) => {
        e.preventDefault();
        const q = questionInput.value.trim();
        if (!q) return;
        questionInput.value = "";
        askQuestion(q);
    });

    btnClear.addEventListener("click", clearConversation);
    btnSchema.addEventListener("click", showSchema);
    btnHealth.addEventListener("click", showHealth);

    // Example buttons
    document.querySelectorAll(".example-btn").forEach((btn) => {
        btn.addEventListener("click", () => {
            const q = btn.getAttribute("data-question");
            if (q) {
                questionInput.value = q;
                questionInput.focus();
            }
        });
    });

    // Modal close handlers
    document.querySelectorAll(".modal-close, .modal-backdrop").forEach((el) => {
        el.addEventListener("click", () => {
            document.querySelectorAll(".modal.open").forEach(closeModal);
        });
    });

    document.addEventListener("keydown", (e) => {
        if (e.key === "Escape") {
            document.querySelectorAll(".modal.open").forEach(closeModal);
        }
    });

    // Initial cache stats
    updateCacheStats();
})();
