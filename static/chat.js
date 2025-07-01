// Filename: chat.js
// Version: 1.0.0

document.getElementById("searchBtn").addEventListener("click", async () => {
    const query = document.getElementById("searchInput").value.trim();
    if (!query) return;

    const filters = getProgramFilters(); // Extract from checkboxes
    showLoading();

    try {
        const embedResponse = await fetch("/embed", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ texts: [query] })
        });
        const embedData = await embedResponse.json();
        const embedding = embedData.embeddings?.[0];

        if (!embedding) throw new Error("No embedding returned");

        const vectorResponse = await fetch("/search/vector", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                query: query,
                embedding: embedding,
                filters: filters,
                top_k: 5
            })
        });

        const results = await vectorResponse.json();
        renderSearchResults(results);
    } catch (error) {
        console.error("Search failed:", error);
        showError("Search failed. Please try again.");
    }
});

function getProgramFilters() {
    const checkboxes = document.querySelectorAll(".program-filters input[type=checkbox]:checked");
    const selected = Array.from(checkboxes).map(cb => cb.value);
    if (selected.length === 0) return {};
    return {
        "metadata.program": selected[0] // Currently only 1 supported
    };
}

function showLoading() {
    const el = document.getElementById("search-results");
    el.style.display = "block";
    el.innerHTML = '<div class="loading">Searching...</div>';
}

function showError(msg) {
    const el = document.getElementById("search-results");
    el.style.display = "block";
    el.innerHTML = `<div class="empty-state">${msg}</div>`;
}

function renderSearchResults(data) {
    const el = document.getElementById("search-results");
    el.style.display = "block";
    if (!data.results || data.results.length === 0) {
        el.innerHTML = '<div class="empty-state">No results found.</div>';
        return;
    }

    el.innerHTML = data.results.map(hit => {
        const doc = hit._source || hit;
        return `
        <div class="search-result-item">
            <div class="search-title">${doc.metadata?.agent || "Agent"} • ${doc.metadata?.disposition || "Unknown"}</div>
            <div class="search-preview">${doc.text}</div>
            <div class="search-collection">${doc.metadata?.template || "unknown collection"}</div>
        </div>`;
    }).join("");
}
