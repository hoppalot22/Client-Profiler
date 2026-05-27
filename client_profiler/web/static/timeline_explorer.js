(function () {
  const root = document.querySelector("[data-timeline-visual]");
  const source = document.getElementById("timeline-visual-data");
  if (!root || !source) {
    return;
  }

  const items = JSON.parse(source.textContent || "[]");
  if (!Array.isArray(items) || items.length === 0) {
    return;
  }

  const viewport = root.querySelector(".timeline-visual-viewport");
  const track = root.querySelector(".timeline-visual-track");
  const scaleReadout = root.querySelector("[data-scale-readout]");
  const padding = 48;
  const millisecondsPerDay = 24 * 60 * 60 * 1000;

  const timestamps = items
    .map((item) => Number(item.timestamp) || 0)
    .sort((left, right) => left - right);
  const minTime = timestamps[0] || Date.now();
  const maxTime = timestamps[timestamps.length - 1] || minTime + millisecondsPerDay;
  const rangeDays = Math.max(2, (maxTime - minTime) / millisecondsPerDay || 2);

  let pixelsPerDay = Math.max(24, Math.min(110, (viewport.clientHeight - padding * 2) / Math.min(rangeDays, 28)));
  const baseScale = pixelsPerDay;

  function render() {
    const totalHeight = Math.max(viewport.clientHeight, padding * 2 + rangeDays * pixelsPerDay);
    track.style.height = `${totalHeight}px`;
    track.replaceChildren();

    const axis = document.createElement("div");
    axis.className = "timeline-visual-axis";
    track.appendChild(axis);

    items.forEach((item, index) => {
      const timestamp = Number(item.timestamp) || minTime;
      const top = padding + ((timestamp - minTime) / millisecondsPerDay) * pixelsPerDay;
      const event = document.createElement("article");
      event.className = `timeline-visual-event ${index % 2 === 0 ? "is-left" : "is-right"}`;
      event.style.top = `${top}px`;

      const marker = document.createElement("span");
      marker.className = "timeline-visual-marker";
      event.appendChild(marker);

      const leader = document.createElement("span");
      leader.className = "timeline-visual-leader";
      event.appendChild(leader);

      const label = document.createElement("div");
      label.className = "timeline-visual-label";

      const title = document.createElement("strong");
      title.textContent = item.summary;
      label.appendChild(title);

      const documentLink = document.createElement("a");
      documentLink.href = item.document_href;
      documentLink.target = "_blank";
      documentLink.rel = "noopener noreferrer";
      documentLink.textContent = item.document_name;
      label.appendChild(documentLink);

      const meta = document.createElement("div");
      meta.className = "timeline-visual-meta";
      meta.textContent = [item.client_name, item.project_name, item.document_kind]
        .filter(Boolean)
        .join(" • ");
      label.appendChild(meta);

      const dateLabel = document.createElement("span");
      dateLabel.className = "timeline-visual-date";
      dateLabel.textContent = item.date_label;
      label.appendChild(dateLabel);

      event.appendChild(label);
      track.appendChild(event);
    });

    if (scaleReadout) {
      scaleReadout.textContent = `${(pixelsPerDay / baseScale).toFixed(1)}x`;
    }
  }

  viewport.addEventListener(
    "wheel",
    (event) => {
      event.preventDefault();
      const rect = viewport.getBoundingClientRect();
      const cursorOffset = event.clientY - rect.top;
      const contentOffset = viewport.scrollTop + cursorOffset;
      const cursorTime = minTime + ((contentOffset - padding) / pixelsPerDay) * millisecondsPerDay;
      const factor = event.deltaY < 0 ? 1.14 : 0.88;
      const nextScale = Math.max(8, Math.min(220, pixelsPerDay * factor));
      if (nextScale === pixelsPerDay) {
        return;
      }
      pixelsPerDay = nextScale;
      render();
      const newOffset = padding + ((cursorTime - minTime) / millisecondsPerDay) * pixelsPerDay;
      viewport.scrollTop = Math.max(0, newOffset - cursorOffset);
    },
    { passive: false }
  );

  render();
  viewport.scrollTop = Math.max(0, track.scrollHeight - viewport.clientHeight);
})();