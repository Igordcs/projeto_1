
function renderPlotlyChart(elementId, figureJson) {
  const container = document.getElementById(elementId);
  if (!container || !figureJson) return;
  const fig = typeof figureJson === "string" ? JSON.parse(figureJson) : figureJson;
  const config = {
    responsive: true,
    displaylogo: false,
    modeBarButtonsToRemove: ["lasso2d", "select2d"],
  };
  Plotly.newPlot(container, fig.data || [], fig.layout || {}, config);
}

document.addEventListener("DOMContentLoaded", () => {
  document.querySelectorAll("select[data-autosubmit='true']").forEach((sel) => {
    sel.addEventListener("change", () => {
      if (sel.form) {
        sel.form.submit();
      }
    });
  });
});

