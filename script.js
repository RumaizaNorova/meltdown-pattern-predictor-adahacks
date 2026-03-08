const STORAGE_KEY = "meltdownLogs";

function getDefaultSampleLogs() {
    return [
        { date: "2026-01-15", sleepHours: 8, noiseLevel: "Low", sugarAfter6: "No", screenAfter7: "No", routineChange: "No", mealAfter7: "No", meltdownOccurred: "No" },
        { date: "2026-01-16", sleepHours: 6, noiseLevel: "High", sugarAfter6: "Yes", screenAfter7: "Yes", routineChange: "Yes", mealAfter7: "Yes", meltdownOccurred: "Yes" },
        { date: "2026-01-17", sleepHours: 7.5, noiseLevel: "Medium", sugarAfter6: "No", screenAfter7: "No", routineChange: "No", mealAfter7: "No", meltdownOccurred: "No" },
        { date: "2026-01-18", sleepHours: 5.5, noiseLevel: "High", sugarAfter6: "Yes", screenAfter7: "Yes", routineChange: "No", mealAfter7: "Yes", meltdownOccurred: "Yes" },
        { date: "2026-01-20", sleepHours: 8.5, noiseLevel: "Low", sugarAfter6: "No", screenAfter7: "No", routineChange: "No", mealAfter7: "No", meltdownOccurred: "No" },
        { date: "2026-01-21", sleepHours: 6.5, noiseLevel: "Medium", sugarAfter6: "No", screenAfter7: "Yes", routineChange: "Yes", mealAfter7: "No", meltdownOccurred: "Yes" },
        { date: "2026-01-22", sleepHours: 7, noiseLevel: "Low", sugarAfter6: "No", screenAfter7: "No", routineChange: "No", mealAfter7: "No", meltdownOccurred: "No" },
        { date: "2026-01-23", sleepHours: 5, noiseLevel: "High", sugarAfter6: "Yes", screenAfter7: "Yes", routineChange: "Yes", mealAfter7: "Yes", meltdownOccurred: "Yes" },
        { date: "2026-01-25", sleepHours: 8, noiseLevel: "Medium", sugarAfter6: "No", screenAfter7: "No", routineChange: "No", mealAfter7: "No", meltdownOccurred: "No" },
        { date: "2026-01-26", sleepHours: 7, noiseLevel: "High", sugarAfter6: "No", screenAfter7: "Yes", routineChange: "No", mealAfter7: "Yes", meltdownOccurred: "Yes" },
        { date: "2026-01-27", sleepHours: 9, noiseLevel: "Low", sugarAfter6: "No", screenAfter7: "No", routineChange: "No", mealAfter7: "No", meltdownOccurred: "No" },
        { date: "2026-01-28", sleepHours: 6, noiseLevel: "Medium", sugarAfter6: "Yes", screenAfter7: "Yes", routineChange: "Yes", mealAfter7: "Yes", meltdownOccurred: "Yes" },
        { date: "2026-01-30", sleepHours: 7.5, noiseLevel: "Low", sugarAfter6: "No", screenAfter7: "No", routineChange: "No", mealAfter7: "No", meltdownOccurred: "No" },
        { date: "2026-01-31", sleepHours: 6.5, noiseLevel: "High", sugarAfter6: "No", screenAfter7: "Yes", routineChange: "Yes", mealAfter7: "No", meltdownOccurred: "Yes" },
        { date: "2026-02-02", sleepHours: 8, noiseLevel: "Medium", sugarAfter6: "No", screenAfter7: "No", routineChange: "No", mealAfter7: "No", meltdownOccurred: "No" },
        { date: "2026-02-03", sleepHours: 4.5, noiseLevel: "High", sugarAfter6: "Yes", screenAfter7: "Yes", routineChange: "Yes", mealAfter7: "Yes", meltdownOccurred: "Yes" },
        { date: "2026-02-04", sleepHours: 7, noiseLevel: "Low", sugarAfter6: "No", screenAfter7: "No", routineChange: "No", mealAfter7: "No", meltdownOccurred: "No" },
        { date: "2026-02-06", sleepHours: 6, noiseLevel: "High", sugarAfter6: "Yes", screenAfter7: "Yes", routineChange: "No", mealAfter7: "Yes", meltdownOccurred: "Yes" },
        { date: "2026-02-07", sleepHours: 8.5, noiseLevel: "Low", sugarAfter6: "No", screenAfter7: "No", routineChange: "No", mealAfter7: "No", meltdownOccurred: "No" },
        { date: "2026-02-08", sleepHours: 7, noiseLevel: "Medium", sugarAfter6: "No", screenAfter7: "Yes", routineChange: "Yes", mealAfter7: "Yes", meltdownOccurred: "No" },
        { date: "2026-02-10", sleepHours: 6.5, noiseLevel: "High", sugarAfter6: "No", screenAfter7: "Yes", routineChange: "No", mealAfter7: "Yes", meltdownOccurred: "Yes" },
    ];
}

const logForm = document.getElementById("logForm");
const logsTableBody = document.querySelector("#logsTable tbody");
const emptyState = document.getElementById("emptyState");
const insightsDiv = document.getElementById("insights");
const sampleDataBtn = document.getElementById("sampleDataBtn");
const analyzeBtn = document.getElementById("analyzeBtn");
const clearBtn = document.getElementById("clearBtn");
const predictBtn = document.getElementById("predictBtn");
const predictionCard = document.getElementById("predictionCard");
const predictionResult = document.getElementById("predictionResult");

function getLogs() {
    const logs = localStorage.getItem(STORAGE_KEY);
    return logs ? JSON.parse(logs) : [];
}

function saveLogs(logs) {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(logs));
}

function renderLogs() {
    const logs = getLogs();
    logsTableBody.innerHTML = "";

    if (logs.length === 0) {
        emptyState.style.display = "block";
        return;
    }

    emptyState.style.display = "none";

    const sortedLogs = [...logs].sort((a, b) => (b.date > a.date ? 1 : -1));

    sortedLogs.forEach((log) => {
        const row = document.createElement("tr");
        const meltdownClass = log.meltdownOccurred === "Yes" ? "meltdown-yes" : "meltdown-no";
        row.innerHTML = `
      <td>${log.date}</td>
      <td>${log.sleepHours}</td>
      <td>${log.noiseLevel}</td>
      <td>${log.sugarAfter6}</td>
      <td>${log.screenAfter7}</td>
      <td>${log.routineChange}</td>
      <td>${log.mealAfter7}</td>
      <td class="${meltdownClass}">${log.meltdownOccurred}</td>
    `;
        logsTableBody.appendChild(row);
    });
}

function addLog(log) {
    const logs = getLogs();
    logs.push(log);
    saveLogs(logs);
    renderLogs();
}

function meltdownRate(conditionFn) {
    const logs = getLogs();
    const filtered = logs.filter(conditionFn);

    if (filtered.length === 0) {
        return null;
    }

    const meltdownCount = filtered.filter(
        (log) => log.meltdownOccurred === "Yes"
    ).length;

    return {
        total: filtered.length,
        meltdowns: meltdownCount,
        rate: (meltdownCount / filtered.length) * 100,
    };
}

function analyzePatterns() {
    const logs = getLogs();

    if (logs.length < 3) {
        if (triggerChartInstance) {
            triggerChartInstance.destroy();
            triggerChartInstance = null;
        }
        insightsDiv.innerHTML = `
      <div class="warning-box">
        Please add at least 3 logs so the analysis makes sense.
      </div>
    `;
        return;
    }

    const patternDefs = [
        {
            name: "Sleep below 7 hours",
            whenPresent: (log) => Number(log.sleepHours) < 7,
            explanation:
                "Lower sleep may increase stress, irritability, or sensitivity to sensory input.",
        },
        {
            name: "High sensory environment",
            whenPresent: (log) => log.noiseLevel === "High",
            explanation:
                "Loud environments may contribute to sensory overload and emotional distress.",
        },
        {
            name: "Late sugar intake",
            whenPresent: (log) => log.sugarAfter6 === "Yes",
            explanation:
                "Late sugar intake may be linked to energy spikes and difficulty regulating emotions.",
        },
        {
            name: "Late screen exposure",
            whenPresent: (log) => log.screenAfter7 === "Yes",
            explanation:
                "Late screen exposure may interfere with sleep preparation and emotional regulation.",
        },
        {
            name: "Routine change",
            whenPresent: (log) => log.routineChange === "Yes",
            explanation:
                "Unexpected routine changes may increase anxiety and difficulty with transitions.",
        },
        {
            name: "Late meal",
            whenPresent: (log) => log.mealAfter7 === "Yes",
            explanation:
                "Late meals may be connected to discomfort, disrupted routine, or sleep issues.",
        },
    ];

    const patterns = patternDefs.map((def) => ({
        ...def,
        whenPresentData: meltdownRate(def.whenPresent),
        whenAbsentData: meltdownRate((log) => !def.whenPresent(log)),
    }));

    let html = "";

    const overallMeltdownRate = meltdownRate(() => true);
    html += `
    <div class="insight-box">
      <strong>Overall meltdown rate:</strong>
      ${overallMeltdownRate.rate.toFixed(1)}% 
      (${overallMeltdownRate.meltdowns} out of ${overallMeltdownRate.total} logs)
    </div>
  `;

    patterns.forEach((pattern) => {
        const present = pattern.whenPresentData;
        const absent = pattern.whenAbsentData;
        if (!present && !absent) return;

        const presentStr = present
            ? `${present.rate.toFixed(1)}% when present (${present.meltdowns}/${present.total})`
            : "no data when present";
        const absentStr = absent
            ? `${absent.rate.toFixed(1)}% when absent (${absent.meltdowns}/${absent.total})`
            : "no data when absent";

        html += `
      <div class="insight-box">
        <strong>${pattern.name}</strong><br>
        Meltdown rate: <strong>${presentStr}</strong><br>
        vs ${absentStr}<br><br>
        <em>${pattern.explanation}</em>
      </div>
    `;
    });

    const strongTriggers = patterns
        .filter((p) => p.whenPresentData && p.whenPresentData.total >= 1)
        .map((p) => ({
            ...p,
            lift:
                p.whenPresentData.rate -
                (p.whenAbsentData ? p.whenAbsentData.rate : 0),
        }))
        .sort((a, b) => b.lift - a.lift)
        .slice(0, 3);

    if (strongTriggers.length > 0) {
        html += `
      <div class="warning-box">
        <strong>Top possible triggers (biggest difference vs. baseline):</strong><br>
        ${strongTriggers
                .map(
                    (trigger, index) =>
                        `${index + 1}. ${trigger.name} (+${trigger.lift.toFixed(0)}% when present vs absent)`
                )
                .join("<br>")}
      </div>
    `;
    }

    html += `
    <div class="disclaimer-box">
      <strong>Important:</strong> This tool does not diagnose autism or predict behavior with certainty.
      It is designed to help caregivers notice patterns that may support better routine planning and understanding.
    </div>
  `;

    const chartCard = `
    <div class="card">
      <h2>Trigger Visualization</h2>
      <canvas id="triggerChart"></canvas>
    </div>
  `;
    insightsDiv.innerHTML = chartCard + html;
    drawTriggerChart(patterns);
}

logForm.addEventListener("submit", function (event) {
    event.preventDefault();

    const log = {
        date: document.getElementById("date").value,
        sleepHours: document.getElementById("sleepHours").value,
        noiseLevel: document.getElementById("noiseLevel").value,
        sugarAfter6: document.getElementById("sugarAfter6").value,
        screenAfter7: document.getElementById("screenAfter7").value,
        routineChange: document.getElementById("routineChange").value,
        mealAfter7: document.getElementById("mealAfter7").value,
        meltdownOccurred: document.getElementById("meltdownOccurred").value,
    };

    addLog(log);
    logForm.reset();
});

sampleDataBtn.addEventListener("click", function () {
    saveLogs(getDefaultSampleLogs());
    renderLogs();
    if (triggerChartInstance) {
        triggerChartInstance.destroy();
        triggerChartInstance = null;
    }
    insightsDiv.innerHTML = `
    <div class="insight-box">
      Sample data loaded. Now click <strong>Analyze Patterns</strong>.
    </div>
  `;
});

analyzeBtn.addEventListener("click", analyzePatterns);

let predictGrid = null;

async function loadPredictGrid() {
    if (predictGrid) return predictGrid;
    const res = await fetch("predict_grid.json");
    if (!res.ok) throw new Error("Could not load prediction model");
    predictGrid = await res.json();
    return predictGrid;
}

function getPredictionKey(sleep, noise, sugar, screen, routine, meal) {
    const sleepRounded = Math.round(Math.max(4, Math.min(12, parseFloat(sleep) || 7)) * 2) / 2;
    const sleepStr = Number.isInteger(sleepRounded) ? sleepRounded + ".0" : String(sleepRounded);
    return `${sleepStr}|${noise}|${sugar}|${screen}|${routine}|${meal}`;
}

function getTopContributors(sleep, noise, sugar, screen, routine, meal) {
    const contrib = [];
    if (parseFloat(sleep) < 7) contrib.push("Low sleep (<7h)");
    if (noise === "High") contrib.push("High sensory environment");
    if (noise === "Medium") contrib.push("Medium sensory environment");
    if (routine === "Yes") contrib.push("Routine change");
    if (screen === "Yes") contrib.push("Late screen exposure");
    if (sugar === "Yes") contrib.push("Late sugar intake");
    if (meal === "Yes") contrib.push("Late meal");
    return contrib;
}

let contributorChartInstance = null;

function drawContributorChart(contributors) {
    const ctx = document.getElementById("contributorChart");
    if (!ctx) return;
    if (contributorChartInstance) {
        contributorChartInstance.destroy();
        contributorChartInstance = null;
    }
    contributorChartInstance = new Chart(ctx, {
        type: "bar",
        data: {
            labels: contributors,
            datasets: [{
                label: "Model weight",
                data: contributors.map((_, i) => 100 - i * 15),
                backgroundColor: "rgba(124, 58, 237, 0.25)",
                borderColor: "rgba(124, 58, 237, 0.6)",
                borderWidth: 2,
                borderRadius: 6,
            }],
        },
        options: {
            indexAxis: "y",
            responsive: true,
            maintainAspectRatio: false,
            aspectRatio: 1.2,
            plugins: {
                legend: { display: false },
                tooltip: { enabled: false },
            },
            scales: {
                x: { display: false, max: 100 },
                y: {
                    grid: { display: false },
                    ticks: { font: { size: 13 }, color: "#78716C" },
                },
            },
        },
    });
}

function renderWhatIf(sleep, noise, sugar, screen, routine, meal, currentPred) {
    const wrap = document.getElementById("whatIfSection");
    if (!wrap) return;
    const sleepNum = parseFloat(sleep) || 7;
    const suggestions = [];
    if (sleepNum < 7) {
        const key8 = getPredictionKey("8", noise, sugar, screen, routine, meal);
        const pred8 = predictGrid[key8];
        if (pred8 && pred8.probability < currentPred.probability) {
            suggestions.push({ label: "Sleep 8h", key: key8, pred: pred8 });
        }
    }
    if (noise !== "Low") {
        const keyLow = getPredictionKey(sleep, "Low", sugar, screen, routine, meal);
        const predLow = predictGrid[keyLow];
        if (predLow && predLow.probability < currentPred.probability) {
            suggestions.push({ label: "Low sensory", key: keyLow, pred: predLow });
        }
    }
    if (routine === "Yes") {
        const keyRoutine = getPredictionKey(sleep, noise, sugar, screen, "No", meal);
        const predRoutine = predictGrid[keyRoutine];
        if (predRoutine && predRoutine.probability < currentPred.probability) {
            suggestions.push({ label: "No routine change", key: keyRoutine, pred: predRoutine });
        }
    }
    if (suggestions.length === 0) {
        wrap.innerHTML = "";
        return;
    }
    wrap.innerHTML = `
        <div class="what-if-title"><strong>What if?</strong> Click to see improved scenarios.</div>
        <div class="what-if-btns">
            ${suggestions.map((s) =>
        `<button type="button" class="what-if-btn" data-key="${s.key}">
                    ${s.label}: ${s.pred.probability}%
                </button>`
    ).join("")}
        </div>
    `;
    wrap.querySelectorAll(".what-if-btn").forEach((btn) => {
        btn.addEventListener("click", () => {
            const key = btn.getAttribute("data-key");
            const pred = predictGrid[key];
            if (pred) {
                const [s, n, sug, scr, r, m] = key.split("|");
                const tierClass = pred.tier === "Low" ? "insight-box" : pred.tier === "Medium" ? "warning-box" : "disclaimer-box";
                const contrib = pred.contributors ? pred.contributors : getTopContributors(s, n, sug, scr, r, m);
                const contribHtml = contrib.length ? `<br><br><strong>Risk factors:</strong> ${contrib.join(", ")}` : "";
                const predBox = predictionResult.querySelector(".insight-box, .warning-box, .disclaimer-box");
                if (predBox) {
                    predBox.outerHTML = `
                        <div class="${tierClass}">
                            <strong>Estimated meltdown likelihood: ${pred.probability}%</strong>
                            <br>Risk level: <strong>${pred.tier}</strong>${contribHtml}
                        </div>
                    `;
                }
                if (contributorChartInstance) {
                    if (contrib.length) {
                        contributorChartInstance.data.labels = contrib;
                        contributorChartInstance.data.datasets[0].data = contrib.map((_, i) => 100 - i * 15);
                        contributorChartInstance.update();
                    } else {
                        contributorChartInstance.destroy();
                        contributorChartInstance = null;
                        const wrap = predictionResult.querySelector(".contributor-chart-wrap");
                        if (wrap) wrap.remove();
                    }
                }
            }
        });
    });
}

predictBtn.addEventListener("click", async function () {
    const sleep = document.getElementById("sleepHours").value;
    const noise = document.getElementById("noiseLevel").value;
    const sugar = document.getElementById("sugarAfter6").value;
    const screen = document.getElementById("screenAfter7").value;
    const routine = document.getElementById("routineChange").value;
    const meal = document.getElementById("mealAfter7").value;

    if (!sleep || !noise || !sugar || !screen || !routine || !meal) {
        predictionResult.innerHTML = '<div class="warning-box">Please fill in all factors above (Sleep, Sensory Environment, Late Sugar, Late Screen, Routine Change, Late Meal) to get a prediction.</div>';
        predictionCard.style.display = "block";
        predictionCard.scrollIntoView({ behavior: "smooth" });
        return;
    }

    try {
        predictBtn.disabled = true;
        await loadPredictGrid();
        const key = getPredictionKey(sleep, noise, sugar, screen, routine, meal);
        const pred = predictGrid[key];
        if (!pred) {
            predictionResult.innerHTML = '<div class="warning-box">Prediction not available for this combination. Try rounding sleep to the nearest half hour.</div>';
        } else {
            const tierClass = pred.tier === "Low" ? "insight-box" : pred.tier === "Medium" ? "warning-box" : "disclaimer-box";
            const contributors = (pred.contributors && pred.contributors.length) ? pred.contributors : getTopContributors(sleep, noise, sugar, screen, routine, meal);
            const contribHtml = contributors.length ? `<br><br><strong>Risk factors today:</strong> ${contributors.join(", ")}` : "";
            const chartHtml = contributors.length ? `
                <div class="contributor-chart-wrap">
                    <div class="contributor-chart-label">Longer bar = stronger influence</div>
                    <canvas id="contributorChart"></canvas>
                </div>
            ` : "";
            predictionResult.innerHTML = `
                <div class="${tierClass}">
                    <strong>Estimated meltdown likelihood: ${pred.probability}%</strong>
                    <br>Risk level: <strong>${pred.tier}</strong>${contribHtml}
                </div>
                ${chartHtml}
                <div id="whatIfSection" class="what-if-section"></div>
            `;
            if (contributors.length) drawContributorChart(contributors);
            renderWhatIf(sleep, noise, sugar, screen, routine, meal, pred);
        }
        predictionCard.style.display = "block";
        predictionCard.scrollIntoView({ behavior: "smooth" });
    } catch (err) {
        predictionResult.innerHTML = '<div class="warning-box">Prediction model could not be loaded. Make sure predict_grid.json is available.</div>';
        predictionCard.style.display = "block";
    } finally {
        predictBtn.disabled = false;
    }
});

clearBtn.addEventListener("click", function () {
    localStorage.removeItem(STORAGE_KEY);
    renderLogs();
    if (triggerChartInstance) {
        triggerChartInstance.destroy();
        triggerChartInstance = null;
    }
    insightsDiv.innerHTML = `
    <div class="card">
      <h2>Trigger Visualization</h2>
      <canvas id="triggerChart"></canvas>
    </div>
    <p class="insights-prompt">Click <strong>Analyze Patterns</strong> to see possible triggers.</p>`;
});

let triggerChartInstance = null;

function drawTriggerChart(patterns) {
    const ctx = document.getElementById("triggerChart");
    if (!ctx) return;

    const withData = patterns.filter(
        (p) => p.whenPresentData || p.whenAbsentData
    );
    if (withData.length === 0) return;

    if (triggerChartInstance) {
        triggerChartInstance.destroy();
        triggerChartInstance = null;
    }

    const labels = withData.map((p) => p.name);
    const whenPresentValues = withData.map((p) =>
        p.whenPresentData ? p.whenPresentData.rate : null
    );
    const whenAbsentValues = withData.map((p) =>
        p.whenAbsentData ? p.whenAbsentData.rate : null
    );

    const createBarGradient = (chart, colorStops, datasetIndex) => {
        const { chartArea } = chart;
        if (!chartArea) return colorStops[0];
        const g = chart.ctx.createLinearGradient(0, chartArea.bottom, 0, chartArea.top);
        colorStops.forEach(([offset, color]) => g.addColorStop(offset, color));
        return g;
    };

    triggerChartInstance = new Chart(ctx, {
        type: "bar",
        data: {
            labels: labels,
            datasets: [
                {
                    label: "When condition present",
                    data: whenPresentValues,
                    backgroundColor: (c) =>
                        createBarGradient(c.chart, [
                            [0, "rgba(251, 207, 232, 1)"],
                            [1, "rgba(249, 168, 212, 1)"],
                        ]),
                    borderColor: "rgba(219, 39, 119, 0.6)",
                    borderWidth: 2,
                    borderRadius: 8,
                },
                {
                    label: "When condition absent",
                    data: whenAbsentValues,
                    backgroundColor: (c) =>
                        createBarGradient(c.chart, [
                            [0, "rgba(187, 247, 208, 1)"],
                            [1, "rgba(134, 239, 172, 1)"],
                        ]),
                    borderColor: "rgba(22, 163, 74, 0.6)",
                    borderWidth: 2,
                    borderRadius: 8,
                },
            ],
        },
        options: {
            animation: {
                duration: 600,
            },
            font: {
                family: "'Plus Jakarta Sans', system-ui, sans-serif",
                size: 13,
            },
            color: "#292524",
            layout: {
                padding: { top: 20, right: 20, bottom: 10, left: 10 },
            },
            scales: {
                y: {
                    beginAtZero: true,
                    max: 100,
                    title: {
                        display: true,
                        text: "Meltdown rate (%)",
                        font: { size: 14, weight: "600" },
                    },
                    grid: { color: "rgba(124, 58, 237, 0.08)" },
                    ticks: { color: "#78716C", font: { size: 12 } },
                    border: { display: false },
                },
                x: {
                    grid: { display: false },
                    ticks: {
                        color: "#78716C",
                        maxRotation: 35,
                        font: { size: 12 },
                    },
                    border: { display: false },
                },
            },
            barPercentage: 0.75,
            categoryPercentage: 0.85,
            plugins: {
                legend: {
                    position: "top",
                    align: "end",
                    labels: {
                        font: { family: "'Plus Jakarta Sans', system-ui, sans-serif", size: 13 },
                        padding: 16,
                        usePointStyle: true,
                        pointStyle: "rectRounded",
                        color: "#78716C",
                    },
                },
            },
        },
        plugins: [
            {
                id: "chartAreaBackground",
                beforeDraw: (chart) => {
                    const { ctx, chartArea } = chart;
                    if (!chartArea) return;
                    const g = ctx.createLinearGradient(0, chartArea.top, 0, chartArea.bottom);
                    g.addColorStop(0, "rgba(248, 247, 252, 0.95)");
                    g.addColorStop(0.5, "rgba(243, 241, 250, 0.95)");
                    g.addColorStop(1, "rgba(237, 234, 248, 0.95)");
                    ctx.save();
                    ctx.fillStyle = g;
                    ctx.fillRect(chartArea.left, chartArea.top, chartArea.right - chartArea.left, chartArea.bottom - chartArea.top);
                    ctx.restore();
                },
            },
            {
                id: "legendBackground",
                beforeDraw: (chart) => {
                    if (chart.legend?.visible) {
                        const ctx = chart.ctx;
                        const { top, left, width, height } = chart.legend;
                        const pad = 12;
                        ctx.save();
                        ctx.fillStyle = "rgba(237, 233, 254, 0.7)";
                        ctx.beginPath();
                        if (typeof ctx.roundRect === "function") {
                            ctx.roundRect(left - pad, top - 8, width + pad * 2, height + 16, 10);
                        } else {
                            ctx.rect(left - pad, top - 8, width + pad * 2, height + 16);
                        }
                        ctx.fill();
                        ctx.restore();
                    }
                },
            },
        ],
    });
}

if (getLogs().length === 0) {
    saveLogs(getDefaultSampleLogs());
}
renderLogs();

/* ============================================
   Help Chatbot (OpenAI)
   ============================================ */

const CHATBOT_SYSTEM_PROMPT = `You are a warm, supportive help assistant for the Meltdown Pattern Predictor app. This app helps caregivers of autistic children track daily factors (sleep, sensory environment, sugar, screen time, routine changes, meals) and identify patterns that may correlate with meltdowns.

Your role:
- Explain how to use the app (logging, analyzing, reading the chart)
- Explain what each field means (Sensory Environment = noise level, Late Sugar = after 6pm, etc.)
- Help users understand the chart and insights (meltdown rate when condition present vs absent)
- Offer general, supportive tips for common triggers—never medical or behavioral diagnosis
- Be concise but caring; caregivers are often exhausted

Rules:
- Never diagnose autism or prescribe treatments
- Always recommend consulting healthcare providers for medical/behavioral advice
- Keep answers practical and actionable
- Use simple language`;

function getInsightsContext() {
    const logs = getLogs();
    if (logs.length < 3) return null;
    try {
        const patterns = [
            { name: "Sleep below 7 hours", fn: (l) => Number(l.sleepHours) < 7 },
            { name: "High sensory environment", fn: (l) => l.noiseLevel === "High" },
            { name: "Late sugar intake", fn: (l) => l.sugarAfter6 === "Yes" },
            { name: "Late screen exposure", fn: (l) => l.screenAfter7 === "Yes" },
            { name: "Routine change", fn: (l) => l.routineChange === "Yes" },
            { name: "Late meal", fn: (l) => l.mealAfter7 === "Yes" },
        ];
        const withRates = patterns.map((p) => ({
            ...p,
            present: meltdownRate(p.fn),
            absent: meltdownRate((l) => !p.fn(l)),
        }));
        const top = withRates
            .filter((p) => p.present)
            .map((p) => ({
                name: p.name,
                lift: p.present.rate - (p.absent ? p.absent.rate : 0),
            }))
            .sort((a, b) => b.lift - a.lift)
            .slice(0, 3);
        const overall = meltdownRate(() => true);
        return {
            logCount: logs.length,
            overallRate: overall.rate.toFixed(1),
            topTriggers: top.map((t) => t.name),
        };
    } catch {
        return null;
    }
}

const chatbotFab = document.getElementById("chatbotFab");
const chatbotPanel = document.getElementById("chatbotPanel");
const chatbotClose = document.getElementById("chatbotClose");
const chatbotMessages = document.getElementById("chatbotMessages");
const chatbotQuickPrompts = document.getElementById("chatbotQuickPrompts");
const chatbotInput = document.getElementById("chatbotInput");
const chatbotSend = document.getElementById("chatbotSend");

let chatHistory = [];
let chatbotWelcomed = false;

function appendMessage(role, content, isTyping = false) {
    const div = document.createElement("div");
    div.className = `chatbot-msg ${role}${isTyping ? " typing" : ""}`;
    if (isTyping) {
        div.innerHTML = '<div class="chatbot-typing-dots"><span></span><span></span><span></span></div>';
    } else if (role === "bot" && typeof marked !== "undefined") {
        div.innerHTML = marked.parse(content || "");
    } else {
        div.textContent = content;
    }
    chatbotMessages.appendChild(div);
    chatbotMessages.scrollTop = chatbotMessages.scrollHeight;
}

function removeTypingIndicator() {
    const typing = chatbotMessages.querySelector(".chatbot-msg.typing");
    if (typing) typing.remove();
}

function showWelcome() {
    if (chatbotWelcomed) return;
    chatbotWelcomed = true;
    const ctx = getInsightsContext();
    let msg = "Hi! I'm here to help. Ask me about using the app, understanding your insights, or general tips for supporting your child. I'm not a substitute for professional advice—always consult healthcare providers for medical questions.";
    if (ctx) {
        msg += `\n\n(You have ${ctx.logCount} logs with an overall meltdown rate of ${ctx.overallRate}%. Your top possible triggers: ${ctx.topTriggers.join(", ")})`;
    }
    appendMessage("bot", msg);
}

chatbotFab.addEventListener("click", () => {
    chatbotPanel.classList.add("open");
    chatbotPanel.setAttribute("aria-hidden", "false");
    showWelcome();
});

chatbotClose.addEventListener("click", () => {
    chatbotPanel.classList.remove("open");
    chatbotPanel.setAttribute("aria-hidden", "true");
});

async function sendToOpenAI(userMsg) {
    const apiKey = typeof OPENAI_API_KEY !== "undefined" ? OPENAI_API_KEY : "";
    const useProxy = !apiKey || apiKey === "your-openai-api-key-here";

    const ctx = getInsightsContext();
    const contextStr = ctx
        ? `\n\nCurrent user data: ${ctx.logCount} logs, overall meltdown rate ${ctx.overallRate}%, top triggers: ${ctx.topTriggers.join(", ")}.`
        : "";

    chatHistory.push({ role: "user", content: userMsg });
    const messages = [
        { role: "system", content: CHATBOT_SYSTEM_PROMPT + contextStr },
        ...chatHistory.slice(-12).map((m) => ({ role: m.role === "assistant" ? "assistant" : m.role, content: m.content })),
    ];

    appendMessage("bot", "", true);

    try {
        let res;
        if (useProxy) {
            res = await fetch("/api/chat", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ messages }),
            });
        } else {
            res = await fetch("https://api.openai.com/v1/chat/completions", {
                method: "POST",
                headers: {
                    "Content-Type": "application/json",
                    Authorization: `Bearer ${apiKey}`,
                },
                body: JSON.stringify({
                    model: "gpt-4o-mini",
                    messages,
                    max_tokens: 500,
                    temperature: 0.7,
                }),
            });
        }

        removeTypingIndicator();

        if (!res.ok) {
            const err = await res.json().catch(() => ({}));
            const msg = useProxy && res.status === 500
                ? "AI help isn't configured. Add OPENAI_API_KEY in Vercel environment variables."
                : "Sorry, I couldn't get a response. Please try again.";
            appendMessage("bot", msg);
            return;
        }

        let reply;
        if (useProxy) {
            const data = await res.json();
            reply = data.reply || "I'm not sure how to respond.";
        } else {
            const data = await res.json();
            reply = data.choices?.[0]?.message?.content?.trim() || "I'm not sure how to respond.";
        }
        chatHistory.push({ role: "assistant", content: reply });
        appendMessage("bot", reply);
    } catch (err) {
        removeTypingIndicator();
        appendMessage("bot", "Something went wrong. Please check your internet connection and try again.");
    }
}

function sendChatMessage() {
    const text = chatbotInput.value.trim();
    if (!text) return;
    appendMessage("user", text);
    chatbotInput.value = "";
    chatbotInput.style.height = "auto";
    chatbotSend.disabled = true;
    sendToOpenAI(text).finally(() => (chatbotSend.disabled = false));
}

chatbotSend.addEventListener("click", sendChatMessage);

chatbotInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
        e.preventDefault();
        sendChatMessage();
    }
});

chatbotQuickPrompts.querySelectorAll(".chatbot-quick-btn").forEach((btn) => {
    btn.addEventListener("click", () => {
        const prompt = btn.getAttribute("data-prompt");
        if (prompt) {
            appendMessage("user", prompt);
            chatbotQuickPrompts.style.display = "none";
            chatbotSend.disabled = true;
            sendToOpenAI(prompt).finally(() => (chatbotSend.disabled = false));
        }
    });
});

chatbotInput.addEventListener("input", function () {
    this.style.height = "auto";
    this.style.height = Math.min(this.scrollHeight, 120) + "px";
});