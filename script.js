const STORAGE_KEY = "meltdownLogs";

const logForm = document.getElementById("logForm");
const logsTableBody = document.querySelector("#logsTable tbody");
const emptyState = document.getElementById("emptyState");
const insightsDiv = document.getElementById("insights");
const sampleDataBtn = document.getElementById("sampleDataBtn");
const analyzeBtn = document.getElementById("analyzeBtn");
const clearBtn = document.getElementById("clearBtn");

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

    logs.forEach((log) => {
        const row = document.createElement("tr");
        row.innerHTML = `
      <td>${log.date}</td>
      <td>${log.sleepHours}</td>
      <td>${log.noiseLevel}</td>
      <td>${log.sugarAfter6}</td>
      <td>${log.screenAfter7}</td>
      <td>${log.routineChange}</td>
      <td>${log.mealAfter7}</td>
      <td>${log.meltdownOccurred}</td>
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
        insightsDiv.innerHTML = `
      <div class="warning-box">
        Please add at least 3 logs so the analysis makes sense.
      </div>
    `;
        return;
    }

    const patterns = [
        {
            name: "Sleep below 7 hours",
            data: meltdownRate((log) => Number(log.sleepHours) < 7),
            explanation:
                "Lower sleep may increase stress, irritability, or sensitivity to sensory input.",
        },
        {
            name: "High sensory environment",
            data: meltdownRate((log) => log.noiseLevel === "High"),
            explanation:
                "Loud environments may contribute to sensory overload and emotional distress.",
        },
        {
            name: "Late sugar intake",
            data: meltdownRate((log) => log.sugarAfter6 === "Yes"),
            explanation:
                "Late sugar intake may be linked to energy spikes and difficulty regulating emotions.",
        },
        {
            name: "Late screen exposure",
            data: meltdownRate((log) => log.screenAfter7 === "Yes"),
            explanation:
                "Late screen exposure may interfere with sleep preparation and emotional regulation.",
        },
        {
            name: "Routine change",
            data: meltdownRate((log) => log.routineChange === "Yes"),
            explanation:
                "Unexpected routine changes may increase anxiety and difficulty with transitions.",
        },
        {
            name: "Late meal",
            data: meltdownRate((log) => log.mealAfter7 === "Yes"),
            explanation:
                "Late meals may be connected to discomfort, disrupted routine, or sleep issues.",
        },
    ];

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
        if (!pattern.data) return;

        html += `
      <div class="insight-box">
        <strong>${pattern.name}</strong><br>
        Meltdown rate under this condition:
        <strong>${pattern.data.rate.toFixed(1)}%</strong>
        (${pattern.data.meltdowns} out of ${pattern.data.total} logs)<br><br>
        <em>${pattern.explanation}</em>
      </div>
    `;
    });

    const strongTriggers = patterns
        .filter((pattern) => pattern.data && pattern.data.total >= 1)
        .sort((a, b) => b.data.rate - a.data.rate)
        .slice(0, 3);

    if (strongTriggers.length > 0) {
        html += `
      <div class="warning-box">
        <strong>Top possible triggers:</strong><br>
        ${strongTriggers
                .map(
                    (trigger, index) =>
                        `${index + 1}. ${trigger.name} (${trigger.data.rate.toFixed(1)}% meltdown rate)`
                )
                .join("<br>")}
      </div>
    `;
    }

    html += `
    <div class="warning-box">
      <strong>Important:</strong> This tool does not diagnose autism or predict behavior with certainty.
      It is designed to help caregivers notice patterns that may support better routine planning and understanding.
    </div>
  `;

    insightsDiv.innerHTML = html;
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
    const sampleLogs = [
        {
            date: "2026-03-01",
            sleepHours: 8,
            noiseLevel: "Low",
            sugarAfter6: "No",
            screenAfter7: "No",
            routineChange: "No",
            mealAfter7: "No",
            meltdownOccurred: "No",
        },
        {
            date: "2026-03-02",
            sleepHours: 6,
            noiseLevel: "High",
            sugarAfter6: "Yes",
            screenAfter7: "Yes",
            routineChange: "Yes",
            mealAfter7: "Yes",
            meltdownOccurred: "Yes",
        },
        {
            date: "2026-03-03",
            sleepHours: 6.5,
            noiseLevel: "High",
            sugarAfter6: "No",
            screenAfter7: "Yes",
            routineChange: "No",
            mealAfter7: "Yes",
            meltdownOccurred: "Yes",
        },
        {
            date: "2026-03-04",
            sleepHours: 7.5,
            noiseLevel: "Medium",
            sugarAfter6: "No",
            screenAfter7: "No",
            routineChange: "No",
            mealAfter7: "No",
            meltdownOccurred: "No",
        },
        {
            date: "2026-03-05",
            sleepHours: 5.5,
            noiseLevel: "High",
            sugarAfter6: "Yes",
            screenAfter7: "Yes",
            routineChange: "Yes",
            mealAfter7: "Yes",
            meltdownOccurred: "Yes",
        },
    ];

    saveLogs(sampleLogs);
    renderLogs();
    insightsDiv.innerHTML = `
    <div class="insight-box">
      Sample data loaded. Now click <strong>Analyze Patterns</strong>.
    </div>
  `;
});

analyzeBtn.addEventListener("click", analyzePatterns);

clearBtn.addEventListener("click", function () {
    localStorage.removeItem(STORAGE_KEY);
    renderLogs();
    insightsDiv.innerHTML = `Click <strong>Analyze Patterns</strong> to see possible triggers.`;
});

renderLogs();
function drawTriggerChart(patterns) {

    const ctx = document.getElementById('triggerChart');

    if (!ctx) return;

    const labels = patterns.map(p => p.name);
    const values = patterns.map(p => p.rate);

    new Chart(ctx, {
        type: 'bar',
        data: {
            labels: labels,
            datasets: [{
                label: 'Meltdown Rate (%)',
                data: values,
                backgroundColor: [
                    '#ff6384',
                    '#36a2eb',
                    '#ffcd56',
                    '#4bc0c0',
                    '#9966ff',
                    '#ff9f40'
                ]
            }]
        },
        options: {
            scales: {
                y: {
                    beginAtZero: true,
                    max: 100
                }
            }
        }
    });

}