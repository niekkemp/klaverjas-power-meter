const errorMessage = document.getElementById('form-error');
const results = document.querySelector('.results');
const loading = document.querySelector('.loading')

document.getElementById('analyze-form').addEventListener('submit', async function (event) {
    // Override default form submission
    event.preventDefault();

    // Process form data
    const form = event.target;
    const formData = new FormData(form);
    if (formData.get("ridge")) {
        formData.set("method", "ridge");
    } else {
        formData.set("method", "mle")
    }
    const alpha = formData.get("alpha");

    // Close all info texts
    document.querySelectorAll('.info-text').forEach(el => {
        el.style.display = 'none';
    });
    document.querySelectorAll('.info-btn').forEach(btn => {
        btn.textContent = 'ℹ️';
    });

    // Show loading icon
    loading.style.display = 'block';

    // Ensure previous results or errors are hidden
    results.style.display = "none";
    results.classList.remove('active')
    errorMessage.textContent = "";

    // Do api request
    try {
        // Api request
        const response = await fetch('http://127.0.0.1:5000/analyze', {
            method: 'POST',
            body: formData
        });
        const result = await response.json();

        // Done loading
        loading.style.display = 'none';

        if (!response.ok) {
            // Display response error
            errorMessage.textContent = result.error;
        } else { // Handle succesfull analysis result
            // Log response for debugging
            console.log(result);

            // Display the results section
            results.style.display = "block";

            // Update result sections
            updateRanking(result.ranking_table);
            updateSkillComparison(result.statements, alpha);
            updatePairwiseTable(result.players, result.pairwise_table);

            // Let all fade in
            requestAnimationFrame(() => {
                requestAnimationFrame(() => {
                    results.classList.add('active')
                });
            });


        }
    } catch (error) { // Handle errors in api request or response handling
        errorMessage.textContent = `Something went wrong during the api request or response handling ${error}`;

        // Ensure loading and results are hidden in case of error
        loading.style.display = 'none';
        results.style.display = "none";
        results.classList.remove('active')
    }
});

function updateRanking(ranking_table) {
    const tbody = document.querySelector(".ranking tbody");
    tbody.replaceChildren();

    ranking_table.forEach(({ player, skill, win_rate, sig }) => {
        const tr = document.createElement("tr");

        tr.innerHTML = `
            <td>${player}</td>
            <td>${(skill >= 0)
                ? `+${skill.toFixed(2)}` : skill.toFixed(2)
            }</td>
            <td>${(win_rate * 100).toFixed(0)}%</td>
            <td>${sig
                ? (skill > 0 ? '🟢 Better' : '🔴 Worse')
                : '⚪️ Neutral'
            }</td>
        `;

        tbody.appendChild(tr);
    });
}

function updateSkillComparison(statements, alpha) {
    const alpha_holder = document.getElementById("alpha-holder");
    alpha_holder.textContent = alpha * 100;

    const any_sig = document.getElementById("any-sig");
    const no_sig = document.getElementById("no-sig");

    any_sig.style.display = "none";
    no_sig.style.display = "none";

    const ul = document.querySelector(".pairwise ul");
    ul.replaceChildren();

    if (statements.length > 0) {
        any_sig.style.display = "block";
        for (let i = 0; i < statements.length; i++) {
            const li = document.createElement("li");
            li.textContent = statements[i];

            ul.appendChild(li);
        }
    }
    else {
        no_sig.style.display = "block";
    }
}

function updatePairwiseTable(players, pairwise_table) {
    const table = document.querySelector("#pairwise-table table");
    table.replaceChildren(); // Clear old content

    // Build <thead>
    const thead = document.createElement("thead");
    const headerRow = document.createElement("tr");

    const vsTh = document.createElement("th");
    vsTh.textContent = "vs"; // Top-left label
    headerRow.appendChild(vsTh);

    players.forEach(player => {
        const th = document.createElement("th");
        th.textContent = player;
        headerRow.appendChild(th);
    });

    thead.appendChild(headerRow);
    table.appendChild(thead);

    // Build <tbody>
    const tbody = document.createElement("tbody");

    for (let i = 0; i < players.length; i++) {
        const row = document.createElement("tr");

        const rowHeader = document.createElement("th");
        rowHeader.textContent = players[i];
        row.appendChild(rowHeader);

        for (let j = 0; j < players.length; j++) {
            const cell = document.createElement("td");
            const entry = pairwise_table[i][j];

            if (entry == null) {
                cell.textContent = "—";
                cell.className = "diag";
            } else {
                const { diff, std, sig } = entry;
                cell.textContent = `${diff >= 0 ? '+' : ''}${diff.toFixed(2)} (±${std.toFixed(2)})`;

                if (sig) {
                    cell.textContent += diff > 0 ? " 🟢" : " 🔴";
                    cell.className = diff > 0 ? "sig-pos" : "sig-neg";
                } else {
                    cell.textContent += " ⚪️";
                }
            }

            row.appendChild(cell);
        }

        tbody.appendChild(row);
    }

    table.appendChild(tbody);
}


