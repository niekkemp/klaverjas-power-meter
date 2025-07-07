const errorMessage = document.getElementById('form-error')
const results = document.querySelector('.results')

document.getElementById('analyze-form').addEventListener('submit', async function (event) {
    event.preventDefault();

    const form = event.target;
    const formData = new FormData(form);

    const alpha = formData.get("alpha");
    // console.log(alpha);

    try {
        const response = await fetch('http://127.0.0.1:5000/analyze', {
            method: 'POST',
            body: formData
        });

        const result = await response.json();

        errorMessage.textContent = "";
        results.style.display = "none";

        if (!response.ok) {
            errorMessage.textContent = result.error;
        } else {
            // Handle succesfull analysis result
            console.log(result);
            results.style.display = "block";
            updateScoreboard(result.players, result.skill_scores, result.win_ratios);
            updateInterpretation(result.significant_statements, alpha);
            updateStatistical(result.players, result.pairwise_table);
        }
    } catch (error) {
        console.error('Unexpected error:', error);
        alert('Something unexpected went wrong during analysis.');
    }
});

function updateScoreboard(players, skill_scores, win_ratios) {
    const tbody = document.querySelector(".scoreboard tbody");
    tbody.replaceChildren();

    for (let i = 0; i < players.length; i++) {
        const th = document.createElement("th");
        th.textContent = players[i];

        const td = document.createElement("td");
        td.textContent = `${skill_scores[i].toFixed(2)} (${(100 * win_ratios[i]).toFixed(1)}%)`;
        td.style.textAlign = "right";

        const tr = document.createElement("tr");
        tr.appendChild(th);
        tr.appendChild(td);

        tbody.appendChild(tr);
    }
}

function updateInterpretation(significant_statements, alpha) {
    const alpha_holder = document.getElementById("alpha-holder");
    alpha_holder.textContent = alpha * 100;

    const sig_description = document.getElementById("sig-description");
    const unsig_description = document.getElementById("unsig-description");

    sig_description.style.display = "none";
    unsig_description.style.display = "none";

    const ul = document.querySelector(".interpretation ul");
    ul.replaceChildren();

    if (significant_statements.length > 0) {
        sig_description.style.display = "block";
        for (let i = 0; i < significant_statements.length; i++) {
            const li = document.createElement("li");
            li.textContent = significant_statements[i];

            ul.appendChild(li);
        }
    }
    else {
        unsig_description.style.display = "block";
    }
}

function updateStatistical(players, pairwise_table) {
    const table = document.querySelector(".statistical table");
    table.replaceChildren();

    const thead = document.createElement("thead");
    const tr = document.createElement("tr");
    const th = document.createElement("th");
    tr.appendChild(th);
    for (let i = 0; i < players.length; i++) {
        const th = document.createElement("th");
        th.textContent = players[i];
        tr.appendChild(th);
    }
    thead.appendChild(tr);
    table.appendChild(thead);

    const tbody = document.createElement("tbody");

    for (let i = 0; i < pairwise_table.length; i++) {
        const tr = document.createElement("tr");
        const th = document.createElement("th");
        th.textContent = players[i];
        tr.appendChild(th);
        for (let j = 0; j < pairwise_table[i].length; j++) {
            const td = document.createElement("td")
            if (pairwise_table[i][j] == null) {
                td.className = "diag";
            } else {
                td.textContent = `${pairwise_table[i][j].diff.toFixed(2)} (${pairwise_table[i][j].std.toFixed(2)})`
                if (pairwise_table[i][j].significant) {
                    if (pairwise_table[i][j].diff > 0) {
                        td.className = "sig-pos";
                    } else {
                        td.className = "sig-neg";
                    }
                }
            }
            tr.append(td)
        }
        tbody.appendChild(tr);
    }
    table.appendChild(tbody);
}