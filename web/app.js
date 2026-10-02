/**
 * Automated Decision-Making Core — Frontend Application Controller
 * Handles live ranking execution, microsecond latency waterfall rendering,
 * Monte Carlo Bandit A/B simulations, and online feedback ingestion.
 */

document.addEventListener('DOMContentLoaded', () => {
    // DOM Elements
    const personaSelect = document.getElementById('personaSelect');
    const banditPolicySelect = document.getElementById('banditPolicySelect');
    const alphaSlider = document.getElementById('alphaSlider');
    const alphaVal = document.getElementById('alphaVal');
    const lambdaSlider = document.getElementById('lambdaSlider');
    const lambdaVal = document.getElementById('lambdaVal');
    const modelSelect = document.getElementById('modelSelect');
    const topKInput = document.getElementById('topKInput');
    const runDecisionBtn = document.getElementById('runDecisionBtn');
    const runSimBtn = document.getElementById('runSimBtn');

    const totalLatencyVal = document.getElementById('totalLatencyVal');
    const timeRetrieval = document.getElementById('timeRetrieval');
    const timeEnrich = document.getElementById('timeEnrich');
    const timeRank = document.getElementById('timeRank');
    const timeBandit = document.getElementById('timeBandit');
    const timeRerank = document.getElementById('timeRerank');
    const rankingTableBody = document.getElementById('rankingTableBody');
    const armsContainer = document.getElementById('armsContainer');
    const cacheHitRateBadge = document.getElementById('cacheHitRateBadge');

    let banditChart = null;

    // Slider inputs
    alphaSlider.addEventListener('input', (e) => {
        alphaVal.textContent = parseFloat(e.target.value).toFixed(2);
    });

    lambdaSlider.addEventListener('input', (e) => {
        lambdaVal.textContent = parseFloat(e.target.value).toFixed(2);
    });

    // Execute Multi-Stage Ranking
    async function executeRanking() {
        runDecisionBtn.disabled = true;
        runDecisionBtn.style.opacity = '0.7';

        const payload = {
            user_context: {
                user_id: personaSelect.value,
                session_id: "sess_" + Math.floor(Math.random() * 10000),
                past_interacted_categories: getPersonaCategories(personaSelect.value)
            },
            config: {
                top_k: parseInt(topKInput.value) || 6,
                exploration_policy: banditPolicySelect.value,
                exploration_factor_alpha: parseFloat(alphaSlider.value),
                diversity_penalty_lambda: parseFloat(lambdaSlider.value),
                ranker_model_type: modelSelect.value
            }
        };

        try {
            const response = await fetch('/api/v1/rank', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            });

            const data = await response.json();
            renderWaterfall(data.latency_breakdown);
            renderRankedTable(data.ranked_items, personaSelect.value);
            fetchBanditStats();
        } catch (error) {
            console.error("Ranking API Error:", error);
        } finally {
            runDecisionBtn.disabled = false;
            runDecisionBtn.style.opacity = '1';
        }
    }

    function getPersonaCategories(persona) {
        if (persona === 'usr_tech_enthusiast') return ['Electronics & Gaming', 'Smart Home'];
        if (persona === 'usr_fashion_shopper') return ['Apparel & Fashion'];
        if (persona === 'usr_vip_power') return ['Electronics & Gaming', 'Wearables', 'Apparel & Fashion'];
        return [];
    }

    // Render Stage Latency Waterfall
    function renderWaterfall(latency) {
        if (!latency) return;
        totalLatencyVal.textContent = latency.total_pipeline_ms.toFixed(2);
        timeRetrieval.textContent = latency.candidate_retrieval_ms.toFixed(2);
        timeEnrich.textContent = latency.feature_enrichment_ms.toFixed(2);
        timeRank.textContent = latency.heavy_ranking_ms.toFixed(2);
        timeBandit.textContent = latency.contextual_bandit_ms.toFixed(2);
        timeRerank.textContent = latency.reranking_diversity_ms.toFixed(2);
    }

    // Render Ranked Items
    function renderRankedTable(items, userId) {
        if (!items || items.length === 0) {
            rankingTableBody.innerHTML = '<tr><td colspan="6" class="placeholder-row">No candidates returned.</td></tr>';
            return;
        }

        rankingTableBody.innerHTML = items.map(item => `
            <tr>
                <td class="score-num">#${item.rank_position}</td>
                <td>
                    <div style="font-weight: 700; color: #fff;">${item.title || item.item_id}</div>
                    <div style="font-size: 0.72rem; color: #9ca3af;">
                        ${item.category} • $${item.price.toFixed(2)}
                        ${item.was_explored ? '<span class="item-badge-explored">⚡ EXPLORED</span>' : ''}
                    </div>
                </td>
                <td class="score-num" style="color: #60a5fa;">${item.ml_rank_score.toFixed(4)}</td>
                <td class="score-num" style="color: #fbbf24;">+${item.bandit_exploration_bonus.toFixed(4)}</td>
                <td class="score-num" style="color: #34d399; font-size: 0.95rem;">${item.final_score.toFixed(4)}</td>
                <td>
                    <div class="feedback-btn-group">
                        <button class="fb-btn" onclick="window.sendFeedback('${userId}', '${item.item_id}', 'click', 1.0)">+Click</button>
                        <button class="fb-btn buy" onclick="window.sendFeedback('${userId}', '${item.item_id}', 'conversion', 5.0)">+Buy</button>
                    </div>
                </td>
            </tr>
        `).join('');
    }

    // Feedback Ingestion
    window.sendFeedback = async function(userId, itemId, eventType, reward) {
        try {
            const res = await fetch('/api/v1/feedback', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    user_id: userId,
                    item_id: itemId,
                    event_type: eventType,
                    reward_value: reward
                })
            });
            const data = await res.json();
            fetchBanditStats();
        } catch (e) {
            console.error('Feedback error:', e);
        }
    };

    // Fetch Bandit stats
    async function fetchBanditStats() {
        try {
            const res = await fetch('/api/v1/bandit-stats');
            const data = await res.json();
            if (data.arms_summary) {
                armsContainer.innerHTML = data.arms_summary.map(a => `
                    <div class="arm-card-mini">
                        <span class="arm-name">${a.arm_id.replace('item_', '')}</span>
                        <span class="arm-metrics">${a.pull_count} pulls | Rew: ${a.total_reward}</span>
                    </div>
                `).join('');
            }
        } catch (e) {
            console.error(e);
        }
    }

    // Run Monte Carlo Simulation & Render Chart.js
    async function runSimulation() {
        runSimBtn.disabled = true;
        runSimBtn.textContent = "Running Simulation...";

        try {
            const res = await fetch('/api/v1/simulate?steps=100', { method: 'POST' });
            const sim = await res.json();
            renderBanditChart(sim.results);
        } catch (e) {
            console.error(e);
        } finally {
            runSimBtn.disabled = false;
            runSimBtn.textContent = "▶ Run 100-Step Simulation";
        }
    }

    function renderBanditChart(results) {
        const ctx = document.getElementById('banditChart').getContext('2d');
        const labels = Array.from({ length: 100 }, (_, i) => i + 1);

        if (banditChart) {
            banditChart.destroy();
        }

        banditChart = new Chart(ctx, {
            type: 'line',
            data: {
                labels: labels,
                datasets: [
                    {
                        label: 'LinUCB (Ours)',
                        data: results.linucb.cumulative_rewards,
                        borderColor: '#3b82f6',
                        backgroundColor: 'rgba(59, 130, 246, 0.1)',
                        tension: 0.2,
                        borderWidth: 2.5,
                        pointRadius: 0
                    },
                    {
                        label: 'Thompson Sampling',
                        data: results.thompson_sampling.cumulative_rewards,
                        borderColor: '#10b981',
                        backgroundColor: 'transparent',
                        tension: 0.2,
                        borderWidth: 2,
                        pointRadius: 0
                    },
                    {
                        label: 'ε-Greedy (20%)',
                        data: results.epsilon_greedy.cumulative_rewards,
                        borderColor: '#f59e0b',
                        backgroundColor: 'transparent',
                        tension: 0.2,
                        borderWidth: 1.5,
                        pointRadius: 0
                    },
                    {
                        label: 'Random Baseline',
                        data: results.random.cumulative_rewards,
                        borderColor: '#6b7280',
                        borderDash: [4, 4],
                        backgroundColor: 'transparent',
                        borderWidth: 1.5,
                        pointRadius: 0
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        labels: {
                            color: '#9ca3af',
                            font: { size: 10, family: 'Plus Jakarta Sans' }
                        }
                    },
                    tooltip: {
                        mode: 'index',
                        intersect: false
                    }
                },
                scales: {
                    x: {
                        grid: { color: 'rgba(255,255,255,0.05)' },
                        ticks: { color: '#6b7280', font: { size: 9 } }
                    },
                    y: {
                        grid: { color: 'rgba(255,255,255,0.05)' },
                        ticks: { color: '#6b7280', font: { size: 9 } },
                        title: { display: true, text: 'Cumulative Reward', color: '#9ca3af', font: { size: 10 } }
                    }
                }
            }
        });
    }

    // Event Listeners
    runDecisionBtn.addEventListener('click', executeRanking);
    runSimBtn.addEventListener('click', runSimulation);

    // Initial warm load
    executeRanking();
    runSimulation();
});
