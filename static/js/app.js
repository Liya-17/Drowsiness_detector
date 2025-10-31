// Advanced Drowsiness Detection System - Frontend JavaScript

// Initialize Socket.IO connection
const socket = io();

// State
let isRunning = false;
let isCalibrating = false;
let fatigueChart = null;
let metricsChart = null;

// Chart data buffers
const chartDataLimit = 90; // 30 seconds at 3 samples/sec
const fatigueData = [];
const earData = [];
const perclosData = [];
const timestamps = [];

// DOM Elements
const startBtn = document.getElementById('startBtn');
const stopBtn = document.getElementById('stopBtn');
const calibrateBtn = document.getElementById('calibrateBtn');
const statusDot = document.getElementById('statusDot');
const statusText = document.getElementById('statusText');
const videoFeed = document.getElementById('videoFeed');
const videoOverlay = document.getElementById('videoOverlay');
const calibrationModal = document.getElementById('calibrationModal');
const cancelCalibrationBtn = document.getElementById('cancelCalibrationBtn');
const alertToast = document.getElementById('alertToast');

// Event Listeners
startBtn.addEventListener('click', startDetection);
stopBtn.addEventListener('click', stopDetection);
calibrateBtn.addEventListener('click', startCalibration);
cancelCalibrationBtn.addEventListener('click', cancelCalibration);

// Socket.IO event handlers
socket.on('connect', () => {
    console.log('Connected to server');
});

socket.on('metrics_update', (metrics) => {
    updateMetrics(metrics);
    updateCharts(metrics);
});

socket.on('alert', (alert) => {
    showAlert(alert);
});

socket.on('calibration_status', (status) => {
    handleCalibrationStatus(status);
});

// Initialize charts
function initCharts() {
    // Fatigue Trend Chart
    const fatigueCtx = document.getElementById('fatigueChart').getContext('2d');
    fatigueChart = new Chart(fatigueCtx, {
        type: 'line',
        data: {
            labels: [],
            datasets: [{
                label: 'Fatigue Score',
                data: [],
                borderColor: '#ef4444',
                backgroundColor: 'rgba(239, 68, 68, 0.1)',
                borderWidth: 2,
                tension: 0.4,
                fill: true,
                pointRadius: 0
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                y: {
                    beginAtZero: true,
                    max: 100,
                    ticks: { color: '#94a3b8' },
                    grid: { color: '#475569' }
                },
                x: {
                    ticks: { color: '#94a3b8', maxTicksLimit: 10 },
                    grid: { color: '#475569' }
                }
            },
            plugins: {
                legend: {
                    display: false
                }
            },
            animation: {
                duration: 0
            }
        }
    });

    // EAR & PERCLOS Chart
    const metricsCtx = document.getElementById('metricsChart').getContext('2d');
    metricsChart = new Chart(metricsCtx, {
        type: 'line',
        data: {
            labels: [],
            datasets: [
                {
                    label: 'EAR',
                    data: [],
                    borderColor: '#3b82f6',
                    backgroundColor: 'rgba(59, 130, 246, 0.1)',
                    borderWidth: 2,
                    tension: 0.4,
                    yAxisID: 'y',
                    pointRadius: 0
                },
                {
                    label: 'PERCLOS',
                    data: [],
                    borderColor: '#f59e0b',
                    backgroundColor: 'rgba(245, 158, 11, 0.1)',
                    borderWidth: 2,
                    tension: 0.4,
                    yAxisID: 'y1',
                    pointRadius: 0
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                y: {
                    type: 'linear',
                    position: 'left',
                    beginAtZero: true,
                    max: 0.5,
                    ticks: { color: '#94a3b8' },
                    grid: { color: '#475569' },
                    title: {
                        display: true,
                        text: 'EAR',
                        color: '#3b82f6'
                    }
                },
                y1: {
                    type: 'linear',
                    position: 'right',
                    beginAtZero: true,
                    max: 100,
                    ticks: { color: '#94a3b8' },
                    grid: { display: false },
                    title: {
                        display: true,
                        text: 'PERCLOS (%)',
                        color: '#f59e0b'
                    }
                },
                x: {
                    ticks: { color: '#94a3b8', maxTicksLimit: 10 },
                    grid: { color: '#475569' }
                }
            },
            plugins: {
                legend: {
                    labels: { color: '#f1f5f9' }
                }
            },
            animation: {
                duration: 0
            }
        }
    });
}

// Start detection
async function startDetection() {
    try {
        const response = await fetch('/api/start', { method: 'POST' });
        const data = await response.json();

        if (data.status === 'success') {
            isRunning = true;
            startBtn.disabled = true;
            stopBtn.disabled = false;
            calibrateBtn.disabled = true;
            statusDot.classList.add('active');
            statusText.textContent = 'Active - Monitoring';
            videoOverlay.classList.add('hidden');
            videoFeed.src = '/video_feed?' + new Date().getTime();

            // Start statistics polling
            startStatisticsPolling();
        } else if (data.message.includes('Calibration required')) {
            alert('Please calibrate the system first before starting detection.');
            startCalibration();
        } else {
            alert(data.message);
        }
    } catch (error) {
        console.error('Error starting detection:', error);
        alert('Failed to start detection');
    }
}

// Stop detection
async function stopDetection() {
    try {
        const response = await fetch('/api/stop', { method: 'POST' });
        const data = await response.json();

        if (data.status === 'success') {
            isRunning = false;
            startBtn.disabled = false;
            stopBtn.disabled = true;
            calibrateBtn.disabled = false;
            statusDot.classList.remove('active');
            statusText.textContent = 'Idle';
            videoOverlay.classList.remove('hidden');
            videoFeed.src = '';

            // Stop statistics polling
            stopStatisticsPolling();
        }
    } catch (error) {
        console.error('Error stopping detection:', error);
    }
}

// Start calibration
async function startCalibration() {
    try {
        const response = await fetch('/api/calibrate', { method: 'POST' });
        const data = await response.json();

        if (data.status === 'success') {
            isCalibrating = true;
            calibrationModal.classList.add('show');
            document.getElementById('calibrationProgress').style.width = '0%';
            document.getElementById('calibrationPercent').textContent = '0%';
        } else {
            alert(data.message);
        }
    } catch (error) {
        console.error('Error starting calibration:', error);
    }
}

// Cancel calibration
async function cancelCalibration() {
    try {
        await fetch('/api/cancel_calibration', { method: 'POST' });
        isCalibrating = false;
        calibrationModal.classList.remove('show');
    } catch (error) {
        console.error('Error cancelling calibration:', error);
    }
}

// Handle calibration status updates
function handleCalibrationStatus(status) {
    const progressBar = document.getElementById('calibrationProgress');
    const progressPercent = document.getElementById('calibrationPercent');
    const message = document.getElementById('calibrationMessage');

    progressBar.style.width = status.progress + '%';
    progressPercent.textContent = status.progress + '%';
    message.textContent = status.message;

    if (status.status === 'completed') {
        setTimeout(() => {
            calibrationModal.classList.remove('show');
            isCalibrating = false;
            alert('Calibration completed successfully! You can now start detection.');
        }, 1500);
    } else if (status.status === 'failed') {
        setTimeout(() => {
            calibrationModal.classList.remove('show');
            isCalibrating = false;
        }, 1500);
    }
}

// Update metrics display
function updateMetrics(metrics) {
    // Update numeric values
    document.getElementById('fatigueScore').textContent = metrics.fatigue_score.toFixed(1) + '%';
    document.getElementById('attentionScore').textContent = metrics.attention_score.toFixed(1) + '%';
    document.getElementById('perclos').textContent = metrics.perclos.toFixed(1) + '%';
    document.getElementById('ear').textContent = metrics.ear.toFixed(3);
    document.getElementById('mar').textContent = metrics.mar.toFixed(3);
    document.getElementById('blinkCount').textContent = metrics.blink_count;
    document.getElementById('blinkRate').textContent = metrics.blink_rate;
    document.getElementById('yawnCount').textContent = metrics.yawn_count;
    document.getElementById('yawnRate').textContent = metrics.yawn_rate;
    document.getElementById('microsleepCount').textContent = metrics.microsleep_count;
    document.getElementById('fpsCounter').textContent = metrics.fps.toFixed(1) + ' FPS';

    // Update progress bars
    document.getElementById('fatigueFill').style.width = metrics.fatigue_score + '%';
    document.getElementById('attentionFill').style.width = metrics.attention_score + '%';

    // Update alert badge
    const alertBadge = document.getElementById('alertBadge');
    const alertLevels = ['NORMAL', 'WARNING', 'MODERATE', 'HIGH', 'CRITICAL'];
    alertBadge.textContent = alertLevels[metrics.alert_level];
    alertBadge.className = 'alert-badge level-' + metrics.alert_level;

    // Update session time
    const elapsed = metrics.elapsed_time;
    const hours = Math.floor(elapsed / 3600);
    const minutes = Math.floor((elapsed % 3600) / 60);
    const seconds = Math.floor(elapsed % 60);
    document.getElementById('sessionTime').textContent =
        String(hours).padStart(2, '0') + ':' +
        String(minutes).padStart(2, '0') + ':' +
        String(seconds).padStart(2, '0');
}

// Update charts
function updateCharts(metrics) {
    // Add data to buffers
    timestamps.push(metrics.elapsed_time.toFixed(0) + 's');
    fatigueData.push(metrics.fatigue_score);
    earData.push(metrics.ear);
    perclosData.push(metrics.perclos);

    // Keep only recent data
    if (timestamps.length > chartDataLimit) {
        timestamps.shift();
        fatigueData.shift();
        earData.shift();
        perclosData.shift();
    }

    // Update fatigue chart
    fatigueChart.data.labels = timestamps;
    fatigueChart.data.datasets[0].data = fatigueData;
    fatigueChart.update();

    // Update metrics chart
    metricsChart.data.labels = timestamps;
    metricsChart.data.datasets[0].data = earData;
    metricsChart.data.datasets[1].data = perclosData;
    metricsChart.update();
}

// Show alert toast
function showAlert(alert) {
    const toast = alertToast;
    const alertLevels = ['Normal', 'Early Warning', 'Moderate Fatigue', 'High Fatigue', 'Critical Drowsiness'];

    toast.innerHTML = `
        <h3>${alertLevels[alert.level]}</h3>
        <p>${alert.message}</p>
    `;

    // Set border color based on level
    const colors = ['#10b981', '#fbbf24', '#f59e0b', '#f97316', '#ef4444'];
    toast.style.borderLeftColor = colors[alert.level];

    toast.classList.add('show');

    // Auto-hide after 5 seconds
    setTimeout(() => {
        toast.classList.remove('show');
    }, 5000);
}

// Statistics polling
let statisticsInterval = null;

function startStatisticsPolling() {
    statisticsInterval = setInterval(updateStatistics, 5000); // Every 5 seconds
    updateStatistics(); // Initial update
}

function stopStatisticsPolling() {
    if (statisticsInterval) {
        clearInterval(statisticsInterval);
        statisticsInterval = null;
    }
}

async function updateStatistics() {
    try {
        const response = await fetch('/api/statistics');
        const stats = await response.json();

        if (Object.keys(stats).length > 0) {
            document.getElementById('avgFatigue').textContent = stats.avg_fatigue.toFixed(1) + '%';
            document.getElementById('maxFatigue').textContent = stats.max_fatigue.toFixed(1) + '%';
            document.getElementById('avgPerclos').textContent = stats.avg_perclos.toFixed(1) + '%';

            const totalAlerts = Object.values(stats.alert_distribution).reduce((a, b) => a + b, 0);
            document.getElementById('totalAlerts').textContent = totalAlerts;
        }
    } catch (error) {
        console.error('Error fetching statistics:', error);
    }
}

// Initialize on page load
document.addEventListener('DOMContentLoaded', () => {
    initCharts();
    console.log('Advanced Drowsiness Detection System initialized');
});
