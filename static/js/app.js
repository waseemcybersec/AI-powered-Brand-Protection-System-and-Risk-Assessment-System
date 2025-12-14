// Brand Protection System - Frontend JavaScript

let currentTaskId = null;
let currentResults = [];
let pollInterval = null;

// Initialize on page load
document.addEventListener('DOMContentLoaded', function() {
    setupEventListeners();
});

// Setup event listeners
function setupEventListeners() {
    // Pipeline form submission
    document.getElementById('pipelineForm').addEventListener('submit', function(e) {
        e.preventDefault();
        startPipeline();
    });

    // Show/hide thresholds based on detection mode selection
    const mimicRadio = document.getElementById('mimicDetection');
    const logoRadio = document.getElementById('logoDetection');
    const threatRadio = document.getElementById('threatIntelligence');
    const darkWebRadio = document.getElementById('darkWebMonitoring');
    const socialMediaRadio = document.getElementById('socialMediaMonitoring');
    const thresholdsDiv = document.getElementById('mimicThresholds');
    
    function updateDetectionMode() {
        if (mimicRadio.checked) {
            thresholdsDiv.style.display = 'block';
        } else if (logoRadio.checked || threatRadio.checked || darkWebRadio.checked || socialMediaRadio.checked) {
            thresholdsDiv.style.display = 'none';
        }
    }
    
    mimicRadio.addEventListener('change', updateDetectionMode);
    logoRadio.addEventListener('change', updateDetectionMode);
    threatRadio.addEventListener('change', updateDetectionMode);
    darkWebRadio.addEventListener('change', updateDetectionMode);
    socialMediaRadio.addEventListener('change', updateDetectionMode);
    updateDetectionMode(); // Initial update
}

// Start pipeline
async function startPipeline() {
    const form = document.getElementById('pipelineForm');
    const submitBtn = form.querySelector('button[type="submit"]');
    
    // Disable form
    form.querySelectorAll('input, select, button').forEach(el => el.disabled = true);
    submitBtn.innerHTML = '<span class="spinner-border spinner-border-sm"></span> Starting...';
    
    // Get form data
    const detectionMode = document.querySelector('input[name="detectionMode"]:checked').value;
    const runMimic = detectionMode === 'mimic';
    const runLogo = detectionMode === 'logo';
    const runThreat = detectionMode === 'threat';
    const runDarkWeb = detectionMode === 'darkweb';
    const runSocialMedia = detectionMode === 'socialmedia';
    
    const data = {
        domain: document.getElementById('brandDomain').value.trim(),
        tlds: document.getElementById('tlds').value.split(',').map(t => t.trim()),
        detection_mode: detectionMode,
        capture_screenshots: runMimic || runLogo,  // Always capture for both modes
        capture_reference: runMimic,   // Only capture reference for mimic detection
        run_mimic_detection: runMimic,
        run_logo_detection: runLogo,
        run_threat_intelligence: runThreat,
        run_dark_web_monitoring: runDarkWeb,
        run_social_media_monitoring: runSocialMedia,
        phash_threshold: runMimic ? parseInt(document.getElementById('phashThreshold').value) : 2,  // For logo: secondary validation (default: 2, strict)
        similarity_threshold: runMimic ? parseFloat(document.getElementById('clipThreshold').value) : 0.85  // For logo: DINOv2 similarity (default: 0.85, strict)
    };
    
    try {
        const response = await fetch('/api/pipeline/run', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(data)
        });
        
        if (!response.ok) {
            throw new Error('Failed to start pipeline');
        }
        
        const result = await response.json();
        currentTaskId = result.task_id;
        
        // Show progress section
        document.getElementById('progressSection').style.display = 'block';
        document.getElementById('progressLog').innerHTML = '';
        
        // Start polling for status
        startPolling();
        
        // Scroll to progress
        document.getElementById('progressSection').scrollIntoView({ behavior: 'smooth' });
        
    } catch (error) {
        alert('Error starting pipeline: ' + error.message);
        form.querySelectorAll('input, select, button').forEach(el => el.disabled = false);
        submitBtn.innerHTML = '<i class="bi bi-play-fill"></i> Start Analysis';
    }
}

// Poll for task status
function startPolling() {
    if (pollInterval) {
        clearInterval(pollInterval);
    }
    
    pollInterval = setInterval(async () => {
        if (!currentTaskId) return;
        
        try {
            const response = await fetch(`/api/status/${currentTaskId}`);
            if (!response.ok) return;
            
            const status = await response.json();
            updateProgress(status);
            
            if (status.status === 'completed' || status.status === 'error') {
                clearInterval(pollInterval);
                pollInterval = null;
                
            if (status.status === 'completed') {
                    const detectionMode = document.querySelector('input[name="detectionMode"]:checked').value;
                    if (detectionMode === 'threat') {
                        await loadThreatIntelligenceReport(currentTaskId);
                    } else if (detectionMode === 'darkweb') {
                        await loadDarkWebReport(currentTaskId);
                    } else if (detectionMode === 'socialmedia') {
                        await loadSocialMediaReport(currentTaskId);
                    } else {
                await loadReport(currentTaskId);
                    }
            }
                
                // Re-enable form
                const form = document.getElementById('pipelineForm');
                form.querySelectorAll('input, select, button').forEach(el => el.disabled = false);
                const submitBtn = form.querySelector('button[type="submit"]');
                submitBtn.innerHTML = '<i class="bi bi-play-fill"></i> Start Analysis';
            }
        } catch (error) {
            console.error('Error polling status:', error);
        }
    }, 2000); // Poll every 2 seconds
}

// Update progress display
function updateProgress(status) {
    const progressBar = document.getElementById('progressBar');
    const progressPercent = document.getElementById('progressPercent');
    const progressMessage = document.getElementById('progressMessage');
    const progressLog = document.getElementById('progressLog');
    
    const percent = status.progress || 0;
    progressBar.style.width = percent + '%';
    progressBar.textContent = percent + '%';
    progressPercent.textContent = percent + '%';
    progressMessage.textContent = status.message || 'Processing...';
    
    // Add log entry
    const logClass = status.status === 'error' ? 'error' : 
                     status.status === 'completed' ? 'success' : 'info';
    const timestamp = new Date().toLocaleTimeString();
    const logEntry = document.createElement('div');
    logEntry.className = `log-entry ${logClass}`;
    logEntry.textContent = `[${timestamp}] ${status.message}`;
    progressLog.appendChild(logEntry);
    progressLog.scrollTop = progressLog.scrollHeight;
    
    // Update progress bar color
    if (status.status === 'error') {
        progressBar.classList.remove('bg-info');
        progressBar.classList.add('bg-danger');
    } else if (status.status === 'completed') {
        progressBar.classList.remove('bg-info');
        progressBar.classList.add('bg-success');
        progressBar.classList.remove('progress-bar-animated');
    }
}

// Load report
async function loadReport(taskId) {
    try {
        const response = await fetch(`/api/results/${taskId}`);
        if (!response.ok) {
            throw new Error('Failed to load results');
        }
        
        currentResults = await response.json();
        displayReport(currentResults);
        
        // Show report section
        document.getElementById('report').style.display = 'block';
        document.getElementById('report').scrollIntoView({ behavior: 'smooth' });
        
    } catch (error) {
        console.error('Error loading report:', error);
        alert('Error loading report: ' + error.message);
    }
}

// Display report
function displayReport(results) {
    // Check if this is logo detection or mimic detection
    const isLogoDetection = results.length > 0 && results[0].hasOwnProperty('logo_detected');
    
    // Categorize results
    const mimicSites = [];
    const brandOwnedSites = [];
    const otherSites = [];
    
    results.forEach(row => {
        const realBrand = (row.real_brand || 'no').toLowerCase();
        
        if (realBrand === 'yes') {
            brandOwnedSites.push(row);
        } else if (isLogoDetection) {
            // Logo detection mode - ONLY check logo_detected
            const logoDetected = (row.logo_detected || 'No').toLowerCase();
            if (logoDetected === 'yes') {
                mimicSites.push(row);
            } else {
                otherSites.push(row);
            }
        } else {
            // Mimic detection mode - ONLY check mimic_brand
            const mimic = (row.mimic_brand || 'No').toLowerCase();
            if (mimic === 'yes') {
                mimicSites.push(row);
            } else {
                otherSites.push(row);
            }
        }
    });
    
    // Calculate risk assessment statistics
    let totalRiskScore = 0;
    let riskCount = 0;
    let criticalCount = 0;
    let highCount = 0;
    let mediumCount = 0;
    let lowCount = 0;
    let veryLowCount = 0;
    
    results.forEach(row => {
        const riskScore = parseFloat(row.risk_score) || 0;
        const riskLevel = (row.risk_level || 'Very Low').toLowerCase();
        
        if (riskScore > 0) {
            totalRiskScore += riskScore;
            riskCount++;
        }
        
        if (riskScore >= 80) criticalCount++;
        else if (riskScore >= 60) highCount++;
        else if (riskScore >= 40) mediumCount++;
        else if (riskScore >= 20) lowCount++;
        else veryLowCount++;
    });
    
    const avgRiskScore = riskCount > 0 ? (totalRiskScore / riskCount) : 0;
    
    // Determine overall risk color
    let overallRiskColor = '#198754'; // Green
    let overallRiskLevel = 'Very Low';
    if (avgRiskScore >= 80) {
        overallRiskColor = '#dc3545';
        overallRiskLevel = 'Critical';
    } else if (avgRiskScore >= 60) {
        overallRiskColor = '#fd7e14';
        overallRiskLevel = 'High';
    } else if (avgRiskScore >= 40) {
        overallRiskColor = '#ffc107';
        overallRiskLevel = 'Medium';
    } else if (avgRiskScore >= 20) {
        overallRiskColor = '#0dcaf0';
        overallRiskLevel = 'Low';
    }
    
    // Display risk assessment summary
    const riskSummaryHtml = `
        <div class="row mb-4">
            <div class="col-md-12">
                <div class="card" style="border-left: 5px solid ${overallRiskColor};">
                    <div class="card-body text-center">
                        <h3 class="mb-3">Risk Assessment Summary</h3>
                        <div style="font-size: 72px; font-weight: bold; margin: 20px 0; color: ${overallRiskColor};">
                            ${avgRiskScore.toFixed(1)}/100
                        </div>
                        <div style="font-size: 24px; margin-bottom: 20px; color: ${overallRiskColor};">
                            Overall Risk Level: <strong>${overallRiskLevel}</strong>
                        </div>
                        <div class="row mt-4">
                            <div class="col-md-2">
                                <div class="card bg-danger text-white">
                                    <div class="card-body text-center">
                                        <h4>${criticalCount}</h4>
                                        <small>Critical (80-100)</small>
                                    </div>
                                </div>
                            </div>
                            <div class="col-md-2">
                                <div class="card" style="background-color: #fd7e14; color: white;">
                                    <div class="card-body text-center">
                                        <h4>${highCount}</h4>
                                        <small>High (60-79)</small>
                                    </div>
                                </div>
                            </div>
                            <div class="col-md-2">
                                <div class="card bg-warning text-white">
                                    <div class="card-body text-center">
                                        <h4>${mediumCount}</h4>
                                        <small>Medium (40-59)</small>
                                    </div>
                                </div>
                            </div>
                            <div class="col-md-2">
                                <div class="card" style="background-color: #0dcaf0; color: white;">
                                    <div class="card-body text-center">
                                        <h4>${lowCount}</h4>
                                        <small>Low (20-39)</small>
                                    </div>
                                </div>
                            </div>
                            <div class="col-md-2">
                                <div class="card bg-success text-white">
                                    <div class="card-body text-center">
                                        <h4>${veryLowCount}</h4>
                                        <small>Very Low (0-19)</small>
                                    </div>
                                </div>
                            </div>
                            <div class="col-md-2">
                                <div class="card bg-secondary text-white">
                                    <div class="card-body text-center">
                                        <h4>${results.length}</h4>
                                        <small>Total Domains</small>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    `;
    
    // Insert risk summary before summary stats
    const summaryStats = document.getElementById('summaryStats');
    if (summaryStats && !document.getElementById('riskAssessmentSummary')) {
        const riskDiv = document.createElement('div');
        riskDiv.id = 'riskAssessmentSummary';
        riskDiv.innerHTML = riskSummaryHtml;
        summaryStats.parentNode.insertBefore(riskDiv, summaryStats);
    } else if (document.getElementById('riskAssessmentSummary')) {
        document.getElementById('riskAssessmentSummary').innerHTML = riskSummaryHtml;
    }
    
    // Update counts and labels
    document.getElementById('mimicCount').textContent = mimicSites.length;
    document.getElementById('brandOwnedCount').textContent = brandOwnedSites.length;
    document.getElementById('otherCount').textContent = otherSites.length;
    document.getElementById('mimicBadge').textContent = mimicSites.length;
    document.getElementById('brandOwnedBadge').textContent = brandOwnedSites.length;
    document.getElementById('otherBadge').textContent = otherSites.length;
    
    // Update section title
    if (isLogoDetection) {
        document.getElementById('mimicCountLabel').textContent = 'Sites Using Logo';
        document.getElementById('mimicSectionTitle').textContent = 'Sites Using Brand Logo';
    } else {
        document.getElementById('mimicCountLabel').textContent = 'Mimic Sites';
        document.getElementById('mimicSectionTitle').textContent = 'Mimic Sites';
    }
    
    // Display mimic/logo sites
    displaySiteCards(mimicSites, 'mimicSites', 'danger', isLogoDetection);
    
    // Display brand-owned sites
    displaySiteCards(brandOwnedSites, 'brandOwnedSites', 'info', false);
    
    // Display other sites
    displaySiteCards(otherSites, 'otherSites', 'secondary', false);
}

// Display site cards
function displaySiteCards(sites, containerId, colorClass, isLogoDetection = false) {
    const container = document.getElementById(containerId);
    container.innerHTML = '';
    
    if (sites.length === 0) {
        container.innerHTML = '<div class="col-12"><p class="text-muted">No sites in this category</p></div>';
        return;
    }
    
    sites.forEach(site => {
        const col = document.createElement('div');
        col.className = 'col-md-6 col-lg-4 mb-3';
        
        const card = document.createElement('div');
        card.className = `card border-${colorClass} h-100`;
        
        const domain = site.domain || '';
        const url = site.final_url || `https://${domain}`;
        const title = site.title || 'No title';
        const ip = site.ip || 'N/A';
        const status = site.http_status || 'N/A';
        const sslIssuer = site.ssl_issuer || 'N/A';
        
        let cardBody = `
            <div class="card-header bg-${colorClass} text-white">
                <h6 class="mb-0"><i class="bi bi-link-45deg"></i> <a href="${url}" target="_blank" class="text-white text-decoration-none">${domain}</a></h6>
            </div>
            <div class="card-body">
                <p class="card-text"><strong>Title:</strong> ${title.substring(0, 60)}${title.length > 60 ? '...' : ''}</p>
                <p class="card-text small"><strong>IP:</strong> ${ip}</p>
                <p class="card-text small"><strong>Status:</strong> <span class="badge bg-${status === '200' ? 'success' : status === '404' ? 'danger' : 'warning'}">${status}</span></p>
                <p class="card-text small"><strong>SSL Issuer:</strong> ${sslIssuer}</p>
        `;
        
        // Add risk assessment (for both logo and mimic detection)
        if (site.risk_score !== null && site.risk_score !== undefined) {
            const riskScore = parseFloat(site.risk_score);
            const riskLevel = site.risk_level || 'Unknown';
            let riskColor = '#198754'; // Green
            if (riskScore >= 80) riskColor = '#dc3545'; // Red
            else if (riskScore >= 60) riskColor = '#fd7e14'; // Orange
            else if (riskScore >= 40) riskColor = '#ffc107'; // Yellow
            else if (riskScore >= 20) riskColor = '#0dcaf0'; // Cyan
            
            cardBody += `<p class="card-text"><strong>Risk Score:</strong> <span style="color: ${riskColor}; font-weight: bold;">${riskScore.toFixed(1)}/100</span> <span class="badge" style="background-color: ${riskColor};">${riskLevel}</span></p>`;
        }
        
        // Add logo detection metrics if available
        if (isLogoDetection) {
            if (site.detection_method !== null && site.detection_method !== undefined && site.detection_method !== 'none') {
                cardBody += `<p class="card-text small"><strong>Detection Method:</strong> ${site.detection_method}</p>`;
            }
            if (site.confidence !== null && site.confidence !== undefined) {
                cardBody += `<p class="card-text small"><strong>Confidence:</strong> ${(parseFloat(site.confidence) * 100).toFixed(1)}%</p>`;
            }
        } else {
            // Add mimic detection metrics if available
            if (site.phash_distance !== null && site.phash_distance !== undefined) {
                cardBody += `<p class="card-text small"><strong>pHash Distance:</strong> ${site.phash_distance}</p>`;
            }
            if (site.clip_similarity !== null && site.clip_similarity !== undefined) {
                cardBody += `<p class="card-text small"><strong>CLIP Similarity:</strong> ${(parseFloat(site.clip_similarity) * 100).toFixed(1)}%</p>`;
            }
        }
        
        cardBody += `</div>`;
        
        card.innerHTML = cardBody;
        col.appendChild(card);
        container.appendChild(col);
    });
}

// Show screenshot in modal
function showScreenshot(src) {
    document.getElementById('screenshotImage').src = src;
    const modal = new bootstrap.Modal(document.getElementById('screenshotModal'));
    modal.show();
}

// Download report
function downloadReport() {
    if (!currentTaskId || !currentResults || currentResults.length === 0) {
        alert('No report to download');
        return;
    }
    
    // Check if this is logo detection or mimic detection
    const isLogoDetection = currentResults.length > 0 && currentResults[0].hasOwnProperty('logo_detected');
    
    // Download CSV file
    const filename = isLogoDetection ? 'dataset_with_logo.csv' : 'dataset_with_mimic.csv';
    const link = document.createElement('a');
    link.href = `/api/results/file/${filename}`;
    link.download = isLogoDetection ? 'logo_detection_report.csv' : 'brand_protection_report.csv';
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
}

// Load threat intelligence report
async function loadThreatIntelligenceReport(taskId) {
    try {
        const response = await fetch(`/api/threat-intelligence/results/${taskId}`);
        if (!response.ok) {
            throw new Error('Failed to load threat intelligence results');
        }
        
        const tiResults = await response.json();
        displayThreatIntelligenceReport(tiResults);
        
        // Show report section
        document.getElementById('report').style.display = 'block';
        document.getElementById('report').scrollIntoView({ behavior: 'smooth' });
        
    } catch (error) {
        console.error('Error loading threat intelligence report:', error);
        alert('Error loading threat intelligence report: ' + error.message);
    }
}

// Display threat intelligence report
function displayThreatIntelligenceReport(results) {
    // Hide regular report sections
    document.getElementById('summaryStats').style.display = 'none';
    document.getElementById('mimicSection').style.display = 'none';
    document.getElementById('brandOwnedSites').parentElement.style.display = 'none';
    document.getElementById('otherSites').parentElement.style.display = 'none';
    document.getElementById('downloadReportBtn').style.display = 'none';
    
    // Show threat intelligence report
    document.getElementById('threatIntelligenceReport').style.display = 'block';
    document.getElementById('downloadTIReportBtn').style.display = 'inline-block';
    
    const riskScore = results.risk_score || 0;
    const riskLevel = results.risk_level || 'Unknown';
    const threats = results.threats_found || [];
    const summary = results.summary || {};
    const sources = results.sources || {};
    
    // Determine risk color
    let riskColor = '#198754'; // Green
    if (riskScore >= 70) riskColor = '#dc3545'; // Red
    else if (riskScore >= 50) riskColor = '#fd7e14'; // Orange
    else if (riskScore >= 30) riskColor = '#ffc107'; // Yellow
    else if (riskScore >= 10) riskColor = '#0dcaf0'; // Cyan
    
    // Display risk score
    const riskScoreDisplay = document.getElementById('riskScoreDisplay');
    riskScoreDisplay.textContent = `${riskScore}/100`;
    riskScoreDisplay.style.color = riskColor;
    
    const riskLevelDisplay = document.getElementById('riskLevelDisplay');
    riskLevelDisplay.textContent = `Risk Level: ${riskLevel}`;
    riskLevelDisplay.style.color = riskColor;
    
    // Update risk score card border
    document.getElementById('riskScoreCard').style.borderLeft = `5px solid ${riskColor}`;
    
    // Display summary
    const riskSummary = document.getElementById('riskSummary');
    riskSummary.innerHTML = `
        <div class="col-md-3">
            <div class="card bg-primary text-white">
                <div class="card-body text-center">
                    <h4>${summary.total_sources || 0}</h4>
                    <p class="mb-0">Sources Checked</p>
                </div>
            </div>
        </div>
        <div class="col-md-3">
            <div class="card bg-danger text-white">
                <div class="card-body text-center">
                    <h4>${summary.sources_with_threats || 0}</h4>
                    <p class="mb-0">Sources with Threats</p>
                </div>
            </div>
        </div>
        <div class="col-md-3">
            <div class="card bg-warning text-white">
                <div class="card-body text-center">
                    <h4>${summary.high_risk_indicators || 0}</h4>
                    <p class="mb-0">High Risk Indicators</p>
                </div>
            </div>
        </div>
        <div class="col-md-3">
            <div class="card bg-info text-white">
                <div class="card-body text-center">
                    <h4>${threats.length}</h4>
                    <p class="mb-0">Total Threats</p>
                </div>
            </div>
        </div>
    `;
    
    // Display threats
    const threatsList = document.getElementById('threatsList');
    threatsList.innerHTML = '';
    if (threats.length === 0) {
        threatsList.innerHTML = '<div class="alert alert-success"><i class="bi bi-check-circle"></i> No threats identified.</div>';
    } else {
        threats.forEach(threat => {
            const item = document.createElement('div');
            item.className = 'list-group-item list-group-item-warning';
            item.innerHTML = `<i class="bi bi-exclamation-triangle"></i> ${threat}`;
            threatsList.appendChild(item);
        });
    }
    
    // Display source details
    const sourceDetails = document.getElementById('sourceDetails');
    sourceDetails.innerHTML = '';
    let accordionIndex = 0;
    
    for (const [sourceName, sourceData] of Object.entries(sources)) {
        const status = sourceData.status || 'unknown';
        const statusClass = status === 'success' ? 'success' : (status === 'error' ? 'danger' : 'secondary');
        const statusIcon = status === 'success' ? 'check-circle' : (status === 'error' ? 'x-circle' : 'info-circle');
        
        const accordionItem = document.createElement('div');
        accordionItem.className = 'accordion-item';
        accordionItem.innerHTML = `
            <h2 class="accordion-header" id="heading${accordionIndex}">
                <button class="accordion-button ${accordionIndex === 0 ? '' : 'collapsed'}" type="button" data-bs-toggle="collapse" data-bs-target="#collapse${accordionIndex}">
                    <i class="bi bi-${statusIcon} text-${statusClass} me-2"></i>
                    ${sourceData.source || sourceName.charAt(0).toUpperCase() + sourceName.slice(1)}
                    <span class="badge bg-${statusClass} ms-2">${status}</span>
                </button>
            </h2>
            <div id="collapse${accordionIndex}" class="accordion-collapse collapse ${accordionIndex === 0 ? 'show' : ''}" data-bs-parent="#sourceDetails">
                <div class="accordion-body">
                    ${formatSourceDetails(sourceData)}
                </div>
            </div>
        `;
        sourceDetails.appendChild(accordionItem);
        accordionIndex++;
    }
    
    // Store TI results for download
    window.currentTIResults = results;
}

// Format source details for display
function formatSourceDetails(sourceData) {
    let html = '';
    
    if (sourceData.status === 'success') {
        // Create a nice formatted display
        html = '<div class="table-responsive"><table class="table table-sm table-bordered">';
        
        // Special handling for different sources
        if (sourceData.source === 'AbuseIPDB') {
            html += formatAbuseIPDBDetails(sourceData);
        } else if (sourceData.source === 'Google Safe Browsing') {
            html += formatSafeBrowsingDetails(sourceData);
        } else if (sourceData.source === 'WHOIS') {
            html += formatWHOISDetails(sourceData);
        } else if (sourceData.source === 'VirusTotal') {
            html += formatVirusTotalDetails(sourceData);
        } else {
            // Generic formatting
            for (const [key, value] of Object.entries(sourceData)) {
                if (key !== 'source' && key !== 'status' && key !== 'error' && key !== 'message' && key !== 'summary' && key !== 'traceback') {
                    let displayValue = formatValue(value);
                    html += `<tr><td class="fw-bold" style="width: 40%;">${formatKey(key)}</td><td>${displayValue}</td></tr>`;
                }
            }
        }
        
        html += '</table></div>';
        
        // Add summary if available
        if (sourceData.summary) {
            html += `<div class="alert alert-info mt-2 mb-0"><i class="bi bi-info-circle"></i> ${sourceData.summary}</div>`;
        }
    } else if (sourceData.status === 'no_api_key') {
        html = `<div class="alert alert-secondary"><i class="bi bi-info-circle"></i> ${sourceData.message || 'API key not provided'}</div>`;
    } else if (sourceData.status === 'error') {
        html = `<div class="alert alert-danger"><i class="bi bi-exclamation-triangle"></i> <strong>Error:</strong> ${sourceData.error || sourceData.message || 'Unknown error'}</div>`;
    } else if (sourceData.status === 'partial') {
        html = `<div class="alert alert-warning"><i class="bi bi-exclamation-circle"></i> ${sourceData.message || 'Partial data available'}</div>`;
        if (sourceData.summary) {
            html += `<p class="mt-2">${sourceData.summary}</p>`;
        }
    } else {
        html = `<div class="alert alert-secondary">Status: ${sourceData.status}</div>`;
    }
    
    return html || '<p class="text-muted">No additional details available.</p>';
}

function formatKey(key) {
    return key.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase());
}

function formatValue(value) {
    if (value === null || value === undefined) return '<span class="text-muted">N/A</span>';
    if (typeof value === 'boolean') return value ? '<span class="badge bg-success">Yes</span>' : '<span class="badge bg-secondary">No</span>';
    if (typeof value === 'object') {
        if (Array.isArray(value)) {
            if (value.length === 0) return '<span class="text-muted">None</span>';
            return value.map(v => `<span class="badge bg-secondary me-1">${v}</span>`).join('');
        }
        return `<pre class="mb-0" style="font-size: 0.85em;">${JSON.stringify(value, null, 2)}</pre>`;
    }
    return String(value);
}

function formatAbuseIPDBDetails(data) {
    let html = '';
    html += `<tr><td class="fw-bold">IP Address</td><td><code>${data.ip || 'N/A'}</code></td></tr>`;
    html += `<tr><td class="fw-bold">Abuse Confidence</td><td><span class="badge ${data.abuse_confidence > 50 ? 'bg-danger' : data.abuse_confidence > 25 ? 'bg-warning' : 'bg-success'}">${data.abuse_confidence || 0}%</span></td></tr>`;
    html += `<tr><td class="fw-bold">Total Reports</td><td>${data.total_reports || 0}</td></tr>`;
    html += `<tr><td class="fw-bold">Distinct Users</td><td>${data.num_distinct_users || 0}</td></tr>`;
    html += `<tr><td class="fw-bold">Usage Type</td><td>${data.usage_type || 'Unknown'}</td></tr>`;
    html += `<tr><td class="fw-bold">Country</td><td>${data.country || 'Unknown'}</td></tr>`;
    html += `<tr><td class="fw-bold">ISP</td><td>${data.isp || 'Unknown'}</td></tr>`;
    html += `<tr><td class="fw-bold">Is Public</td><td>${formatValue(data.is_public)}</td></tr>`;
    html += `<tr><td class="fw-bold">Is Whitelisted</td><td>${formatValue(data.is_whitelisted)}</td></tr>`;
    html += `<tr><td class="fw-bold">Last Reported</td><td>${data.last_reported_at || 'Never'}</td></tr>`;
    return html;
}

function formatSafeBrowsingDetails(data) {
    let html = '';
    html += `<tr><td class="fw-bold">Status</td><td>${data.is_unsafe ? '<span class="badge bg-danger">Unsafe</span>' : '<span class="badge bg-success">Safe</span>'}</td></tr>`;
    if (data.threats && data.threats.length > 0) {
        html += `<tr><td class="fw-bold">Threat Types</td><td>${data.threats.map(t => `<span class="badge bg-danger me-1">${t}</span>`).join('')}</td></tr>`;
        if (data.threat_details && data.threat_details.length > 0) {
            html += `<tr><td class="fw-bold">Threat Details</td><td><ul class="mb-0">`;
            data.threat_details.forEach(detail => {
                html += `<li><strong>${detail.threat_type}</strong> on ${detail.platform}</li>`;
            });
            html += `</ul></td></tr>`;
        }
    } else {
        html += `<tr><td class="fw-bold">Threats</td><td><span class="badge bg-success">No threats detected</span></td></tr>`;
    }
    return html;
}

function formatWHOISDetails(data) {
    let html = '';
    html += `<tr><td class="fw-bold">Registrar</td><td>${data.registrar || 'Unknown'}</td></tr>`;
    html += `<tr><td class="fw-bold">Created Date</td><td>${data.created_date || 'Unknown'}</td></tr>`;
    html += `<tr><td class="fw-bold">Expires Date</td><td>${data.expires_date || 'Unknown'}</td></tr>`;
    html += `<tr><td class="fw-bold">Updated Date</td><td>${data.updated_date || 'Unknown'}</td></tr>`;
    if (data.domain_age_days !== undefined) {
        html += `<tr><td class="fw-bold">Domain Age</td><td>${data.domain_age_days} days</td></tr>`;
    }
    html += `<tr><td class="fw-bold">Country</td><td>${data.country || 'Unknown'}</td></tr>`;
    html += `<tr><td class="fw-bold">Organization</td><td>${data.org || 'Unknown'}</td></tr>`;
    if (data.name_servers && data.name_servers.length > 0) {
        html += `<tr><td class="fw-bold">Name Servers</td><td>${data.name_servers.map(ns => `<code class="me-2">${ns}</code>`).join('')}</td></tr>`;
    }
    if (data.ip) {
        html += `<tr><td class="fw-bold">IP Address</td><td><code>${data.ip}</code></td></tr>`;
    }
    return html;
}

function formatVirusTotalDetails(data) {
    let html = '';
    html += `<tr><td class="fw-bold">Detection Count</td><td><span class="badge ${data.detection_count > 0 ? 'bg-danger' : 'bg-success'}">${data.detection_count || 0}</span></td></tr>`;
    html += `<tr><td class="fw-bold">Reputation</td><td>${data.reputation || 0}</td></tr>`;
    html += `<tr><td class="fw-bold">Last Scan</td><td>${data.last_scan || 'Unknown'}</td></tr>`;
    if (data.detections && data.detections.length > 0) {
        html += `<tr><td class="fw-bold">Detected URLs</td><td>${data.detections.length}</td></tr>`;
    }
    if (data.undetected && data.undetected.length > 0) {
        html += `<tr><td class="fw-bold">Undetected URLs</td><td>${data.undetected.length}</td></tr>`;
    }
    return html;
}


// Download threat intelligence report
function downloadTIReport() {
    if (!window.currentTIResults) {
        alert('No threat intelligence report to download');
        return;
    }
    
    const link = document.createElement('a');
    link.href = `/api/threat-intelligence/download/${currentTaskId}`;
    link.download = `threat_intelligence_report_${window.currentTIResults.domain || 'report'}.html`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
}

// Load dark web monitoring report
async function loadDarkWebReport(taskId) {
    try {
        const response = await fetch(`/api/dark-web/results/${taskId}`);
        if (!response.ok) {
            throw new Error('Failed to load dark web monitoring results');
        }
        
        const dwResults = await response.json();
        displayDarkWebReport(dwResults);
        
        // Show report section
        document.getElementById('report').style.display = 'block';
        document.getElementById('report').scrollIntoView({ behavior: 'smooth' });
        
    } catch (error) {
        console.error('Error loading dark web report:', error);
        alert('Error loading dark web report: ' + error.message);
    }
}

// Display dark web monitoring report (reuse threat intelligence display logic)
function displayDarkWebReport(results) {
    // Hide regular report sections
    document.getElementById('summaryStats').style.display = 'none';
    document.getElementById('mimicSection').style.display = 'none';
    document.getElementById('brandOwnedSites').parentElement.style.display = 'none';
    document.getElementById('otherSites').parentElement.style.display = 'none';
    document.getElementById('downloadReportBtn').style.display = 'none';
    
    // Show dark web report (reuse threat intelligence UI)
    document.getElementById('threatIntelligenceReport').style.display = 'block';
    document.getElementById('downloadTIReportBtn').style.display = 'inline-block';
    document.getElementById('downloadTIReportBtn').onclick = downloadDWReport;
    
    const riskScore = results.risk_score || 0;
    const riskLevel = results.risk_level || 'Unknown';
    const threats = results.threats_found || [];
    const summary = results.summary || {};
    const sources = results.sources || {};
    
    // Determine risk color
    let riskColor = '#198754'; // Green
    if (riskScore >= 70) riskColor = '#dc3545'; // Red
    else if (riskScore >= 50) riskColor = '#fd7e14'; // Orange
    else if (riskScore >= 30) riskColor = '#ffc107'; // Yellow
    else if (riskScore >= 10) riskColor = '#0dcaf0'; // Cyan
    
    // Display risk score
    const riskScoreDisplay = document.getElementById('riskScoreDisplay');
    riskScoreDisplay.textContent = `${riskScore}/100`;
    riskScoreDisplay.style.color = riskColor;
    
    const riskLevelDisplay = document.getElementById('riskLevelDisplay');
    riskLevelDisplay.textContent = `Risk Level: ${riskLevel}`;
    riskLevelDisplay.style.color = riskColor;
    
    // Update risk score card border
    document.getElementById('riskScoreCard').style.borderLeft = `5px solid ${riskColor}`;
    
    // Display summary
    const riskSummary = document.getElementById('riskSummary');
    riskSummary.innerHTML = `
        <div class="col-md-3">
            <div class="card bg-primary text-white">
                <div class="card-body text-center">
                    <h4>${summary.total_sources || 0}</h4>
                    <p class="mb-0">Sources Checked</p>
                </div>
            </div>
        </div>
        <div class="col-md-3">
            <div class="card bg-danger text-white">
                <div class="card-body text-center">
                    <h4>${summary.sources_with_findings || 0}</h4>
                    <p class="mb-0">Sources with Findings</p>
                </div>
            </div>
        </div>
        <div class="col-md-3">
            <div class="card bg-warning text-white">
                <div class="card-body text-center">
                    <h4>${summary.high_risk_indicators || 0}</h4>
                    <p class="mb-0">High Risk Indicators</p>
                </div>
            </div>
        </div>
        <div class="col-md-3">
            <div class="card bg-info text-white">
                <div class="card-body text-center">
                    <h4>${threats.length}</h4>
                    <p class="mb-0">Total Threats</p>
                </div>
            </div>
        </div>
    `;
    
    // Display threats
    const threatsList = document.getElementById('threatsList');
    threatsList.innerHTML = '';
    if (threats.length === 0) {
        threatsList.innerHTML = '<div class="alert alert-success"><i class="bi bi-check-circle"></i> No threats identified.</div>';
    } else {
        threats.forEach(threat => {
            const item = document.createElement('div');
            item.className = 'list-group-item list-group-item-warning';
            item.innerHTML = `<i class="bi bi-exclamation-triangle"></i> ${threat}`;
            threatsList.appendChild(item);
        });
    }
    
    // Display source details (reuse formatSourceDetails but add dark web specific formatting)
    const sourceDetails = document.getElementById('sourceDetails');
    sourceDetails.innerHTML = '';
    let accordionIndex = 0;
    
    for (const [sourceName, sourceData] of Object.entries(sources)) {
        const status = sourceData.status || 'unknown';
        const statusClass = status === 'success' ? 'success' : (status === 'error' ? 'danger' : 'secondary');
        const statusIcon = status === 'success' ? 'check-circle' : (status === 'error' ? 'x-circle' : 'info-circle');
        
        const accordionItem = document.createElement('div');
        accordionItem.className = 'accordion-item';
        accordionItem.innerHTML = `
            <h2 class="accordion-header" id="heading${accordionIndex}">
                <button class="accordion-button ${accordionIndex === 0 ? '' : 'collapsed'}" type="button" data-bs-toggle="collapse" data-bs-target="#collapse${accordionIndex}">
                    <i class="bi bi-${statusIcon} text-${statusClass} me-2"></i>
                    ${sourceData.source || sourceName.charAt(0).toUpperCase() + sourceName.slice(1)}
                    <span class="badge bg-${statusClass} ms-2">${status}</span>
                </button>
            </h2>
            <div id="collapse${accordionIndex}" class="accordion-collapse collapse ${accordionIndex === 0 ? 'show' : ''}" data-bs-parent="#sourceDetails">
                <div class="accordion-body">
                    ${formatDarkWebSourceDetails(sourceData)}
                </div>
            </div>
        `;
        sourceDetails.appendChild(accordionItem);
        accordionIndex++;
    }
    
    // Store DW results for download
    window.currentDWResults = results;
}

// Format dark web source details
function formatDarkWebSourceDetails(sourceData) {
    let html = '';
    
    if (sourceData.status === 'success') {
        html = '<div class="table-responsive"><table class="table table-sm table-bordered">';
        
        // Special handling for different sources
        if (sourceData.source === 'IntelligenceX') {
            html += formatIntelXDetails(sourceData);
        } else if (sourceData.source === 'DeHashed') {
            html += formatDeHashedDetails(sourceData);
        } else if (sourceData.source === 'Have I Been Pwned') {
            html += formatHIBPDetails(sourceData);
        } else {
            // Generic formatting
            for (const [key, value] of Object.entries(sourceData)) {
                if (key !== 'source' && key !== 'status' && key !== 'error' && key !== 'message' && key !== 'summary' && key !== 'traceback') {
                    let displayValue = formatValue(value);
                    html += `<tr><td class="fw-bold" style="width: 40%;">${formatKey(key)}</td><td>${displayValue}</td></tr>`;
                }
            }
        }
        
        html += '</table></div>';
        
        // Add summary if available
        if (sourceData.summary) {
            html += `<div class="alert alert-info mt-2 mb-0"><i class="bi bi-info-circle"></i> ${sourceData.summary}</div>`;
        }
    } else if (sourceData.status === 'no_api_key') {
        html = `<div class="alert alert-secondary"><i class="bi bi-info-circle"></i> ${sourceData.message || 'API key not provided'}</div>`;
    } else if (sourceData.status === 'error') {
        html = `<div class="alert alert-danger"><i class="bi bi-exclamation-triangle"></i> <strong>Error:</strong> ${sourceData.error || sourceData.message || 'Unknown error'}</div>`;
    } else if (sourceData.status === 'partial') {
        html = `<div class="alert alert-warning"><i class="bi bi-exclamation-circle"></i> ${sourceData.message || 'Partial data available'}</div>`;
        if (sourceData.summary) {
            html += `<p class="mt-2">${sourceData.summary}</p>`;
        }
    } else {
        html = `<div class="alert alert-secondary">Status: ${sourceData.status}</div>`;
    }
    
    return html || '<p class="text-muted">No additional details available.</p>';
}

function formatIntelXDetails(data) {
    let html = '';
    html += `<tr><td class="fw-bold">Total Results</td><td><span class="badge ${data.total_results > 0 ? 'bg-danger' : 'bg-success'}">${data.total_results || 0}</span></td></tr>`;
    html += `<tr><td class="fw-bold">Unique Sources</td><td>${data.source_count || 0}</td></tr>`;
    if (data.unique_sources && data.unique_sources.length > 0) {
        html += `<tr><td class="fw-bold">Source Types</td><td>${data.unique_sources.map(s => `<span class="badge bg-secondary me-1">${s}</span>`).join('')}</td></tr>`;
    }
    html += `<tr><td class="fw-bold">Has Findings</td><td>${data.has_findings ? '<span class="badge bg-danger">Yes</span>' : '<span class="badge bg-success">No</span>'}</td></tr>`;
    return html;
}

function formatDeHashedDetails(data) {
    let html = '';
    html += `<tr><td class="fw-bold">Total Records</td><td><span class="badge ${data.total > 0 ? 'bg-danger' : 'bg-success'}">${data.total || 0}</span></td></tr>`;
    html += `<tr><td class="fw-bold">Unique Breaches</td><td>${data.unique_breaches || 0}</td></tr>`;
    html += `<tr><td class="fw-bold">Email Count</td><td>${data.email_count || 0}</td></tr>`;
    html += `<tr><td class="fw-bold">Password Count</td><td>${data.password_count || 0}</td></tr>`;
    if (data.breach_sources && data.breach_sources.length > 0) {
        html += `<tr><td class="fw-bold">Breach Sources</td><td>${data.breach_sources.map(s => `<span class="badge bg-danger me-1">${s}</span>`).join('')}</td></tr>`;
    }
    html += `<tr><td class="fw-bold">Has Findings</td><td>${data.has_findings ? '<span class="badge bg-danger">Yes</span>' : '<span class="badge bg-success">No</span>'}</td></tr>`;
    return html;
}

function formatHIBPDetails(data) {
    let html = '';
    html += `<tr><td class="fw-bold">Breach Count</td><td><span class="badge ${data.breach_count > 0 ? 'bg-danger' : 'bg-success'}">${data.breach_count || 0}</span></td></tr>`;
    if (data.total_accounts_pwned) {
        html += `<tr><td class="fw-bold">Total Accounts Affected</td><td>${data.total_accounts_pwned.toLocaleString()}</td></tr>`;
    }
    if (data.breaches && data.breaches.length > 0) {
        html += `<tr><td class="fw-bold">Breaches</td><td><ul class="mb-0">`;
        data.breaches.forEach(breach => {
            html += `<li><strong>${breach.name}</strong> (${breach.breach_date}) - ${breach.pwn_count.toLocaleString()} accounts</li>`;
        });
        html += `</ul></td></tr>`;
    }
    html += `<tr><td class="fw-bold">Has Findings</td><td>${data.has_findings ? '<span class="badge bg-danger">Yes</span>' : '<span class="badge bg-success">No</span>'}</td></tr>`;
    return html;
}

// Download dark web report
function downloadDWReport() {
    if (!window.currentDWResults) {
        alert('No dark web monitoring report to download');
        return;
    }
    
    const link = document.createElement('a');
    link.href = `/api/dark-web/download/${currentTaskId}`;
    link.download = `dark_web_report_${window.currentDWResults.brand_domain || 'report'}.html`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
}

// Load social media monitoring report
async function loadSocialMediaReport(taskId) {
    try {
        const response = await fetch(`/api/social-media/results/${taskId}`);
        if (!response.ok) {
            throw new Error('Failed to load social media monitoring results');
        }
        
        const smResults = await response.json();
        displaySocialMediaReport(smResults);
        
        // Show report section
        document.getElementById('report').style.display = 'block';
        document.getElementById('report').scrollIntoView({ behavior: 'smooth' });
        
    } catch (error) {
        console.error('Error loading social media report:', error);
        alert('Error loading social media report: ' + error.message);
    }
}

// Display social media monitoring report
function displaySocialMediaReport(results) {
    // Hide regular report sections
    document.getElementById('summaryStats').style.display = 'none';
    document.getElementById('mimicSection').style.display = 'none';
    document.getElementById('brandOwnedSites').parentElement.style.display = 'none';
    document.getElementById('otherSites').parentElement.style.display = 'none';
    document.getElementById('downloadReportBtn').style.display = 'none';
    document.getElementById('downloadTIReportBtn').style.display = 'none';
    document.getElementById('downloadDWReportBtn').style.display = 'none';
    
    // Show social media report (reuse threat intelligence UI)
    document.getElementById('threatIntelligenceReport').style.display = 'block';
    document.getElementById('downloadSMReportBtn').style.display = 'inline-block';
    
    const riskScore = results.risk_score || 0;
    const riskLevel = results.risk_level || 'Unknown';
    const threats = results.threats_found || [];
    const summary = results.summary || {};
    const sources = results.sources || {};
    
    // Determine risk color
    let riskColor = '#198754'; // Green
    if (riskScore >= 70) riskColor = '#dc3545'; // Red
    else if (riskScore >= 50) riskColor = '#fd7e14'; // Orange
    else if (riskScore >= 30) riskColor = '#ffc107'; // Yellow
    else if (riskScore >= 10) riskColor = '#0dcaf0'; // Cyan
    
    // Display risk score
    const riskScoreDisplay = document.getElementById('riskScoreDisplay');
    riskScoreDisplay.textContent = `${riskScore}/100`;
    riskScoreDisplay.style.color = riskColor;
    
    const riskLevelDisplay = document.getElementById('riskLevelDisplay');
    riskLevelDisplay.textContent = `Risk Level: ${riskLevel}`;
    riskLevelDisplay.style.color = riskColor;
    
    // Update risk score card border
    document.getElementById('riskScoreCard').style.borderLeft = `5px solid ${riskColor}`;
    
    // Display summary
    const riskSummary = document.getElementById('riskSummary');
    riskSummary.innerHTML = `
        <div class="col-md-3">
            <div class="card bg-primary text-white">
                <div class="card-body text-center">
                    <h4>${summary.total_sources || 0}</h4>
                    <p class="mb-0">Platforms Checked</p>
                </div>
            </div>
        </div>
        <div class="col-md-3">
            <div class="card bg-danger text-white">
                <div class="card-body text-center">
                    <h4>${summary.sources_with_findings || 0}</h4>
                    <p class="mb-0">Platforms with Findings</p>
                </div>
            </div>
        </div>
        <div class="col-md-3">
            <div class="card bg-warning text-white">
                <div class="card-body text-center">
                    <h4>${summary.high_risk_indicators || 0}</h4>
                    <p class="mb-0">High Risk Indicators</p>
                </div>
            </div>
        </div>
        <div class="col-md-3">
            <div class="card bg-info text-white">
                <div class="card-body text-center">
                    <h4>${summary.total_threats || threats.length}</h4>
                    <p class="mb-0">Total Threats</p>
                </div>
            </div>
        </div>
    `;
    
    // Display threats
    const threatsList = document.getElementById('threatsList');
    threatsList.innerHTML = '';
    if (threats.length === 0) {
        threatsList.innerHTML = '<div class="alert alert-success"><i class="bi bi-check-circle"></i> No threats identified on social media platforms.</div>';
    } else {
        threats.forEach(threat => {
            const item = document.createElement('div');
            item.className = 'list-group-item list-group-item-warning';
            item.innerHTML = `<i class="bi bi-exclamation-triangle"></i> ${threat}`;
            threatsList.appendChild(item);
        });
    }
    
    // Display source details
    const sourceDetails = document.getElementById('sourceDetails');
    sourceDetails.innerHTML = '';
    let accordionIndex = 0;
    
    for (const [sourceName, sourceData] of Object.entries(sources)) {
        const status = sourceData.status || 'unknown';
        const statusClass = status === 'success' ? 'success' : (status === 'error' ? 'danger' : 'secondary');
        const statusIcon = getSourceIcon(sourceName);
        
        const accordionItem = document.createElement('div');
        accordionItem.className = 'accordion-item';
        accordionItem.innerHTML = `
            <h2 class="accordion-header" id="heading${accordionIndex}">
                <button class="accordion-button ${accordionIndex === 0 ? '' : 'collapsed'}" type="button" data-bs-toggle="collapse" data-bs-target="#collapse${accordionIndex}">
                    <i class="bi bi-${statusIcon} me-2"></i>
                    ${sourceData.source || sourceName.charAt(0).toUpperCase() + sourceName.slice(1)}
                    <span class="badge bg-${statusClass} ms-2">${status}</span>
                </button>
            </h2>
            <div id="collapse${accordionIndex}" class="accordion-collapse collapse ${accordionIndex === 0 ? 'show' : ''}" data-bs-parent="#sourceDetails">
                <div class="accordion-body">
                    ${formatSocialMediaSourceDetails(sourceData)}
                </div>
            </div>
        `;
        sourceDetails.appendChild(accordionItem);
        accordionIndex++;
    }
    
    // Store SM results for download
    window.currentSMResults = results;
}

// Get platform icon
function getSourceIcon(sourceName) {
    const icons = {
        'youtube': 'youtube text-danger',
        'telegram': 'telegram text-primary',
        'github': 'github'
    };
    return icons[sourceName.toLowerCase()] || 'share-fill';
}

// Format social media source details
function formatSocialMediaSourceDetails(sourceData) {
    let html = '';
    
    if (sourceData.status === 'success') {
        html = '<div class="table-responsive"><table class="table table-sm table-bordered">';
        
        // Special handling for different sources
        if (sourceData.source === 'YouTube') {
            html += formatYouTubeDetails(sourceData);
        } else if (sourceData.source === 'Telegram') {
            html += formatTelegramDetails(sourceData);
        } else if (sourceData.source === 'GitHub') {
            html += formatGitHubDetails(sourceData);
        } else {
            // Generic formatting
            for (const [key, value] of Object.entries(sourceData)) {
                if (key !== 'source' && key !== 'status' && key !== 'error' && key !== 'message' && key !== 'summary' && key !== 'traceback' && key !== 'threats' && key !== 'high_risk_count' && key !== 'has_findings') {
                    let displayValue = formatValue(value);
                    html += `<tr><td class="fw-bold" style="width: 40%;">${formatKey(key)}</td><td>${displayValue}</td></tr>`;
                }
            }
        }
        
        html += '</table></div>';
        
        // Add summary if available
        if (sourceData.summary) {
            html += `<div class="alert alert-info mt-2 mb-0"><i class="bi bi-info-circle"></i> ${sourceData.summary}</div>`;
        }
    } else if (sourceData.status === 'no_api_key') {
        html = `<div class="alert alert-secondary"><i class="bi bi-info-circle"></i> ${sourceData.message || 'API key not provided'}</div>`;
    } else if (sourceData.status === 'error') {
        html = `<div class="alert alert-danger"><i class="bi bi-exclamation-triangle"></i> <strong>Error:</strong> ${sourceData.error || sourceData.message || 'Unknown error'}</div>`;
    } else {
        html = `<div class="alert alert-secondary">Status: ${sourceData.status}</div>`;
    }
    
    return html || '<p class="text-muted">No additional details available.</p>';
}

function formatYouTubeDetails(data) {
    let html = '';
    html += `<tr><td class="fw-bold">Videos Scanned</td><td>${data.total_videos || 0}</td></tr>`;
    html += `<tr><td class="fw-bold">Threats Found</td><td><span class="badge ${data.suspicious_videos > 0 ? 'bg-danger' : 'bg-success'}">${data.suspicious_videos || 0}</span></td></tr>`;
    
    // Show typosquats checked
    if (data.typosquats_checked && data.typosquats_checked.length > 0) {
        html += `<tr><td class="fw-bold">Typosquats Checked</td><td>${data.typosquats_checked.map(t => `<span class="badge bg-secondary me-1">${t}</span>`).join('')}</td></tr>`;
    }
    
    // Show sample queries used
    if (data.queries_used && data.queries_used.length > 0) {
        html += `<tr><td class="fw-bold">Sample Threat Queries</td><td><small>${data.queries_used.slice(0, 5).map(q => `<code class="me-2">${q}</code>`).join('')}</small></td></tr>`;
    }
    
    // List ALL threat videos found
    if (data.videos_found && data.videos_found.length > 0) {
        html += `<tr><td class="fw-bold" colspan="2" style="background: #f8d7da;">⚠️ Threat Videos Detected (${data.videos_found.length})</td></tr>`;
        data.videos_found.forEach(video => {
            html += `<tr><td colspan="2" style="padding-left: 20px;">
                <div class="d-flex align-items-start">
                    ${video.thumbnail ? `<img src="${video.thumbnail}" style="width: 80px; height: 45px; object-fit: cover; margin-right: 10px; border-radius: 4px;">` : ''}
                    <div>
                        <a href="${video.url}" target="_blank" class="fw-bold text-danger">${video.title}</a>
                        <br><small class="text-muted">Channel: ${video.channel || 'Unknown'}</small>
                        <br><span class="badge bg-danger">${video.threat_type || 'Threat'}</span>
                    </div>
                </div>
            </td></tr>`;
        });
    }
    
    // Show all scanned videos (not just threats) for transparency
    if (data.all_videos_scanned && data.all_videos_scanned.length > 0 && data.suspicious_videos === 0) {
        html += `<tr><td class="fw-bold" colspan="2" style="background: #d1e7dd;">✅ Videos Scanned (No Threats Found)</td></tr>`;
        html += `<tr><td colspan="2"><small class="text-muted">Scanned ${data.all_videos_scanned.length} videos using threat-specific queries. No malicious content detected.</small></td></tr>`;
    }
    
    return html;
}

function formatTelegramDetails(data) {
    let html = '';
    html += `<tr><td class="fw-bold">Usernames Checked</td><td>${(data.usernames_checked || []).length}</td></tr>`;
    html += `<tr><td class="fw-bold">Scam Accounts Found</td><td><span class="badge ${(data.impersonation_risk || []).length > 0 ? 'bg-danger' : 'bg-success'}">${(data.impersonation_risk || []).length}</span></td></tr>`;
    
    // Show typosquats checked
    if (data.typosquats_checked && data.typosquats_checked.length > 0) {
        html += `<tr><td class="fw-bold">Typosquats Checked</td><td>${data.typosquats_checked.map(t => `<span class="badge bg-secondary me-1">${t}</span>`).join('')}</td></tr>`;
    }
    
    // Show scam category breakdown
    if (data.scam_categories) {
        const categories = data.scam_categories;
        const hasFindings = Object.values(categories).some(arr => arr && arr.length > 0);
        
        if (hasFindings) {
            html += `<tr><td class="fw-bold" colspan="2" style="background: #f8d7da;">⚠️ Scam Account Breakdown</td></tr>`;
            
            const categoryLabels = {
                'fake_support': '🎧 Fake Support Scams',
                'giveaway_scam': '🎁 Giveaway Scams',
                'recovery_scam': '🔓 Recovery Scams',
                'verification_scam': '✓ Verification Scams',
                'admin_impersonation': '👤 Admin Impersonation',
                'other': '⚠️ Other Impersonation'
            };
            
            for (const [category, accounts] of Object.entries(categories)) {
                if (accounts && accounts.length > 0) {
                    html += `<tr><td class="fw-bold">${categoryLabels[category] || category}</td><td>`;
                    accounts.forEach(acc => {
                        html += `<a href="${acc.url}" target="_blank" class="badge bg-danger me-1 mb-1">${acc.username}</a> `;
                    });
                    html += `</td></tr>`;
                }
            }
        }
    }
    
    // List all impersonation risks with details
    if (data.impersonation_risk && data.impersonation_risk.length > 0) {
        html += `<tr><td class="fw-bold" colspan="2">Detailed Threat List</td></tr>`;
        data.impersonation_risk.forEach(imp => {
            html += `<tr><td colspan="2" style="padding-left: 20px;">
                <a href="${imp.url}" target="_blank" class="fw-bold text-danger">${imp.username}</a>
                <br><small class="text-danger">${imp.reason}</small>
            </td></tr>`;
        });
    }
    
    // If no threats found
    if (!data.impersonation_risk || data.impersonation_risk.length === 0) {
        html += `<tr><td colspan="2" style="background: #d1e7dd;">✅ No scam accounts found among ${(data.usernames_checked || []).length} username patterns checked.</td></tr>`;
    }
    
    return html;
}

function formatGitHubDetails(data) {
    let html = '';
    html += `<tr><td class="fw-bold">Code Files Scanned</td><td>${data.total_code_matches || 0}</td></tr>`;
    html += `<tr><td class="fw-bold">Repositories Scanned</td><td>${data.total_repos || 0}</td></tr>`;
    
    // Show typosquats checked
    if (data.typosquats_checked && data.typosquats_checked.length > 0) {
        html += `<tr><td class="fw-bold">Typosquats Checked</td><td>${data.typosquats_checked.map(t => `<span class="badge bg-secondary me-1">${t}</span>`).join('')}</td></tr>`;
    }
    
    // Summary badges
    html += `<tr><td class="fw-bold">Threat Summary</td><td>`;
    html += `<span class="badge ${(data.exposed_secrets || []).length > 0 ? 'bg-danger' : 'bg-success'} me-2">Exposed Secrets: ${(data.exposed_secrets || []).length}</span>`;
    html += `<span class="badge ${(data.phishing_kits || []).length > 0 ? 'bg-danger' : 'bg-success'} me-2">Phishing Kits: ${(data.phishing_kits || []).length}</span>`;
    html += `<span class="badge ${(data.credential_leaks || []).length > 0 ? 'bg-warning' : 'bg-success'}">Credential Leaks: ${(data.credential_leaks || []).length}</span>`;
    html += `</td></tr>`;
    
    // List exposed secrets with details
    if (data.exposed_secrets && data.exposed_secrets.length > 0) {
        html += `<tr><td class="fw-bold" colspan="2" style="background: #f8d7da;">🔑 Exposed Secrets (${data.exposed_secrets.length})</td></tr>`;
        data.exposed_secrets.forEach(secret => {
            html += `<tr><td colspan="2" style="padding-left: 20px;">
                <a href="${secret.url}" target="_blank" class="fw-bold text-danger">${secret.repo}</a>
                <br><small>File: <code>${secret.path}</code></small>
                ${secret.query ? `<br><small class="text-muted">Found via: ${secret.query}</small>` : ''}
            </td></tr>`;
        });
    }
    
    // List phishing kits with details
    if (data.phishing_kits && data.phishing_kits.length > 0) {
        html += `<tr><td class="fw-bold" colspan="2" style="background: #f8d7da;">🎣 Phishing Kits (${data.phishing_kits.length})</td></tr>`;
        data.phishing_kits.forEach(kit => {
            const name = kit.name || kit.repo || kit.full_name || 'Unknown';
            const url = kit.url || kit.html_url || '#';
            html += `<tr><td colspan="2" style="padding-left: 20px;">
                <a href="${url}" target="_blank" class="fw-bold text-danger">${name}</a>
                ${kit.description ? `<br><small class="text-muted">${kit.description.substring(0, 100)}${kit.description.length > 100 ? '...' : ''}</small>` : ''}
                ${kit.threat_type ? `<br><span class="badge bg-danger">${kit.threat_type}</span>` : ''}
                ${kit.stars ? `<small class="ms-2">⭐ ${kit.stars}</small>` : ''}
            </td></tr>`;
        });
    }
    
    // List credential leaks
    if (data.credential_leaks && data.credential_leaks.length > 0) {
        html += `<tr><td class="fw-bold" colspan="2" style="background: #fff3cd;">📧 Credential Leaks (${data.credential_leaks.length})</td></tr>`;
        data.credential_leaks.forEach(leak => {
            html += `<tr><td colspan="2" style="padding-left: 20px;">
                <a href="${leak.url}" target="_blank" class="fw-bold text-warning">${leak.repo}</a>
                <br><small>File: <code>${leak.path}</code></small>
            </td></tr>`;
        });
    }
    
    // If no threats found
    if ((!data.exposed_secrets || data.exposed_secrets.length === 0) && 
        (!data.phishing_kits || data.phishing_kits.length === 0) &&
        (!data.credential_leaks || data.credential_leaks.length === 0)) {
        html += `<tr><td colspan="2" style="background: #d1e7dd;">✅ No exposed secrets, phishing kits, or credential leaks found.</td></tr>`;
    }
    
    return html;
}

// Download social media report
function downloadSMReport() {
    if (!window.currentSMResults) {
        alert('No social media monitoring report to download');
        return;
    }
    
    const link = document.createElement('a');
    link.href = `/api/social-media/download/${currentTaskId}`;
    link.download = `social_media_report_${window.currentSMResults.brand_domain || 'report'}.html`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
}


