// API base URL
const API_BASE_URL = 'http://localhost:8000/api';

// DOM Elements
const lmssStatus = document.getElementById('lmss-status');
const updateLmssButton = document.getElementById('update-lmss');
const lmssProgress = document.getElementById('lmss-progress');
const lmssStatistics = document.getElementById('lmss-statistics');
const downloadIndexButton = document.getElementById('download-index');
const downloadGraphButton = document.getElementById('download-graph');
const textInput = document.getElementById('text-input');
const fileInput = document.getElementById('file-input');
const lmssClassSelection = document.getElementById('lmss-class-selection');
const processDocumentButton = document.getElementById('process-document');
const resultsTable = document.getElementById('results-table');
const downloadJsonButton = document.getElementById('download-json');
const searchInput = document.getElementById('search-input');
const classFilter = document.getElementById('class-filter');
const searchResults = document.getElementById('search-results');

// Fetch LMSS status
async function fetchLmssStatus() {
    const response = await fetch(`${API_BASE_URL}/lmss/status`);
    const data = await response.json();
    lmssStatus.textContent = `LMSS Status: ${data.status}`;
    if (data.status === 'ready') {
        lmssStatus.style.color = 'green';
        lmssStatus.textContent = 'LMSS Ready!';
        updateLmssButton.disabled = false;
        processDocumentButton.disabled = false;
        downloadIndexButton.disabled = false;
        downloadGraphButton.disabled = false;
        fetchLmssClasses(); // Activate class filter elements
    } else if (data.status === 'processing') {
        lmssStatus.style.color = 'orange';
        lmssStatus.textContent = 'LMSS Status: Processing...';
        updateLmssButton.disabled = true;
        processDocumentButton.disabled = true;
        downloadIndexButton.disabled = true;
        downloadGraphButton.disabled = true;
    } else {
        lmssStatus.style.color = 'red';
        lmssStatus.textContent = 'Get LMSS';
        updateLmssButton.disabled = false;
        processDocumentButton.disabled = true;
        downloadIndexButton.disabled = true;
        downloadGraphButton.disabled = true;
    }
}

// Update LMSS
async function updateLmss() {
    lmssProgress.classList.remove('hidden');
    const response = await fetch(`${API_BASE_URL}/lmss/update`, { method: 'POST' });
    const data = await response.json();
    lmssProgress.classList.add('hidden');
    fetchLmssStatus();
}

// Fetch LMSS statistics
async function fetchLmssStatistics() {
    const response = await fetch(`${API_BASE_URL}/lmss/statistics`);
    const data = await response.json();
    lmssStatistics.innerHTML = `
        <p>Branches: ${data.branches}</p>
        <p>Classes: ${data.classes}</p>
        <p>Attributes with embeddings: ${data.attributes_with_embeddings}</p>
    `;
}

// Download LMSS files
async function downloadLmssFile(fileType) {
    window.location.href = `${API_BASE_URL}/lmss/download/${fileType}`;
}

// Fetch LMSS classes
async function fetchLmssClasses() {
    const response = await fetch(`${API_BASE_URL}/lmss/classes`);
    const classes = await response.json();
    lmssClassSelection.innerHTML = classes.map(cls => `
        <label>
            <input type="checkbox" name="lmss-class" value="${cls.iri}">
            ${cls.label} (${cls.entities_count})
        </label>
    `).join('<br>');
    
    // Populate class filter dropdown
    classFilter.innerHTML = `
        <option value="">All Classes</option>
        ${classes.map(cls => `<option value="${cls.iri}">${cls.label}</option>`).join('')}
    `;
}

// Process document
async function processDocument() {
    const text = textInput.value;
    const selectedClasses = Array.from(document.querySelectorAll('input[name="lmss-class"]:checked')).map(input => input.value);
    const response = await fetch(`${API_BASE_URL}/document/process`, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ text, selectedClasses })
    });
    const data = await response.json();
    displayResults(data.results);
}

// Display results
function displayResults(results) {
    resultsTable.innerHTML = `
        <table>
            <tr>
                <th>Start</th>
                <th>End</th>
                <th>Text</th>
                <th>Branch</th>
                <th>Label</th>
                <th>Score</th>
                <th>IRI</th>
            </tr>
            ${results.map(entity => `
                <tr>
                    <td>${entity.start}</td>
                    <td>${entity.end}</td>
                    <td>${entity.text}</td>
                    <td>${entity.match.branch || ''}</td>
                    <td>${entity.match.label || ''}</td>
                    <td>${entity.match.similarity?.toFixed(2) || ''}</td>
                    <td>${entity.match.iri || ''}</td>
                </tr>
            `).join('')}
        </table>
    `;
    downloadJsonButton.classList.remove('hidden');
}

// Search LMSS
async function searchLmss() {
    const query = searchInput.value;
    const classFilterValue = classFilter.value;
    const response = await fetch(`${API_BASE_URL}/search?query=${query}&class_filter=${classFilterValue}`);
    const data = await response.json();
    displaySearchResults(data.results);
}

// Display search results
function displaySearchResults(results) {
    searchResults.innerHTML = `
        <table>
            <tr>
                <th>Branch</th>
                <th>Label</th>
                <th>Score</th>
                <th>IRI</th>
            </tr>
            ${results.map(result => `
                <tr>
                    <td>${result.branch || ''}</td>
                    <td>${result.label}</td>
                    <td>${result.score.toFixed(2)}</td>
                    <td>${result.iri}</td>
                </tr>
            `).join('')}
        </table>
    `;
}

// Handle file upload
fileInput.addEventListener('change', async (event) => {
    const file = event.target.files[0];
    if (file) {
        const formData = new FormData();
        formData.append('file', file);
        const response = await fetch(`${API_BASE_URL}/document/upload`, {
            method: 'POST',
            body: formData
        });
        const data = await response.json();
        textInput.value = data.text;
    }
});

// Event listeners
updateLmssButton.addEventListener('click', updateLmss);
downloadIndexButton.addEventListener('click', () => downloadLmssFile('index'));
downloadGraphButton.addEventListener('click', () => downloadLmssFile('graph'));
processDocumentButton.addEventListener('click', processDocument);
searchInput.addEventListener('input', searchLmss);
classFilter.addEventListener('change', searchLmss);
downloadJsonButton.addEventListener('click', () => {
    const dataStr = JSON.stringify(JSON.parse(resultsTable.innerHTML), null, 2);
    const dataUri = 'data:application/json;charset=utf-8,'+ encodeURIComponent(dataStr);
    const exportFileDefaultName = 'classification_results.json';
    const linkElement = document.createElement('a');
    linkElement.setAttribute('href', dataUri);
    linkElement.setAttribute('download', exportFileDefaultName);
    linkElement.click();
});

// Initial fetch calls
fetchLmssStatus();
fetchLmssStatistics();
fetchLmssClasses();

// Periodic status check
setInterval(fetchLmssStatus, 5000);