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
const selectAllClassesButton = document.getElementById('select-all-classes');
const clearAllClassesButton = document.getElementById('clear-all-classes');
const processDocumentButton = document.getElementById('process-document');
const resultsTable = document.getElementById('results-table');
const downloadJsonButton = document.getElementById('download-json');
const searchInput = document.getElementById('search-input');
const classFilter = document.getElementById('class-filter');
const searchResults = document.getElementById('search-results');
const textInputMessage = document.getElementById('text-input-message');
const fileUploadMessage = document.getElementById('file-upload-message');

let classificationResults = null;

async function fetchLmssStatus() {
    const response = await fetch(`${API_BASE_URL}/lmss/status`);
    const data = await response.json();
    lmssStatus.textContent = `LMSS Status: ${data.status}`;
    if (data.status === 'ready') {
        lmssStatus.style.color = 'green';
        lmssStatus.textContent = 'LMSS Ready!';
        updateLmssButton.disabled = false;
        downloadIndexButton.disabled = false;
        downloadGraphButton.disabled = false;
        fetchLmssClasses();
    } else if (data.status === 'processing') {
        lmssStatus.style.color = 'orange';
        lmssStatus.textContent = 'LMSS Status: Processing...';
        updateLmssButton.disabled = true;
        downloadIndexButton.disabled = true;
        downloadGraphButton.disabled = true;
    } else {
        lmssStatus.style.color = 'red';
        lmssStatus.textContent = 'Get LMSS';
        updateLmssButton.disabled = false;
        downloadIndexButton.disabled = true;
        downloadGraphButton.disabled = true;
    }
    updateProcessDocumentButton();
}

async function updateLmss() {
    lmssProgress.classList.remove('hidden');
    const response = await fetch(`${API_BASE_URL}/lmss/update`, { method: 'POST' });
    const data = await response.json();
    lmssProgress.classList.add('hidden');
    fetchLmssStatus();
}

async function fetchLmssStatistics() {
    const response = await fetch(`${API_BASE_URL}/lmss/statistics`);
    const data = await response.json();
    lmssStatistics.innerHTML = `
        <p>Branches: ${data.branches}</p>
        <p>Classes: ${data.classes}</p>
        <p>Embedded Attributes: ${data.attributes_with_embeddings}</p>
    `;
}

async function downloadLmssFile(fileType) {
    window.location.href = `${API_BASE_URL}/lmss/download/${fileType}`;
}

async function fetchLmssClasses() {
    const response = await fetch(`${API_BASE_URL}/lmss/classes`);
    const classes = await response.json();
    
    const columns = 3; // Changed from Math.ceil(classes.length / 6) to 3
    const rowsPerColumn = 8; // New constant for rows per column
    let html = '';
    for (let i = 0; i < columns; i++) {
        html += '<div class="class-column">';
        for (let j = i * rowsPerColumn; j < Math.min((i + 1) * rowsPerColumn, classes.length); j++) {
            const cls = classes[j];
            if (cls) { // Add this check to avoid errors if there are fewer classes than expected
                html += `
                    <div class="class-item">
                        <input type="checkbox" name="lmss-class" value="${cls.iri}" checked>
                        <label>${cls.label} (${cls.entities_count})</label>
                    </div>
                `;
            }
        }
        html += '</div>';
    }
    lmssClassSelection.innerHTML = html;
    
    // Populate class filter dropdown
    classFilter.innerHTML = `
        <option value="">All Classes</option>
        ${classes.map(cls => `<option value="${cls.iri}">${cls.label}</option>`).join('')}
    `;
}

function selectAllClasses() {
    const checkboxes = document.querySelectorAll('input[name="lmss-class"]');
    checkboxes.forEach(checkbox => checkbox.checked = true);
}

function clearAllClasses() {
    const checkboxes = document.querySelectorAll('input[name="lmss-class"]');
    checkboxes.forEach(checkbox => checkbox.checked = false);
}

async function processDocument() {
    const text = textInput.value;
    const selectedClasses = Array.from(document.querySelectorAll('input[name="lmss-class"]:checked')).map(input => input.value);
    
    // Show loading state
    resultsTable.innerHTML = '<p>Processing document...</p>';
    
    try {
        const response = await fetch(`${API_BASE_URL}/document/process`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ text, selected_classes: selectedClasses })
        });

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        const data = await response.json();
        
        if (!data || !data.results) {
            throw new Error('Invalid response data');
        }

        classificationResults = data;
        displayResults(classificationResults.results);
    } catch (error) {
        console.error('Error processing document:', error);
        resultsTable.innerHTML = `<p>Error processing document: ${error.message}</p>`;
    }
}

function displayResults(results) {
    if (!Array.isArray(results) || results.length === 0) {
        resultsTable.innerHTML = '<p>No results found</p>';
        return;
    }

    resultsTable.innerHTML = `
        <table>
            <tr>
                <th>Start</th>
                <th>End</th>
                <th>Text</th>
                <th>Branch</th>
                <th>Label</th>
                <th>Score + Explain</th>
                <th>IRI</th>
            </tr>
            ${results.map(entity => `
                <tr>
                    <td>${entity.start}</td>
                    <td>${entity.end}</td>
                    <td>${entity.text}</td>
                    <td>${entity.branch || 'N/A'}</td>
                    <td>${entity.label || 'N/A'}</td>
                    <td>${entity.score ? entity.score.toFixed(2) : 'N/A'}</td>
                    <td>${entity.iri || 'N/A'}</td>
                </tr>
            `).join('')}
        </table>
    `;
    downloadJsonButton.classList.remove('hidden');
}

async function searchLmss() {
    const query = searchInput.value;
    const classFilterValue = classFilter.value;
    const response = await fetch(`${API_BASE_URL}/search?query=${query}&class_filter=${classFilterValue}`);
    const data = await response.json();
    displaySearchResults(data.results);
}

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

async function handleFileUpload(event) {
    const file = event.target.files[0];
    if (file) {
        const formData = new FormData();
        formData.append('file', file);
        try {
            const response = await fetch(`${API_BASE_URL}/document/upload`, {
                method: 'POST',
                body: formData
            });
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            const data = await response.json();
            textInput.value = data.text;
            fileUploadMessage.textContent = 'File uploaded successfully';
            updateProcessDocumentButton();
        } catch (error) {
            console.error('Error uploading file:', error);
            fileUploadMessage.textContent = `Error uploading file: ${error.message}`;
        }
    }
}

function updateProcessDocumentButton() {
    processDocumentButton.disabled = !(lmssStatus.textContent === 'LMSS Ready!' && (textInput.value.trim() !== '' || fileInput.files.length > 0));
}

function handleTextInput(event) {
    if (event.key === 'Enter') {
        textInputMessage.textContent = 'Your text has been received';
    }
    updateProcessDocumentButton();
}

function downloadJsonResults() {
    if (classificationResults) {
        const dataStr = JSON.stringify(classificationResults, null, 2);
        const dataUri = 'data:application/json;charset=utf-8,'+ encodeURIComponent(dataStr);
        const exportFileDefaultName = 'classification_results.json';
        const linkElement = document.createElement('a');
        linkElement.setAttribute('href', dataUri);
        linkElement.setAttribute('download', exportFileDefaultName);
        linkElement.click();
    }
}

// Event listeners
updateLmssButton.addEventListener('click', updateLmss);
downloadIndexButton.addEventListener('click', () => downloadLmssFile('index'));
downloadGraphButton.addEventListener('click', () => downloadLmssFile('graph'));
selectAllClassesButton.addEventListener('click', selectAllClasses);
clearAllClassesButton.addEventListener('click', clearAllClasses);
processDocumentButton.addEventListener('click', processDocument);
searchInput.addEventListener('input', searchLmss);
classFilter.addEventListener('change', searchLmss);
downloadJsonButton.addEventListener('click', downloadJsonResults);
fileInput.addEventListener('change', handleFileUpload);
textInput.addEventListener('input', updateProcessDocumentButton);
textInput.addEventListener('keypress', handleTextInput);

// Initial fetch calls
fetchLmssStatus();
fetchLmssStatistics();
fetchLmssClasses();

// Periodic status check
setInterval(fetchLmssStatus, 5000);