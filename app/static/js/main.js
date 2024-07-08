document.addEventListener('DOMContentLoaded', function() {
    const searchInput = document.getElementById('search-input');
    const searchButton = document.getElementById('search-button');
    const searchResults = document.getElementById('search-results');
    const inputText = document.getElementById('input-text');
    const fileUpload = document.getElementById('file-upload');
    const classSelect = document.getElementById('class-select');
    const processButton = document.getElementById('process-button');
    const resultsTable = document.getElementById('results-table').getElementsByTagName('tbody')[0];
    const downloadButton = document.getElementById('download-button');

    // Load top classes
    axios.get('/top_classes')
        .then(function (response) {
            response.data.top_classes.forEach(function(cls) {
                const option = document.createElement('option');
                option.value = cls.iri;
                option.textContent = cls.label;
                classSelect.appendChild(option);
            });
        })
        .catch(function (error) {
            console.error('Error loading top classes:', error);
        });

    // Search functionality
    searchButton.addEventListener('click', function() {
        const query = searchInput.value;
        const selectedBranches = Array.from(classSelect.selectedOptions).map(option => option.value);
        
        axios.post('/search', {
            query: query,
            selected_branches: selectedBranches
        })
        .then(function (response) {
            searchResults.innerHTML = '';
            response.data.results.forEach(function(result) {
                const div = document.createElement('div');
                div.textContent = `${result.label} (${result.score.toFixed(2)})`;
                searchResults.appendChild(div);
            });
        })
        .catch(function (error) {
            console.error('Error:', error);
            searchResults.innerHTML = 'An error occurred while searching.';
        });
    });

    // Process text
    processButton.addEventListener('click', function() {
        const text = inputText.value;
        const selectedClasses = Array.from(classSelect.selectedOptions).map(option => option.value);
        
        if (text) {
            processText(text, selectedClasses);
        } else if (fileUpload.files.length > 0) {
            const file = fileUpload.files[0];
            const formData = new FormData();
            formData.append('file', file);
            formData.append('selected_classes', JSON.stringify(selectedClasses));
            
            axios.post('/upload', formData, {
                headers: {
                    'Content-Type': 'multipart/form-data'
                }
            })
            .then(handleResponse)
            .catch(handleError);
        } else {
            alert('Please enter text or upload a file.');
        }
    });

    function processText(text, selectedClasses) {
        axios.post('/process', {
            text: text,
            selected_classes: selectedClasses
        })
        .then(handleResponse)
        .catch(handleError);
    }

    function handleResponse(response) {
        resultsTable.innerHTML = '';
        response.data.entities.forEach(function(entity) {
            const row = resultsTable.insertRow();
            row.insertCell(0).textContent = entity.text;
            row.insertCell(1).textContent = entity.start;
            row.insertCell(2).textContent = entity.end;
            row.insertCell(3).textContent = entity.match.label || 'N/A';
            row.insertCell(4).textContent = entity.match.iri || 'N/A';
            row.insertCell(5).textContent = entity.match.match_type;
            row.insertCell(6).textContent = entity.match.similarity ? entity.match.similarity.toFixed(2) : 'N/A';
        });
        downloadButton.style.display = 'block';
    }

    function handleError(error) {
        console.error('Error:', error);
        alert('An error occurred while processing the text.');
    }

    // Download results
    downloadButton.addEventListener('click', function() {
        const rows = Array.from(resultsTable.rows);
        const data = rows.map(row => Array.from(row.cells).map(cell => cell.textContent));
        const csv = [
            ['Text', 'Start', 'End', 'Matching Class', 'IRI', 'Match Type', 'Confidence'],
            ...data
        ].map(row => row.join(',')).join('\n');
        
        const blob = new Blob([csv], { type: 'text/csv' });
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.setAttribute('hidden', '');
        a.setAttribute('href', url);
        a.setAttribute('download', 'results.csv');
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
    });
});