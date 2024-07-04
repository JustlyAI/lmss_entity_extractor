document.addEventListener('DOMContentLoaded', function() {
    const submitBtn = document.getElementById('submit-btn');
    const inputText = document.getElementById('input-text');
    const outputTable = document.getElementById('output-table').getElementsByTagName('tbody')[0];

    submitBtn.addEventListener('click', function() {
        const text = inputText.value;
        
        axios.post('/recognize', {
            text: text
        })
        .then(function (response) {
            outputTable.innerHTML = '';
            response.data.entities.forEach(function(entity) {
                const row = outputTable.insertRow();
                row.insertCell(0).textContent = entity.start;
                row.insertCell(1).textContent = entity.end;
                row.insertCell(2).textContent = entity.text;
                row.insertCell(3).textContent = entity.entity;
                row.insertCell(4).textContent = entity.iri;
            });
        })
        .catch(function (error) {
            console.error('Error:', error);
            alert('An error occurred while processing your request.');
        });
    });
});