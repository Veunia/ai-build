document.getElementById('workflow-form').addEventListener('submit', async (e) => {
    e.preventDefault();
    const form = e.target;
    const data = new FormData(form);

    const response = await fetch('http://localhost:8000/generate', {
        method: 'POST',
        body: data
    });

    const result = await response.json();
    document.getElementById('result').textContent = JSON.stringify(result, null, 2);
});
