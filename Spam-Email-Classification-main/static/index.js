// Theme toggle — persisted in localStorage
const body = document.body;
const themeToggle = document.getElementById('theme-toggle');

// Apply saved theme on load
if (localStorage.getItem('theme') === 'light') {
    body.classList.add('light-mode');
}

if (themeToggle) {
    themeToggle.addEventListener('click', function () {
        body.classList.toggle('light-mode');
        localStorage.setItem('theme', body.classList.contains('light-mode') ? 'light' : 'dark');
    });
}
