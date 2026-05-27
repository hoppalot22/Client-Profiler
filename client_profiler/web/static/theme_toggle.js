(function () {
  const button = document.querySelector('[data-theme-toggle]');
  const root = document.documentElement;
  if (!button || !root) {
    return;
  }

  function getTheme() {
    return root.dataset.theme === 'dark' ? 'dark' : 'light';
  }

  function setTheme(theme) {
    root.dataset.theme = theme;
    localStorage.setItem('cp-theme', theme);
    button.setAttribute('aria-pressed', String(theme === 'dark'));
    button.textContent = theme === 'dark' ? 'Light mode' : 'Dark mode';
  }

  setTheme(getTheme());

  button.addEventListener('click', () => {
    setTheme(getTheme() === 'dark' ? 'light' : 'dark');
  });
})();