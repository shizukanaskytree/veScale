document.addEventListener('DOMContentLoaded', function () {
    const divider = document.getElementById('dragMe');
    const leftPanel = divider.previousElementSibling;
    const rightPanel = divider.nextElementSibling;

    let isDragging = false;

    divider.addEventListener('mousedown', function (e) {
        isDragging = true;
    });

    document.addEventListener('mousemove', function (e) {
        if (!isDragging) return;

        const offsetRight = document.body.clientWidth - e.clientX;
        leftPanel.style.width = `${e.clientX}px`;
        rightPanel.style.width = `${offsetRight}px`;
    });

    document.addEventListener('mouseup', function () {
        isDragging = false;
    });
});