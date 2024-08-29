document.addEventListener('DOMContentLoaded', function () {
    const horizontalDivider = document.getElementById('dragMeHorizontal');
    const upperPanel = horizontalDivider.previousElementSibling;
    const lowerPanel = horizontalDivider.nextElementSibling;

    let isDraggingHorizontal = false;

    horizontalDivider.addEventListener('mousedown', function (e) {
        isDraggingHorizontal = true;
    });

    document.addEventListener('mousemove', function (e) {
        if (!isDraggingHorizontal) return;

        const containerRect = upperPanel.parentElement.getBoundingClientRect();
        const upperPanelHeight = e.clientY - containerRect.top;
        const lowerPanelHeight = containerRect.bottom - e.clientY;

        upperPanel.style.height = `${upperPanelHeight}px`;
        lowerPanel.style.height = `${lowerPanelHeight}px`;
    });

    document.addEventListener('mouseup', function () {
        isDraggingHorizontal = false;
    });
});