document.addEventListener('DOMContentLoaded', function () {
    const horizontalDivider = document.getElementById('dragMeHorizontal');
    const upperPanel = horizontalDivider.previousElementSibling;
    const lowerPanel = horizontalDivider.nextElementSibling;

    const verticalDivider = document.getElementById('dragMe');
    const leftPanel = verticalDivider.previousElementSibling;
    const rightPanel = verticalDivider.nextElementSibling;

    let isDraggingHorizontal = false;
    let isDraggingVertical = false;

    horizontalDivider.addEventListener('mousedown', function (e) {
        isDraggingHorizontal = true;
    });

    verticalDivider.addEventListener('mousedown', function (e) {
        isDraggingVertical = true;
    });

    document.addEventListener('mousemove', function (e) {
        if (isDraggingHorizontal) {
            const containerRect = upperPanel.parentElement.getBoundingClientRect();
            const upperPanelHeight = e.clientY - containerRect.top;
            const lowerPanelHeight = containerRect.bottom - e.clientY;

            upperPanel.style.height = `${upperPanelHeight}px`;
            lowerPanel.style.height = `${lowerPanelHeight}px`;
        }

        if (isDraggingVertical) {
            const containerRect = leftPanel.parentElement.getBoundingClientRect();
            const leftPanelWidth = e.clientX - containerRect.left;
            const rightPanelWidth = containerRect.right - e.clientX;

            leftPanel.style.width = `${leftPanelWidth}px`;
            rightPanel.style.width = `${rightPanelWidth}px`;
        }
    });

    document.addEventListener('mouseup', function () {
        isDraggingHorizontal = false;
        isDraggingVertical = false;
    });
});