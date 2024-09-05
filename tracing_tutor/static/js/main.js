let currentStep = 0;
let totalSteps = 200;  // Adjust total steps dynamically if needed
let sliderIsBeingDragged = false;

// Function to initialize the slider range based on currentStep and totalSteps
function initializeSlider() {
    const slider = document.getElementById('step-slider');
    slider.min = currentStep;  // Set the starting point to currentStep
    slider.max = totalSteps;   // Set the ending point to totalSteps
    slider.value = currentStep; // Set the initial value to currentStep
}

// Call this function when the page loads to initialize the slider
initializeSlider();

document.getElementById('step-slider').addEventListener('input', (event) => {
    sliderIsBeingDragged = true;
    currentStep = parseInt(event.target.value);
    updateStep(false);  // Don't reset the slider while dragging
});

document.getElementById('step-slider').addEventListener('change', (event) => {
    currentStep = parseInt(event.target.value);
    sliderIsBeingDragged = false;
    updateStep(true);
});

document.getElementById('prev-btn').addEventListener('click', () => {
    if (currentStep > parseInt(document.getElementById('step-slider').min)) {
        currentStep--;
        updateStep(true);
    }
});

document.getElementById('next-btn').addEventListener('click', () => {
    if (currentStep < totalSteps) {
        currentStep++;
        updateStep(true);
    }
});

function updateStep(fetchNewContent = true) {
    if (fetchNewContent) {
        // Fetch the previous log content
        fetch(`/steps/${currentStep - 1}`)
            .then(response => response.json())
            .then(data => {
                const prevLogDisplay = document.getElementById('prev-log-display');
                if (!data.error) {
                    prevLogDisplay.innerHTML = Prism.highlight(data.content, Prism.languages.python, 'python');
                } else {
                    prevLogDisplay.textContent = "No previous log available";
                }
            })
            .catch(err => console.error(err));

        // Fetch the next log content
        fetch(`/steps/${currentStep}`)
            .then(response => response.json())
            .then(data => {
                const nextLogDisplay = document.getElementById('next-log-display');
                if (!data.error) {
                    nextLogDisplay.innerHTML = Prism.highlight(data.content, Prism.languages.python, 'python');
                } else {
                    nextLogDisplay.textContent = "No next log available";
                }
            })
            .catch(err => console.error(err));

        document.getElementById('step-indicator').textContent = `Step ${currentStep} of ${totalSteps}`;
    }

    // Ensure slider reflects the current step
    if (!sliderIsBeingDragged) {
        document.getElementById('step-slider').value = currentStep;
    }
}

// Initial load
updateStep();