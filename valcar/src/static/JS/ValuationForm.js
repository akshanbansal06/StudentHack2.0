// Set max year to current year + 1
const currentYear = new Date().getFullYear();
document.getElementById('year').setAttribute('max', currentYear + 1);

// Form validation
const form = document.getElementById('carValuationForm');
const submitButton = document.getElementById('submitButton');
const requiredFields = form.querySelectorAll('[required]');

// Validation function
function validateField(field) {
    const errorElement = document.getElementById(field.id + 'Error');
    if (errorElement) {
        if (!field.validity.valid) {
            errorElement.style.display = 'block';
            return false;
        } else {
            errorElement.style.display = 'none';
            return true;
        }
    }
    return field.validity.valid;
}

// Add blur and input event listeners to all required fields
requiredFields.forEach(field => {
    field.addEventListener('blur', () => validateField(field));
    field.addEventListener('input', () => validateField(field));
});

form.addEventListener('submit', async function (e) {
    e.preventDefault();

    // Validate all required fields
    let isValid = true;
    requiredFields.forEach(field => {
        if (!validateField(field)) {
            isValid = false;
        }
    });

    if (isValid) {
        // Show loading indicator or disable button
        submitButton.disabled = true;
        if (submitButton.textContent) {
            submitButton.textContent = 'Processing...';
        }

        // Collect data from the form
        const formData = {
            make: document.getElementById("make").value,
            model: document.getElementById("model").value,
            year: document.getElementById("year").value,
            trim: document.getElementById("trim").value,
            mileage: document.getElementById("mileage").value,
            fuel: document.getElementById("fuel").value,
            transmission: document.getElementById("transmission").value,
            color: document.getElementById("color").value
        };

        try {
            const response = await fetch('/submit_data', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(formData)
            });
            
            if (!response.ok) {
                throw new Error('Network response was not ok');
            }
            
            const result = await response.json();
            
            // Redirect based on server response
            if (result.redirect) {
                window.location.href = result.redirect;
            } else {
                // Default redirect to loading page
                window.location.href = "/loadingpage";
            }

        } catch (error) {
            console.error('Error:', error);
            alert('An error occurred while submitting the data.');
            
            // Re-enable button on error
            submitButton.disabled = false;
            if (submitButton.textContent) {
                submitButton.textContent = 'Submit';
            }
        }
    }
});