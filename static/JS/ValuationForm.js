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

requiredFields.forEach(field => {
    field.addEventListener('blur', () => validateField(field));
    field.addEventListener('input', () => validateField(field));
});


// Form submission
form.addEventListener('submit', function(e) {
    e.preventDefault();

    // Validate all required fields
    let isValid = true;
    requiredFields.forEach(field => {
        if (!validateField(field)) {
            isValid = false;
        }
    });

    
    if (isValid) {
        // Collect form data
        const make = document.getElementById('make').value;
        const model = document.getElementById('model').value;
        const year = document.getElementById('year').value;
        const trim = document.getElementById('trim').value;
        const mileage = document.getElementById('mileage').value;
        const fuel = document.getElementById('fuel').value;
        const transmission = document.getElementById('transmission').value;
        const color = document.getElementById('color').value;

        // Show loading state on the submit button
        submitButton.setAttribute('data-loading', 'true');
        submitButton.disabled = true;

        // Prepare form data to be sent to the server
        const formData = {
            make,
            model,
            year,
            trim,
            mileage,
            fuel,
            transmission,
            color
        };

        // Send form data to the Flask backend using Fetch API
        fetch('http://127.0.0.1:5000/get_valuation', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(formData)
        })
        .then(response => response.json())
        .then(data => {
            // Hide loading state
            submitButton.removeAttribute('data-loading');
            submitButton.disabled = false;

            // Handle the response data (e.g., display the valuation result)
            const valuationResult = document.getElementById('valuationResult');
            const valuationDetails = document.getElementById('valuationDetails');
            valuationResult.style.display = 'block';
            valuationDetails.innerHTML = `
                <p>Make: ${data.make}</p>
                <p>Model: ${data.model}</p>
                <p>Year: ${data.year}</p>
                <p>Mileage: ${data.mileage}</p>
                <p>Fuel: ${data.fuel}</p>
                <p>Transmission: ${data.transmission}</p>
                <p>Color: ${data.color}</p>
                <h3>Estimated Valuation: $${data.valuation.toFixed(2)}</h3>
            `;
        })
        .catch(error => {
            // Hide loading state in case of an error
            submitButton.removeAttribute('data-loading');
            submitButton.disabled = false;

            // Handle the error (e.g., display a message)
            console.error('Error:', error);
            alert('There was an error submitting your form. Please try again later.');
        });
    } else {
        // Form is not valid, do not submit
        console.log('Form is not valid');
    }
});
