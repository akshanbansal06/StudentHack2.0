form.addEventListener('submit', function (e) {
    e.preventDefault(); // Prevent form default submission

    // Validate all required fields
    let isValid = true;
    requiredFields.forEach(field => {
        if (!validateField(field)) {
            isValid = false;
        }
    });

    if (isValid) {
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

        // Submit the form without awaiting the response
        fetch('/submit_data', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(formData)
        });

        // Redirect immediately after firing the request
        window.location.href = "/loadingpage";
    }
});