document.getElementById("fileUpload").addEventListener("change", (event) => {
    const file = event.target.files[0];
    if (file) {
        // Inform the user the file is ready for upload
        document.getElementById("resultContainer").textContent = "File ready for upload.";
    }
});

async function submitFile() {
    const fileInput = document.getElementById("fileUpload");
    const resultContainer = document.getElementById("resultContainer");

    // Check if a file has been selected
    if (fileInput.files.length === 0) {
        resultContainer.textContent = "Please upload an image file.";
        return;
    }

    // Prepare the form data with the selected file
    const formData = new FormData();
    formData.append("file", fileInput.files[0]);

    try {
        // Send the file to the server for prediction
        const response = await fetch('/predict', {
            method: 'POST',
            body: formData,
        });

        const result = await response.json();

        if (result.error) {
            resultContainer.textContent = `Error: ${result.error}`;
        } else {
            // Display the prediction result
            resultContainer.textContent = `Prediction: ${result.prediction}`;
        }
    } catch (error) {
        console.error('Error during file upload:', error);
        resultContainer.textContent = "An error occurred. Please try again.";
    }
}
