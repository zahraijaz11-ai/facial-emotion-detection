
// ======================================================
// EMOTION AI - FRONTEND JAVASCRIPT
// ======================================================


// ------------------------------
// HTML ELEMENTS
// ------------------------------

const imageInput = document.getElementById("imageInput");

const uploadBox = document.getElementById("uploadBox");

const uploadContent = document.getElementById("uploadContent");
const imagePreviewState = document.getElementById("imagePreviewState");

const previewImage = document.getElementById("previewImage");

const fileName = document.getElementById("fileName");
const selectedFileName = document.getElementById("selectedFileName");

const removeButton = document.getElementById("removeButton");

const analyzeButton = document.getElementById("analyzeButton");
const buttonText = document.getElementById("buttonText");
const buttonLoader = document.getElementById("buttonLoader");

const errorMessage = document.getElementById("errorMessage");

const result = document.getElementById("result");

const emotion = document.getElementById("emotion");
const emotionIcon = document.getElementById("emotionIcon");

const confidence = document.getElementById("confidence");
const confidenceFill = document.getElementById("confidenceFill");


// ------------------------------
// SELECTED FILE
// ------------------------------

let selectedFile = null;


// ------------------------------
// EMOTION ICONS
// ------------------------------

const emotionIcons = {
    angry: "😠",
    disgust: "🤢",
    fear: "😨",
    happy: "😊",
    neutral: "😐",
    sad: "😢",
    surprise: "😲"
};


// ------------------------------
// ERROR
// ------------------------------

function showError(message) {

    errorMessage.textContent = message;
    errorMessage.style.display = "block";
}


function hideError() {

    errorMessage.textContent = "";
    errorMessage.style.display = "none";
}


// ------------------------------
// DISPLAY IMAGE INSIDE UPLOAD BOX
// ------------------------------

function displayImage(file) {

    hideError();

    if (!file) {
        return;
    }


    // Check file type
    if (!file.type.startsWith("image/")) {

        showError("Please select a valid image file.");

        return;
    }


    // Save selected file
    selectedFile = file;


    // Create preview URL
    const imageURL = URL.createObjectURL(file);

    previewImage.src = imageURL;


    // Show filename
    selectedFileName.textContent = file.name;


    // SWITCH UI
    uploadContent.style.display = "none";

    imagePreviewState.style.display = "block";


    // Hide previous result
    result.style.display = "none";


    // Reset confidence
    confidence.textContent = "0%";
    confidenceFill.style.width = "0%";


    console.log("Image selected:", file.name);
}


// ------------------------------
// CHOOSE IMAGE
// ------------------------------

imageInput.addEventListener("change", function () {

    const file = imageInput.files[0];

    displayImage(file);

});


// ------------------------------
// DRAG OVER
// ------------------------------

uploadBox.addEventListener("dragover", function (event) {

    event.preventDefault();

    uploadBox.classList.add("dragging");

});


// ------------------------------
// DRAG LEAVE
// ------------------------------

uploadBox.addEventListener("dragleave", function () {

    uploadBox.classList.remove("dragging");

});


// ------------------------------
// DROP IMAGE
// ------------------------------

uploadBox.addEventListener("drop", function (event) {

    event.preventDefault();

    uploadBox.classList.remove("dragging");


    const file = event.dataTransfer.files[0];

    displayImage(file);

});


// ------------------------------
// CTRL + V
// ------------------------------

document.addEventListener("paste", function (event) {

    const items = event.clipboardData.items;


    for (const item of items) {

        if (item.type.startsWith("image/")) {

            const file = item.getAsFile();

            displayImage(file);

            break;
        }
    }

});


// ------------------------------
// REMOVE IMAGE
// ------------------------------

removeButton.addEventListener("click", function () {

    selectedFile = null;

    imageInput.value = "";

    previewImage.src = "";

    selectedFileName.textContent = "";


    // Switch back to upload screen
    imagePreviewState.style.display = "none";

    uploadContent.style.display = "block";


    // Hide result
    result.style.display = "none";


    // Reset confidence
    confidence.textContent = "0%";

    confidenceFill.style.width = "0%";


    hideError();

});


// ------------------------------
// ANALYZE
// ------------------------------

analyzeButton.addEventListener("click", async function () {

    hideError();


    if (!selectedFile) {

        showError("Please select an image first.");

        return;
    }


    // Loading state
    analyzeButton.disabled = true;

    buttonText.textContent = "Analyzing...";

    buttonLoader.style.display = "inline-block";

    result.style.display = "none";


    // Form data
    const formData = new FormData();

    formData.append("file", selectedFile);


    console.log("Sending image to FastAPI...");
    console.log("File:", selectedFile.name);


    try {

        const response = await fetch(
            "http://127.0.0.1:8000/api/emotion/predict",
            {
                method: "POST",
                body: formData
            }
        );


        console.log("HTTP status:", response.status);


        const data = await response.json();

        console.log("Backend response:", data);


        if (!response.ok) {

            throw new Error(
                data.detail || "The server could not process the image."
            );
        }


        // Emotion
        const detectedEmotion = data.emotion;

        emotion.textContent = detectedEmotion;

        emotionIcon.textContent =
            emotionIcons[detectedEmotion.toLowerCase()] || "🙂";


        // Confidence
        let confidenceValue = parseFloat(data.confidence);


        if (isNaN(confidenceValue)) {
            confidenceValue = 0;
        }


        confidenceValue = Math.max(
            0,
            Math.min(100, confidenceValue)
        );


        confidence.textContent =
            confidenceValue.toFixed(2) + "%";


        confidenceFill.style.width =
            confidenceValue + "%";


        // Show result
        result.style.display = "block";

    } catch (error) {

        console.error("Analysis error:", error);

        showError(
            "Could not analyze the image: " + error.message
        );

    } finally {

        analyzeButton.disabled = false;

        buttonText.textContent = "Analyze Emotion";

        buttonLoader.style.display = "none";

    }

});
