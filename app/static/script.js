const dropzone = document.getElementById("dropzone");
const dropzoneText = document.getElementById("dropzone-text");
const fileInput = document.getElementById("fileInput");
const preview = document.getElementById("preview");
const loading = document.getElementById("loading");
const result = document.getElementById("result");
const predictedClass = document.getElementById("predictedClass");
const confidenceBar = document.getElementById("confidenceBar");
const confidenceText = document.getElementById("confidenceText");

dropzone.addEventListener("click", () => fileInput.click());

dropzone.addEventListener("dragover", (e) => {
    e.preventDefault();
    dropzone.classList.add("dragover");
});
dropzone.addEventListener("dragleave", () => dropzone.classList.remove("dragover"));
dropzone.addEventListener("drop", (e) => {
    e.preventDefault();
    dropzone.classList.remove("dragover");
    if (e.dataTransfer.files.length) {
        handleFile(e.dataTransfer.files[0]);
    }
});

fileInput.addEventListener("change", () => {
    if (fileInput.files.length) handleFile(fileInput.files[0]);
});

function handleFile(file) {
    const reader = new FileReader();
    reader.onload = (e) => {
        preview.src = e.target.result;
        preview.classList.remove("hidden");
        dropzoneText.classList.add("hidden");
    };
    reader.readAsDataURL(file);
    predictImage(file);
}

async function predictImage(file) {
    result.classList.add("hidden");
    loading.classList.remove("hidden");

    const formData = new FormData();
    formData.append("image", file);

    try {
        const response = await fetch("/predict", { method: "POST", body: formData });
        const data = await response.json();
        showResult(data);
    } catch (err) {
        loading.classList.add("hidden");
        alert("Something went wrong. Is the server running?");
    }
}

function showResult(data) {
    loading.classList.add("hidden");
    result.classList.remove("hidden");
    predictedClass.textContent = data.class;
    confidenceText.textContent = `${data.confidence}% confidence`;
    confidenceBar.style.width = "0%";
    setTimeout(() => { confidenceBar.style.width = data.confidence + "%"; }, 50);
}

// Sample image click-to-try
document.querySelectorAll(".sample-img").forEach(img => {
    img.addEventListener("click", async () => {
        preview.src = img.src;
        preview.classList.remove("hidden");
        dropzoneText.classList.add("hidden");

        const blob = await (await fetch(img.src)).blob();
        const file = new File([blob], img.dataset.name, { type: blob.type });
        predictImage(file);
    });
});