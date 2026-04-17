const imageInput = document.getElementById("imageInput");
const preview = document.getElementById("preview");
const predictBtn = document.getElementById("predictBtn");
const output = document.getElementById("output");

// ---------------------------------------------------------
// HANDLE IMAGE INPUT
// ---------------------------------------------------------
imageInput.onchange = () => {
    const file = imageInput.files[0];
    if (!file) return;

    preview.src = URL.createObjectURL(file);
    predictBtn.disabled = false;
    output.innerHTML = "";
};

// ---------------------------------------------------------
// PREDICTION VIA API
// ---------------------------------------------------------
predictBtn.onclick = async () => {
    const file = imageInput.files[0];
    if (!file) {
        alert("Silakan pilih gambar terlebih dahulu!");
        return;
    }

    output.innerHTML = "<p>Memproses prediksi...</p>";

    const formData = new FormData();
    formData.append("file", file);

    try {

        const gradcamImage = document.getElementById("gradcamImage");

        const response = await fetch("http://127.0.0.1:5000/predict", {
            method: "POST",
            body: formData
        });

        if (!response.ok) {
            throw new Error("Gagal memanggil API");
        }

        const data = await response.json();

        // -----------------------------
        // TAMPILKAN HASIL
        // -----------------------------
        let html = `
        <strong>${data.predicted_class}</strong><br>
        Akurasi: <b>${data.confidence.toFixed(2)}%</b>
        <hr>
        <h4>Detail Probabilitas</h4>
        <ul>
        `;

        for (const [label, value] of Object.entries(data.all_predictions)) {
            html += `<li>${label}: ${value.toFixed(2)}%</li>`;
        }

        html += `
            </ul>
            <hr>
            <h4>Visualisasi Grad-CAM</h4>
            <img src="data:image/png;base64,${data.gradcam_image}"
                style="max-width:100%; border:1px solid #ccc;">
        `;

        output.innerHTML = html;

        gradcamImage.src = `data:image/png;base64,${data.gradcam_image}`;
        gradcamImage.style.display = "block";

    } catch (error) {
        console.error(error);
        output.innerHTML = "<p style='color:red'>Gagal memproses gambar.</p>";
    }
};