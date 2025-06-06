// Image Processing
// Image Processing
// Image Processing
function previewImage(event) {
    var previewProcessImage = document.getElementById('previewProcessImage');
    var imageUpload = event.target.files[0];
    var reader = new FileReader();

    reader.onload = function () {
        previewProcessImage.src = reader.result;
        previewProcessImage.style.display = 'block';
    }

    if (imageUpload) {
        reader.readAsDataURL(imageUpload);
    }
}