async function uploadImage(event) {
  const file = event.target.files[0];
  const previewImageElement = document.getElementById("previewProcessImage");
  const canvasElement = document.getElementById("processedCanvas");
  const resultTableBody = document.getElementById("resultTableBody");

  if (file) {
      const reader = new FileReader();

      // Show preview of uploaded image
      reader.onload = function (e) {
          previewImageElement.src = e.target.result;
          previewImageElement.style.display = "block";
      };
      reader.readAsDataURL(file);

      const formData = new FormData();
      formData.append("image", file);

      try {
          const response = await fetch("http://127.0.0.1:5000/process_image", {
              method: "POST",
              body: formData,
          });

          const result = await response.json();
          console.log("AI Model Output:", result);

          // Check for errors in the response
          if (result.error) {
              alert(`Error: ${result.error}`);
              return;
          }

          // Update table with predictions, ensuring keys exist
          resultTableBody.innerHTML = `
              <tr>
                  <td>${result.prediction }</td>
                  <td>${result.accuracy }</td>
                  <td>${result.RGB }</td>
              </tr>
          `;

        
        //document.getElementById("accuracy").textContent = result.accuracy;
        //document.getElementById("rgb").textContent = result.RGB;

          // Render processed image to canvas (if part of response)
          if (result.processed_image) {
              const ctx = canvasElement.getContext("2d");
              const img = new Image();
              img.onload = function () {
                  canvasElement.width = img.width;
                  canvasElement.height = img.height;
                  ctx.drawImage(img, 0, 0);
              };
              img.src = `data:image/png;base64,${result.processed_image}`;
          }
      } catch (error) {
          console.error("Error uploading image:", error);
          alert("An error occurred while processing the image.");
      }
  }
}
