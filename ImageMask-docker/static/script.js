// JS stuff for docker trace app
// R. Tillman 3.11.2025
// Copyright (c) 2025 Tillman. All Rights Reserved.

let canvas, ctx;
let polygonPoints = [];

function initCanvas(imageUrl) {
  canvas = document.getElementById('traceCanvas');
  ctx = canvas.getContext('2d');
  const img = new Image();
  img.onload = function() {
    // Set canvas size to match image
    canvas.width = img.width;
    canvas.height = img.height;
    ctx.drawImage(img, 0, 0);
  };
  img.src = imageUrl;

  // Listen for clicks on the canvas
  canvas.addEventListener('click', function(event) {
    const rect = canvas.getBoundingClientRect();
    const x = Math.round(event.clientX - rect.left);
    const y = Math.round(event.clientY - rect.top);
    polygonPoints.push([x, y]);
    drawPoints();
  });
}

function drawPoints() {
  // Redraw the image first
  const img = new Image();
  img.onload = function() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.drawImage(img, 0, 0);
    // Draw the polygon
    if (polygonPoints.length > 0) {
      ctx.beginPath();
      ctx.moveTo(polygonPoints[0][0], polygonPoints[0][1]);
      for (let i = 1; i < polygonPoints.length; i++) {
        ctx.lineTo(polygonPoints[i][0], polygonPoints[i][1]);
      }
      ctx.strokeStyle = 'red';
      ctx.lineWidth = 2;
      ctx.stroke();
    }
  };
  img.src = canvas.toDataURL();  // temporarily use canvas content to redraw background
}

function resetPolygon() {
  polygonPoints = [];
  // Redraw the original image
  const img = new Image();
  img.onload = function() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.drawImage(img, 0, 0);
  };
  img.src = imageUrl;
}


function submitPolygon() {
  if (polygonPoints.length < 3) {
    alert("Please select at least 3 points to form a polygon.");
    return;
  }
  fetch(`/process/${filename}`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json'
    },
    body: JSON.stringify({ points: polygonPoints })
  })
  .then(response => response.json())
  .then(data => {
    if (data.error) {
      document.getElementById('result').innerText = data.error;
      return;
    }
    document.getElementById('result').innerText = `Average angle: ${data.avg_angle} degrees`;
    if (data.next_image) {
      // Redirect to the next image after a short delay
      setTimeout(() => {
         window.location.href = `/trace/${data.next_image}`;
      }, 1000);
    } else {
      // If no next image, redirect to the home page or show a message
      setTimeout(() => {
         window.location.href = "/";
      }, 2000);
    }
  })
  .catch(error => {
    console.error('Error:', error);
  });
}
