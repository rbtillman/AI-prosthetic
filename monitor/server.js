const express = require('express');
const path = require('path');
const app = express();
const port = process.env.PORT || 3000;
// API_URL should point to your Raspberry Pi server
const apiUrl = process.env.API_URL || 'http://192.168.25.129:5000';

// Serve configuration to client
app.get('/config.js', (req, res) => {
  res.type('application/javascript');
  res.send(`window.API_URL = '${apiUrl}';`);
});

// Serve static files
app.use(express.static(path.join(__dirname, 'public')));

app.listen(port, () => console.log(`Client server listening on port ${port}`));