# Deploying a Docker Container to Azure with Persistent Storage

This guide explains how to deploy your Dockerized Flask application to Azure App Service with persistent storage using an Azure File Share. It includes steps for setting up your app, creating and mounting the file share, and accessing it using Azure Storage Explorer.

---

## Prerequisites

- **Azure Account:** Ensure you have an active Azure subscription.
- **Docker:** Your application is containerized (Dockerfile available).
- **Container Registry:** Use Docker Hub or Azure Container Registry (ACR) to host your image.
- **Azure CLI (optional):** For command-line operations.
- **Azure Storage Explorer:** Download from [Microsoft’s website](https://azure.microsoft.com/en-us/features/storage-explorer/).

---

## Step 1: Build and Push Your Docker Image

1. **Build the Docker Image Locally:**

   ```bash
   docker build -t myapp:latest .
   ```

2. **Tag Your Image for Docker Hub (or ACR):**

   ```bash
   docker tag myapp:latest yourdockerhubusername/myapp:latest
   ```

3. **Push the Image:**

   ```bash
   docker push yourdockerhubusername/myapp:latest
   ```

   > *Note: If you use ACR, follow its specific push instructions.*

---

## Step 2: Deploy to Azure App Service for Containers

1. **Create a New Web App for Containers:**

   - Log in to the [Azure Portal](https://portal.azure.com).
   - Click **"Create a resource"** and search for **"Web App for Containers"**.
   - Click **"Create"**.

2. **Configure Basic Settings:**

   - **Subscription & Resource Group:** Choose your subscription and select/create a resource group.
   - **Name:** Provide a unique name for your App Service.
   - **Publish:** Choose **"Docker Container"**.
   - **Operating System:** Select **Linux**.

3. **Configure the Docker Container:**

   - Under the **Docker** tab:
     - **Image Source:** Choose **Docker Hub** (or ACR if using).
     - **Access Type:** Select Public (if your repository is public) or Private.
     - **Image and Tag:** Enter `yourdockerhubusername/myapp:latest`.

4. **Review and Create:**

   - Review your settings and click **"Create"** to deploy your app.

5. **Verify Deployment:**

   - Once deployed, navigate to the App Service URL to ensure your application is running correctly.

---

## Step 3: Create an Azure File Share for Persistent Storage

1. **Create a Storage Account:**

   - In the Azure Portal, click **"Create a resource"** and search for **"Storage Account"**.
   - Fill in the required details (e.g., subscription, resource group, name, region) and create the account.

2. **Create a File Share:**

   - Navigate to your Storage Account.
   - Click **"File shares"** in the left-hand menu.
   - Click **"+ File share"**.
   - Provide a name (e.g., `persistentfiles`) and set a quota (e.g., 100 GB).
   - Click **"Create"**.

---

## Step 4: Mount the File Share to Your App Service

1. **Open Your App Service:**

   - In the Azure Portal, navigate to the App Service that hosts your Docker container.

2. **Configure the File Share Mount:**

   - In the App Service menu, click **"Configuration"**.
   - Switch to the **"Path mappings"** (or **"Azure Storage Mounts"**) tab.
   - Click **"Add Azure Storage Mount"**.

3. **Enter Mount Details:**

   - **Name:** Provide a friendly name (e.g., `PERSISTENT_FILES`).
   - **Storage Type:** Select **Azure Files**.
   - **Account Name:** Enter your Storage Account name.
   - **Share Name:** Enter the file share name you created (e.g., `persistentfiles`).
   - **Access Key:** Retrieve and paste the access key from your Storage Account (found under **"Access keys"**).
   - **Mount Path:** Specify the mount path inside your container (e.g., `/persistent` or `/home/appdata`).

4. **Save and Restart:**

   - Save your configuration changes.
   - Restart the App Service to apply the new mount.

5. **Update Your Application:**

   - Modify your app code to read/write files from the mounted directory (e.g., set your output directory to `/persistent`).

---

## Step 5: Access the File Share with Azure Storage Explorer

1. **Download and Install Azure Storage Explorer:**

   - Download from the [Azure Storage Explorer page](https://azure.microsoft.com/en-us/features/storage-explorer/).

2. **Connect to Your Storage Account:**

   - Launch Azure Storage Explorer.
   - Click **"Add an Account"** or **"Sign in to Azure"**.
   - Sign in using your Azure credentials or connect via connection string/SAS token if needed.

3. **Navigate to Your File Share:**

   - In the left pane, expand your Storage Account.
   - Expand **"File Shares"**.
   - Locate your file share (e.g., `persistentfiles`) and click on it.

4. **Download Files:**

   - Browse to the files or folders you wish to download.
   - Right-click the file or folder and select **"Download"**.
   - Choose a local destination to save the files.

---

## Summary

- **Docker Image:** Build and push your containerized Flask app to Docker Hub or ACR.
- **Azure App Service:** Deploy your Docker container on Azure App Service for Containers.
- **Azure File Share:** Create an Azure File Share and mount it to your App Service to ensure persistent file storage.
- **Access Files:** Use Azure Storage Explorer to manage and download files from the file share.

By following these steps, your application will be deployed on Azure with persistent storage, ensuring that files remain accessible even if your container is restarted or scaled. For further details, consult the [Azure App Service documentation](https://docs.microsoft.com/en-us/azure/app-service/) and the [Azure Files documentation](https://docs.microsoft.com/en-us/azure/storage/files/).

```

