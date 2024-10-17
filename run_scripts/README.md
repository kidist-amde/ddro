To install `gcloud` and `gsutil` from the terminal on your server, follow these steps:

### 1. Download and Install Google Cloud SDK

#### For Linux (assuming you're on a Linux-based server):
Run the following commands:

```bash
curl -O https://dl.google.com/dl/cloudsdk/channels/rapid/downloads/google-cloud-sdk-439.0.0-linux-x86_64.tar.gz
```

Extract the downloaded archive:
```bash
tar -xf google-cloud-sdk-439.0.0-linux-x86_64.tar.gz
```

Install the SDK:
```bash
./google-cloud-sdk/install.sh
```

### 2. Initialize the Google Cloud SDK

After installation, restart your terminal or activate the installation:
```bash
source ~/.bashrc  # or ~/.zshrc if you're using zsh
```

Initialize the `gcloud` CLI:
```bash
gcloud init
```

### 3. Verify `gcloud` and `gsutil`

Check if `gcloud` and `gsutil` are installed successfully by running:
```bash
gcloud version
gsutil version
```

That’s it! Now you have `gcloud` and `gsutil` installed on your server.