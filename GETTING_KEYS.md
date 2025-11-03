# Quick Guide: Getting Your API Keys for Orion

This guide walks you through getting the 2 credentials you need to run Orion.

## What You Need

1. **Google Cloud Project ID** - Your project identifier
2. **Service Account JSON Key** - For BigQuery and Vertex AI access

**Note**: Orion now uses Gemini through Vertex AI, so you don't need a separate API key!

---

## Step 1: Google Cloud Project Setup (10 minutes)

### A. Create/Select a Google Cloud Project

1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Sign in with your Google account
3. Click the project dropdown at the top
4. Click "NEW PROJECT" or select an existing project
5. Note your **Project ID** (you'll need this)

**Free Tier**: Google Cloud offers $300 free credit for new users!

### B. Enable Required APIs

Enable both APIs Orion needs:

**BigQuery API:**
1. In Google Cloud Console, search for "BigQuery API" in the top search bar
2. Click on "BigQuery API"
3. Click the blue **"ENABLE"** button
4. Wait for it to enable (usually 10-30 seconds)

**Vertex AI API:**
1. In the search bar, search for "Vertex AI API"
2. Click on "Vertex AI API"
3. Click the blue **"ENABLE"** button
4. Wait for it to enable (usually 10-30 seconds)

### C. Create Service Account & Download Key

1. In the search bar, search for "Service Accounts"
2. Click "Service Accounts" (under IAM & Admin)
3. Click the blue **"+ CREATE SERVICE ACCOUNT"** button at the top
4. Fill in:
   - **Service account name**: `orion-agent` (or any name you like)
   - **Service account ID**: auto-filled (you can leave it)
   - Click **"CREATE AND CONTINUE"**
5. Grant it a role:
   - Type "BigQuery" in the "Grant this service account access to project" box
   - Select **"BigQuery Job User"** from the list
   - Click **"CONTINUE"**
6. Click **"DONE"** (skip the optional step)
7. Find your new service account in the list and click on it
8. Go to the **"KEYS"** tab
9. Click **"ADD KEY"** > **"Create new key"**
10. Select **"JSON"**
11. Click **"CREATE"** 
12. A JSON file will download - **save it somewhere safe!** (like your Desktop or Downloads folder)
13. **Remember the full path** to this file (e.g., `/Users/yourname/Downloads/orion-agent-xxxxx.json`)

✅ **You now have**:
- Your Project ID
- Service Account JSON key file

---

## Step 2: That's It!

✅ You now have everything you need! No separate Gemini API key required.

---

## Step 3: Configure Orion (5 minutes)

### Create your .env file

1. In your project directory, run:
```bash
cp .env.example .env
```

2. Open the `.env` file in any text editor

3. Fill in your 2 credentials:

```env
# Paste your Project ID here (no spaces)
GOOGLE_CLOUD_PROJECT=your-actual-project-id

# Paste the FULL path to your JSON key file
GOOGLE_APPLICATION_CREDENTIALS=/full/path/to/your-downloaded-key.json

# Optional: Vertex AI location (defaults to us-central1)
VERTEX_AI_LOCATION=us-central1
```

**Example**:
```env
GOOGLE_CLOUD_PROJECT=my-orion-project-12345
GOOGLE_APPLICATION_CREDENTIALS=/Users/john/Downloads/orion-agent-abc123.json
VERTEX_AI_LOCATION=us-central1
```

### Important Notes:

- **GOOGLE_CLOUD_PROJECT**: Just the project ID, nothing else
- **GOOGLE_APPLICATION_CREDENTIALS**: Must be the **full absolute path** starting with `/`
- No quotes needed around values
- No spaces after the `=` sign

---

## Step 4: Test It! (1 minute)

1. Make sure you're in the project directory
2. Run:
```bash
python -m src.cli
```

3. If successful, you'll see:
```
    ╔══════════════════════════════════════════════════════════════╗
    ║                    ORION - Data Analyst                      ║
    ║         AI-Powered BigQuery Data Analysis Agent              ║
    ╚══════════════════════════════════════════════════════════════╝

🔗 Connected to: bigquery-public-data.thelook_ecommerce
💡 Ask me anything about the e-commerce data!
   (Type 'exit' or 'quit' to leave)
```

4. Try asking: `What are total sales?`

---

## Troubleshooting

### Error: "Missing required environment variables"
- Check your `.env` file has GOOGLE_CLOUD_PROJECT and GOOGLE_APPLICATION_CREDENTIALS
- Make sure there are no extra spaces
- Verify the file is named exactly `.env` (with the dot)

### Error: "Invalid credentials" or "Permission denied"
- Double-check your service account has "BigQuery Job User" role
- Verify the JSON key file path is correct (use absolute path)
- Make sure BigQuery API and Vertex AI API are both enabled

### Error: "Vertex AI not available" or "Model not found"
- Make sure Vertex AI API is enabled in your project
- Check that you're using the correct project ID
- Verify your service account has appropriate permissions

### Error: "File not found" for credentials
- Use the full absolute path: `/Users/yourname/path/to/file.json`
- Don't use `~` or relative paths
- Make sure the file actually exists at that location

---

## Quick Checklist

- [ ] Google Cloud project created
- [ ] BigQuery API enabled
- [ ] Vertex AI API enabled
- [ ] Service account created with "BigQuery Job User" role
- [ ] Service account JSON key downloaded
- [ ] `.env` file created with credentials
- [ ] Orion runs successfully!

---

## Security Notes

🔒 **Keep your keys safe!**
- Never commit your `.env` file to git (it's already in .gitignore)
- Don't share your keys publicly
- The service account key file is sensitive - treat it like a password

---

## Need More Help?

- Detailed setup: See [SETUP.md](SETUP.md)
- Architecture: See [ARCHITECTURE.md](ARCHITECTURE.md)
- Example queries: See [example_queries.txt](example_queries.txt)

**Still stuck?** Make sure you followed every step in order. Most issues are from missing or incorrect keys in the `.env` file.

