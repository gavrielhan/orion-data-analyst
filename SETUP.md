# Setup Instructions for Orion

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Google Cloud Setup

#### A. Create a Google Cloud Project
1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Create a new project or select an existing one
3. Note your Project ID

#### B. Enable BigQuery API
1. Navigate to "APIs & Services" > "Library"
2. Search for "BigQuery API"
3. Click "Enable"

#### C. Create a Service Account
1. Go to "IAM & Admin" > "Service Accounts"
2. Click "Create Service Account"
3. Name it (e.g., "orion-bigquery-agent")
4. Click "Create and Continue"
5. Grant role: **BigQuery Job User**
6. Click "Done"
7. Click on the created service account
8. Go to "Keys" tab
9. Click "Add Key" > "Create New Key" > "JSON"
10. Download and save the key file

### 3. Gemini API Setup

1. Go to [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Sign in with your Google account
3. Click "Create API Key"
4. Copy your API key

### 4. Configure Environment

Create a `.env` file in the project root:

```bash
cp .env.example .env
```

Edit `.env` with your credentials:

```env
# Google Cloud Configuration
GOOGLE_CLOUD_PROJECT=your-project-id-here
GOOGLE_APPLICATION_CREDENTIALS=/absolute/path/to/your-service-account-key.json

# Gemini API Configuration
GOOGLE_AI_API_KEY=your-gemini-api-key-here

# Optional: BigQuery Settings
BIGQUERY_DATASET=bigquery-public-data.thelook_ecommerce
MAX_QUERY_ROWS=10000
QUERY_TIMEOUT=300
```

**Important**: Use absolute path for `GOOGLE_APPLICATION_CREDENTIALS`

### 5. Test the Installation

Run Orion:

```bash
python -m src.cli
```

Try a simple query:
```
What are total sales?
```

## Troubleshooting

### "Permission denied" or "Access Denied"
- Verify your service account has the **BigQuery Job User** role
- Check that the BigQuery API is enabled
- Ensure you're using the correct project ID

### "API key not valid"
- Verify your Gemini API key is correct
- Check that you haven't exceeded quota limits

### "Module not found" errors
```bash
pip install -r requirements.txt --upgrade
```

### Import errors
- Make sure you're running from the project root directory
- Verify your Python version is 3.10+

## Next Steps

Once setup is complete, you can:
- Try example queries from `example_queries.txt`
- Build more complex queries
- Explore the codebase to understand the architecture

## Need Help?

- Check the [README.md](README.md) for more details
- Review `example_queries.txt` for sample queries
- Open an issue if you encounter problems

