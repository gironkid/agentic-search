# Cloud Run Deployment Guide

## Prerequisites
- Google Cloud Project with billing enabled
- gcloud CLI installed and authenticated
- Docker installed (optional, for local testing)

## Environment Variables Required
Set these in Cloud Run as environment variables:
- `OPENROUTER_API_KEY`
- `OPENAI_API_KEY`
- `PUBMED_API_KEY`
- `TAVILY_API_KEY`
- `FDA_API_KEY` (optional)

## Deployment Steps

### 1. Build and Push to Artifact Registry

```bash
# Set your project ID
export PROJECT_ID="your-project-id"
export REGION="us-central1"
export SERVICE_NAME="agentic-medical-search"

# Configure Docker for Artifact Registry
gcloud auth configure-docker ${REGION}-docker.pkg.dev

# Create Artifact Registry repository (first time only)
gcloud artifacts repositories create medical-search \
    --repository-format=docker \
    --location=${REGION} \
    --description="Medical Search API"

# Build and push the image
gcloud builds submit --tag ${REGION}-docker.pkg.dev/${PROJECT_ID}/medical-search/${SERVICE_NAME}
```

### 2. Deploy to Cloud Run

```bash
gcloud run deploy ${SERVICE_NAME} \
    --image ${REGION}-docker.pkg.dev/${PROJECT_ID}/medical-search/${SERVICE_NAME} \
    --platform managed \
    --region ${REGION} \
    --allow-unauthenticated \
    --memory 2Gi \
    --cpu 2 \
    --timeout 300 \
    --max-instances 10 \
    --set-env-vars "OPENROUTER_API_KEY=your-key" \
    --set-env-vars "OPENAI_API_KEY=your-key" \
    --set-env-vars "PUBMED_API_KEY=your-key" \
    --set-env-vars "TAVILY_API_KEY=your-key"
```

### 3. Alternative: Using Secret Manager (Recommended)

```bash
# Create secrets
echo -n "your-openrouter-key" | gcloud secrets create openrouter-api-key --data-file=-
echo -n "your-openai-key" | gcloud secrets create openai-api-key --data-file=-
echo -n "your-pubmed-key" | gcloud secrets create pubmed-api-key --data-file=-
echo -n "your-tavily-key" | gcloud secrets create tavily-api-key --data-file=-

# Deploy with secrets
gcloud run deploy ${SERVICE_NAME} \
    --image ${REGION}-docker.pkg.dev/${PROJECT_ID}/medical-search/${SERVICE_NAME} \
    --platform managed \
    --region ${REGION} \
    --allow-unauthenticated \
    --memory 2Gi \
    --cpu 2 \
    --timeout 300 \
    --max-instances 10 \
    --set-secrets="OPENROUTER_API_KEY=openrouter-api-key:latest" \
    --set-secrets="OPENAI_API_KEY=openai-api-key:latest" \
    --set-secrets="PUBMED_API_KEY=pubmed-api-key:latest" \
    --set-secrets="TAVILY_API_KEY=tavily-api-key:latest"
```

## Local Testing with Docker

```bash
# Build locally
docker build -t agentic-medical-search .

# Run locally with environment variables
docker run -p 8080:8080 \
    -e OPENROUTER_API_KEY="your-key" \
    -e OPENAI_API_KEY="your-key" \
    -e PUBMED_API_KEY="your-key" \
    -e TAVILY_API_KEY="your-key" \
    agentic-medical-search
```

## Test the Deployment

```bash
# Get the service URL
SERVICE_URL=$(gcloud run services describe ${SERVICE_NAME} --region ${REGION} --format 'value(status.url)')

# Test health endpoint
curl ${SERVICE_URL}/health

# Test search endpoint
curl -X POST ${SERVICE_URL}/search \
    -H "Content-Type: application/json" \
    -d '{"query": "treatment for migraine"}'
```

## Important Security Notes

1. **NEVER commit API keys to git** - The .env file should never be in version control
2. **Use Secret Manager** for production deployments
3. **Enable authentication** if this is not a public API
4. **Set up proper IAM roles** for service accounts

## Monitoring

View logs:
```bash
gcloud run services logs read ${SERVICE_NAME} --region ${REGION}
```

View metrics in Cloud Console:
```
https://console.cloud.google.com/run/detail/${REGION}/${SERVICE_NAME}/metrics
```