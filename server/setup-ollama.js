const fs = require('fs');
const path = require('path');

async function setupOllama() {
  console.log('Setting up AnythingLLM with Ollama...');
  
  try {
    // Create .env file with Ollama configuration
    const envContent = `# LLM Provider Configuration
LLM_PROVIDER=ollama
OLLAMA_BASE_PATH=http://localhost:11434
OLLAMA_MODEL_PREF=llama3:latest
OLLAMA_MODEL_TOKEN_LIMIT=4096

# Embedding Configuration
EMBEDDING_ENGINE=native
EMBEDDING_MODEL_MAX_CHUNK_LENGTH=1000

# Vector Database Configuration
VECTOR_DB=lancedb

# Memory Service Configuration
MEMORY_SERVICE_URL=http://localhost:5001

# Server Configuration
SERVER_PORT=3001
JWT_SECRET=your-secret-key-${Date.now()}

# Storage
STORAGE_DIR=./storage

# Disable telemetry
DISABLE_TELEMETRY=true
`;

    const envPath = path.join(__dirname, '.env');
    fs.writeFileSync(envPath, envContent);
    
    console.log('✅ System configured successfully!');
    console.log('Created .env file with:');
    console.log('- LLM Provider: Ollama');
    console.log('- Embedding Engine: Native');
    console.log('- Vector Database: LanceDB');
    console.log('- Memory Service: Port 5001');
    console.log('\nPlease restart the server for changes to take effect.');
    
    process.exit(0);
  } catch (error) {
    console.error('Setup failed:', error);
    process.exit(1);
  }
}

// Run setup
setupOllama(); 