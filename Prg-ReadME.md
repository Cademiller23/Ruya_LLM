# Notes
**Author** <Cade Miller>
**Start Date** <2025-06-03>

# Added to docker/.env 
- <SERVER_PORT, LLM_PROVIDER, OLLAMA_BASE_URL>

# Summary (06-03-25/06-04-25)
- UI/UX: 
    - Branded (Ruya) the LLM model with a label and title. via frontend/index.html (favicon.ico, favicon.png, anything-llm-dark.png, anything-llm-white.png) via frontend/public
    - Chatbot response image as Ruya's Image via src/components/UserIcon/workspace.svg
    - Changed background surrounding the chatbot to be a dark black and the interior to be a lighter black via frontend/index.css
    - Added a linear gradient border surrounding the chatbot, purple -> turqouise. via frontend/index.css
    - Modified Chat history font to white and added thicknes to divisor via src/components/Workspace/ChatContent/ChatHistory/index.js
    - Modified border between Send Chat messsage and icons for upload via frontend/index.css
    - Created a purple accent around the send message via frontend/index.css
    - Isolated the sidepanels text via frontend/index.css
    - Created four color schemes for Ruya (ruyaPurple, ruyaTurqouise, ruyaPanel, ruyaBlack) via frontend/tailwind.config
    - Isolated chat text with a --theme-chat-input-text via frontend/tailwind.config
    - Modified dropdown arrow to see most recent chatHistory and added a pulse animation via frontend/index.css
    - Modified 4 logos to reflect Ruya AI Logo frontend/media/logo 
# Scope
- General Changes
    - Include: Branding, Tailwind customization, chat history and chat enhancements.
    
# Tailwind changes
- animation: pulse
- chat-input: text
- bg: primary
- colors: ruyaPurple, ruyaTurqouise, ruyaPanel, ruyaBlack
- Allowed files: "./src/**/*.{js,jsx,ts,tsx}",

# Additional Detail/Changes
- Updated App.jsx: added color scheme
- Updated src/components/Workspace/ChatContent/ChatHistory/PromptInput: border style (Above AttachmentManager)
- Updated index.css @layer components (gradient-border)








