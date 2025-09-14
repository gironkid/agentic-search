# Frontend Requirements Document - Agentic Medical Search System

## 1. Overview

### Purpose
Create a clean, minimalist web interface for the Agentic Medical Search System that mirrors the simplicity of the CLI while providing an aesthetic, modern user experience.

### Design Philosophy
- Terminal-inspired aesthetics with modern polish
- Single-page application with minimal navigation
- Focus on content, not chrome
- Fast, responsive, no unnecessary features

## 2. Core Functionality

### 2.1 Chat Input
- **Message Input Field**
  - Clean text input at bottom of chat
  - Placeholder text: "Ask a medical question..."
  - Enter to submit, Shift+Enter for new line
  - Auto-resize for multi-line input
  - No autocomplete or suggestions

- **Send Button**
  - Simple send icon (arrow or paper plane)
  - Disabled while searching
  - Keyboard shortcut indicator (Enter)

### 2.2 Chat Interface
- **Message Display**
  - Linear conversation flow (ChatGPT-style)
  - User messages with subtle user icon/avatar
  - Assistant responses with medical/assistant icon
  - Full-width messages, no bubbles
  - Clear visual separation between messages
  - Markdown rendering with syntax highlighting
  - Clean typography with good readability

- **Search Progress**
  - Inline status updates during search:
    - "Searching PubMed..."
    - "Searching ClinicalTrials.gov..."
    - "Searching FDA database..."
    - "Searching web resources..."
    - "Analyzing results..."
  - Subtle loading animation (dots or spinner)
  - Smooth transition to final response

### 2.3 Results Display
- **Assistant Response**
  - Comprehensive answer with inline citations
  - Natural conversational tone
  - Sources referenced naturally in text

- **Source Results**
  - Collapsible "View Sources" section below each response:
    - PubMed Articles
    - Clinical Trials
    - FDA Information
    - Web Results
  - Each result shows:
    - Title (clickable link to source)
    - Brief excerpt/abstract
    - Source and date
  - Simple expand/collapse for long content

### 2.4 Export Options
- **Simple Export**
  - Copy to clipboard button for messages
  - Download chat history as markdown
  - Print-friendly view

## 3. Visual Design

### 3.1 Color Scheme
- **Light Mode (Default)**
  - Background: Off-white (#FAFAFA)
  - Text: Dark gray (#1A1A1A)
  - Accent: Medical blue (#0066CC)
  - Borders: Light gray (#E0E0E0)

- **Dark Mode (Optional)**
  - Background: Dark gray (#1A1A1A)
  - Text: Light gray (#E0E0E0)
  - Accent: Light blue (#66B2FF)
  - Borders: Medium gray (#404040)

### 3.2 Typography
- **Font Stack**
  - Monospace for status messages: 'JetBrains Mono', 'Fira Code', monospace
  - Body text: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif
  - Clean, readable sizing (16px base)

### 3.3 Layout
- **Centered Conversation**
  - Max width: 768px for optimal reading (ChatGPT-style)
  - Generous padding on sides
  - Messages take full container width
  - Input fixed at bottom of viewport
  - Conversation scrolls above input
  - Clean separation lines or spacing between messages

## 4. Interactions

### 4.1 Animations
- Smooth message appearance (fade in from bottom)
- Typing indicator with animated dots
- Smooth expand/collapse for sources
- Subtle transitions between loading states
- Optional: Response streaming effect (text appears progressively)

### 4.2 Keyboard Support
- Enter: Submit search
- Escape: Clear search
- Tab: Navigate between elements
- Space: Expand/collapse sections

## 5. Technical Implementation

### 5.1 Minimal Stack
- **Framework**: React or Vue (single-page app)
- **Styling**: Tailwind CSS or vanilla CSS
- **HTTP Client**: Native fetch API
- **Markdown**: markdown-it or similar lightweight parser
- **Icons**: Lucide or Heroicons (minimal set)

### 5.2 API Integration
- Single endpoint: POST /search
- Handle long-polling or SSE for real-time updates
- Simple error handling with user-friendly messages
- 30-second timeout handling

### 5.3 State Management
- Keep it simple: useState/ref for search state
- Local storage for:
  - Theme preference
  - Last 5 searches (optional)

## 6. Performance Requirements

- Initial load < 100KB JavaScript
- Time to interactive < 2 seconds
- Smooth 60fps scrolling
- Works on 3G connections
- No external fonts (use system fonts)

## 7. Browser Support

- Chrome/Edge (last 2 versions)
- Firefox (last 2 versions)
- Safari (last 2 versions)
- Mobile browsers (iOS Safari, Chrome Android)

## 8. Responsive Design

- **Desktop**: Full experience with optimal spacing
- **Tablet**: Same as desktop with adjusted padding
- **Mobile**:
  - Full-width search bar
  - Stacked results
  - Touch-friendly tap targets (44px minimum)

## 9. MVP Features

### Required
1. Search input and submission
2. Chat-style conversation interface
3. Real-time status updates in chat
4. Assistant responses with citations
5. Collapsible source results
6. Links to original sources
7. Mobile responsive
8. Copy to clipboard for messages

### Optional for MVP
1. Dark mode toggle
2. Chat history persistence (session)
3. Export chat to markdown
4. Keyboard shortcuts display
5. Clear chat button

## 10. Example User Flow

1. User loads page → sees clean interface with centered input (like ChatGPT)
2. Types medical question → hits Enter
3. Question appears with user icon, input moves to bottom
4. Loading indicator shows with status updates inline
5. Assistant response streams in or appears complete
6. Sources appear as subtle links/buttons below response
7. User can expand sources section for details
8. User asks follow-up question → conversation continues seamlessly
9. Can copy individual messages or export entire conversation

## 11. Error Handling

- Network errors: "Unable to connect. Please check your connection."
- Timeout: "Search is taking longer than expected. Please try again."
- No results: "No relevant results found. Try rephrasing your question."
- API errors: "Something went wrong. Please try again."

## 12. Success Metrics

- Average search completion < 10 seconds
- Mobile usage > 30%
- Zero unnecessary features
- Clean, distraction-free interface
- Fast initial load time
- Intuitive without documentation

## 13. Implementation Notes

### File Structure
```
frontend/
├── src/
│   ├── components/
│   │   ├── ChatContainer.jsx
│   │   ├── Message.jsx
│   │   ├── ChatInput.jsx
│   │   ├── LoadingIndicator.jsx
│   │   └── SourcesSection.jsx
│   ├── api/
│   │   └── search.js
│   ├── styles/
│   │   └── main.css
│   └── App.jsx
├── index.html
└── package.json
```

### Sample API Call
```javascript
const response = await fetch('/search', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({ query: userQuery })
});
```

## 14. Inspiration & References

- Terminal/CLI aesthetics
- Google's search simplicity
- DuckDuckGo's clean interface
- Perplexity AI's result presentation
- Minimal design principles

## 15. What We're NOT Building

- User accounts/authentication
- Search history beyond session
- Complex filters or sorting
- Social features
- Comments or ratings
- Saved searches database
- Multi-step wizards
- Autocomplete/suggestions
- Complex analytics
- Advertisements