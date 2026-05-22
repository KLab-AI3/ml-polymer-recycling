# Frontend React Application

This directory contains the React/TypeScript frontend for the Polymer Aging ML application.

## Architecture

The frontend is a modern React application with TypeScript:

```
frontend/
├── src/
│   ├── components/          # React components
│   │   ├── Header.tsx
│   │   ├── SpectrumChart.tsx
│   │   ├── ResultsDisplay.tsx
│   │   └── ...
│   ├── apiClient.ts         # Centralized API client
│   ├── types/
│   │   ├── api.ts          # Auto-generated API types
│   │   └── index.ts        # Custom types
│   ├── App.tsx             # Main application component
│   └── index.tsx           # Application entry point
├── public/                 # Static assets
├── package.json           # Dependencies and scripts
└── tsconfig.json          # TypeScript configuration
```

## Key Features

- **Type Safety**: Full TypeScript integration with OpenAPI-generated types
- **Centralized API Client**: Single source for all backend communication
- **Component Architecture**: Modular, reusable React components
- **Responsive Design**: Works across desktop and mobile devices
- **Error Handling**: Graceful error handling with user feedback

## Setup and Development

### Prerequisites

- Node.js 16+ and npm 8+
- Backend API server running (for development)

### Installation

```bash
# Install dependencies
npm install --legacy-peer-deps

# Verify TypeScript types are up to date
npm run typegen:file
```

### Development Server

```bash
# Start development server with hot reload
npm start

# Opens http://localhost:3000
```

### Build for Production

```bash
# Create production build
npm run build

# Build files output to build/ directory
```

## API Integration

### Centralized API Client

All backend communication goes through `src/apiClient.ts`:

```typescript
import { ApiClient } from './apiClient';

const api = new ApiClient('http://localhost:8000');

// Example usage
const result = await api.analyzeSpectrum({
  spectrum: spectrumData,
  model_name: 'resnet',
  modality: 'raman'
});
```

### Type Generation

API types are automatically generated from the OpenAPI schema:

```bash
# Generate types from running backend
npm run typegen

# Generate types from schema file
npm run typegen:file
```

## Scripts

Available npm scripts:

```bash
npm start          # Development server
npm run build      # Production build
npm test           # Run tests
npm run lint       # ESLint checking
npm run format     # Prettier formatting
npm run typegen    # Generate API types
```

## API Contract Adherence

The frontend strictly adheres to the OpenAPI contract:
- All requests/responses validated by TypeScript types
- Automatic type generation ensures contract compliance
- No direct backend imports or cross-boundary dependencies
