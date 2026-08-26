# SatQuery AI

## Running the app

The `Start application` workflow runs both services for the Replit preview:

- Frontend: Vite on port 5000
- Backend: FastAPI / Uvicorn on port 8000

The frontend calls the backend through the Vite proxy using relative `/api` and
`/sample-images` URLs. The five sample investigations are served by the
backend and can be loaded from the Field presets row.

## Frontend notes

- Frontend source lives in `frontend/`.
- The visual theme uses the supplied palette: `#E43636`, `#F6EFD2`,
  `#E2DDB4`, and `#000000`.
- The supplied Anthropic Serif display font is served from
  `frontend/public/fonts/`.
- An `ANTHROPIC_API_KEY` is not required for the frontend to boot. Configure
  the backend environment separately if live model inference is enabled.