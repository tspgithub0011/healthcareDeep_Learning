# Healthcare Deep Learning — Commands to Run

## Prerequisites

- **Python 3.12+** installed
- **Node.js v18+** and **npm** installed

---

## 1. Backend Setup (FastAPI + PyTorch)

### Navigate to backend directory
```bash
cd d:\healthcareDeep_Learning\main\backend
```

### Create virtual environment (first time only)
```bash
python -m venv venv
```

### Create virtual environment (each time when you run backend in the current terminal session)
```bash
Set-ExecutionPolicy -ExecutionPolicy Bypass -Scope Process
```

### Activate virtual environment
```bash
# Windows PowerShell
.\venv\Scripts\Activate.ps1

# Windows CMD
.\venv\Scripts\activate.bat
```

### Install dependencies
```bash
pip install -r requirements.txt
```

### Install PyTorch (CPU version — lighter, works on any machine)
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

### Install PyTorch (GPU/CUDA version — faster inference, requires NVIDIA GPU)
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### Start the backend server
```bash
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

> Backend runs at: **http://localhost:8000**  
> API docs available at: **http://localhost:8000/docs**

---

## 2. Frontend Setup (React + Vite)

### Navigate to frontend directory
```bash
cd d:\healthcareDeep_Learning\main\frontend
```

### Install dependencies (first time only)
```bash
npm install
```

### Start the frontend dev server
```bash
npm run dev
```

> Frontend runs at: **http://localhost:5173**

### Build for production (optional)
```bash
npm run build
```

### Preview production build (optional)
```bash
npm run preview
```

---

## 3. Quick Start (Run Both Servers)

Open **two separate terminals** and run:

**Terminal 1 — Backend:**
```bash
cd d:\healthcareDeep_Learning\main\backend
.\venv\Scripts\activate.bat
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

**Terminal 2 — Frontend:**
```bash
cd d:\healthcareDeep_Learning\main\frontend
npm run dev
```

Then open **http://localhost:5173** in your browser.

---

## 4. Useful Commands

### Check backend health
```bash
curl http://localhost:8000/api/health
```

### Run backend tests
```bash
cd d:\healthcareDeep_Learning\main\backend
.\venv\Scripts\python.exe -m pytest tests/ -v
```

### Run a quick API test
```bash
cd d:\healthcareDeep_Learning\main\backend
.\venv\Scripts\python.exe test_api.py
```

---

## 5. Environment Variables

Backend environment variables are in `main/backend/.env`:

| Variable         | Default              | Description                    |
|------------------|----------------------|--------------------------------|
| `PORT`           | `8000`               | Backend server port            |
| `HOST`           | `0.0.0.0`            | Backend server host            |
| `ENV`            | `development`        | Environment mode               |
| `FRONTEND_URL`   | `http://localhost:5173` | Frontend URL for CORS       |
| `MODEL_DIR`      | `./trained_models`   | Path to trained model files    |
| `USE_GPU`        | `true`               | Enable GPU inference           |
| `MAX_FILE_SIZE_MB` | `10`               | Max upload file size in MB     |

---

## Troubleshooting

### PowerShell script execution blocked
If you get "running scripts is disabled on this system":
```bash
Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
```
Or run commands with bypass:
```bash
powershell -ExecutionPolicy Bypass -Command "npm run dev"
```

### Port already in use
Kill the process using the port:
```bash
# Find process on port 8000
netstat -ano | findstr :8000

# Kill it (replace PID with actual process ID)
taskkill /PID <PID> /F
```
