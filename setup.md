# 🏥 Healthcare Deep Learning — Project Setup Commands

Here are the direct terminal commands to set up the backend and frontend. These commands are tailored for your Windows environment so you can copy and paste them directly into your terminal.

---

## 1. Backend Setup

Open a **new terminal** and run these commands one by one to configure the FastAPI and PyTorch environment.

```powershell
# 1. Navigate to the backend directory
cd "d:\healthcareDeep_Learning\main\backend"

# 2. Create the Python virtual environment (if you haven't already)
python -m venv venv

# 3. Activate the virtual environment
.\venv\Scripts\activate

# 4. Install the required backend dependencies
# (We already created requirements.txt which includes the lightweight PyTorch CPU version)
pip install -r requirements.txt

# 5. Build out the required directory structure for the backend app
mkdir -p app\routes app\services app\models app\utils trained_models

# 6. Create the essential backend files
New-Item -ItemType File -Force -Path app\__init__.py
New-Item -ItemType File -Force -Path app\main.py
New-Item -ItemType File -Force -Path app\config.py
New-Item -ItemType File -Force -Path app\routes\__init__.py
New-Item -ItemType File -Force -Path app\routes\predict.py
New-Item -ItemType File -Force -Path app\routes\health.py
New-Item -ItemType File -Force -Path app\routes\report.py

# 7. Start the backend server (ensure your files have content before running this)
uvicorn app.main:app --reload
```

---

## 2. Frontend Setup

Open a **second terminal** and run these commands to set up the React + Vite + Tailwind CSS app.

```powershell
# 1. Navigate to the main directory
cd "d:\healthcareDeep_Learning\main"

# 2. Initialize the Vite React project in the "frontend" folder
npx create-vite@latest frontend --template react

# 3. Move into the frontend directory
cd frontend

# 4. Install the base React dependencies
npm install

# 5. Install Tailwind CSS and its necessary peers
npm install -D tailwindcss postcss autoprefixer

# 6. Generate the tailwind.config.js and postcss.config.js files
npx tailwindcss init -p

# 7. Start the frontend development server
npm run dev
```

---

## 3. Next Steps (After Running the Commands)

Once Tailwind is initialized in your frontend, you'll just need to configure `tailwind.config.js` to scan your React files. 

Replace the content of `frontend/tailwind.config.js` with:
```javascript
/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {},
  },
  plugins: [],
}
```

And add these `@tailwind` directives to the top of your `frontend/src/index.css`:
```css
@tailwind base;
@tailwind components;
@tailwind utilities;
```
