# Vehicle Route Planner
## Setup and Running Locally

**Navigate to the Project Directory:**

Use the `cd` command to navigate into the extracted `veh_route_planner` directory.
```bash
cd path/to/your/veh_route_planner
```

**Create and Activate a Virtual Environment (Recommended):**

-   Create: `python -m venv venv`
-   Activate (Windows): `.\venv\Scripts\activate`
-   Activate (macOS/Linux): `source venv/bin/activate`

**Install Dependencies:**

```bash
pip install -r requirements.txt
```

**Mapbox Access Token :**

You need to replace the stars with your Mapbox Access Token which is hardcoded in both `src/static/js/app.js` and 'src/main.py'.

**Run the Flask Application:**

```bash
python3 -u src/main.py > app.log 2>&1
```

Open your web browser and navigate to `http://127.0.0.1:5001` or `http://localhost:5001`.
