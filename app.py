"""
Cloud FinOps Cost Optimization — FastAPI Server.

Exposes the OpenEnv-compliant HTTP endpoints:
  POST /openenv/reset  → Reset environment
  POST /openenv/step   → Execute one action
  GET  /openenv/state  → Get current observation
  GET  /health         → Health check
  GET  /               → Root info
"""

import os
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from env import CloudFinOpsEnv

app = FastAPI(
    title="Cloud FinOps Cost Optimizer — OpenEnv",
    description="AI agent environment for cloud cost optimization",
    version="1.0.0",
)

# Global environment instance
current_env: CloudFinOpsEnv = None  # type: ignore


# ──────────────────────────────────────────────────────────────────────────
# OpenEnv endpoints
# ──────────────────────────────────────────────────────────────────────────

@app.post("/openenv/reset")
@app.post("/openenv/reset/")
@app.post("/reset")
@app.post("/reset/")
@app.post("/api/reset")
async def reset(request: Request = None):
    """Reset the environment and return the initial observation."""
    global current_env
    try:
        # Sometimes the grader will send the task_id in the JSON body
        task_id = "idle_killer" # fallback default
        if request is not None:
            try:
                body = await request.json()
                if "task_id" in body:
                    task_id = body["task_id"]
                elif "task" in body:
                    task_id = body["task"]
            except:
                pass
        
        # Override with env var if present (for local testing)
        task_id = os.getenv("TASK_ID", task_id)
        
        current_env = CloudFinOpsEnv(task_id=task_id)
        obs = current_env.reset()
        return {
            "status": "success",
            "message": f"Environment reset with task '{task_id}'",
            "observation": obs.model_dump(),
        }
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": str(e)},
        )


@app.post("/openenv/step")
@app.post("/openenv/step/")
@app.post("/step")
@app.post("/step/")
@app.post("/api/step")
async def step(request: Request):
    """Execute one step in the environment."""
    global current_env
    if current_env is None:
        return JSONResponse(
            status_code=400,
            content={"status": "error", "message": "Environment not initialised. Call /openenv/reset first."},
        )
    try:
        data = await request.json()
        action = data.get("action", "done()")
        obs, reward, done, info = current_env.step(action)
        return {
            "status": "success",
            "observation": obs.model_dump(),
            "reward": reward.model_dump(),
            "done": done,
            "info": info,
        }
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": str(e)},
        )


@app.get("/openenv/state")
async def get_state():
    """Return the current observation without taking an action."""
    global current_env
    if current_env is None:
        return JSONResponse(
            status_code=400,
            content={"status": "error", "message": "Environment not initialised. Call /openenv/reset first."},
        )
    try:
        obs = current_env.state()
        return {
            "status": "success",
            "observation": obs.model_dump(),
        }
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": str(e)},
        )


# ──────────────────────────────────────────────────────────────────────────
# Utility endpoints
# ──────────────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    """Health check."""
    return {"status": "ok"}


@app.get("/ping")
async def ping():
    """Returns basic API info natively."""
    return {
        "name": "Cloud FinOps Cost Optimizer",
        "version": "1.0.0",
        "tasks": ["idle_killer", "rightsizer", "minefield"]
    }

from fastapi.responses import FileResponse
import os

@app.get("/")
async def root():
    """Serve the stunning web UI."""
    index_path = os.path.join(os.path.dirname(__file__), "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {"message": "index.html not found! API is active on /openenv"}


# ──────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)
