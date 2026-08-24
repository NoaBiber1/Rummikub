import logging
import time

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from models import GameState, SolveResponse
from solver import solve_rummikub
from validator import validate_board

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Rummikub Solver API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.exception("Unhandled backend error on %s %s: %s", request.method, request.url.path, exc)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal Server Error"},
    )


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/solve", response_model=SolveResponse)
def solve(game_state: GameState) -> SolveResponse:
    is_valid, error_message = validate_board(game_state.board)
    if not is_valid:
        raise HTTPException(status_code=400, detail=error_message)

    start = time.perf_counter()
    optimal_state = solve_rummikub(game_state)
    elapsed_ms = int((time.perf_counter() - start) * 1000)

    return SolveResponse(
        original_state=game_state,
        optimal_state=optimal_state,
        is_valid=True,
        execution_time_ms=elapsed_ms,
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
