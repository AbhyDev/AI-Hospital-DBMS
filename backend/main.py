from fastapi import FastAPI
from .api import router
from .cors_config import add_cors_middleware
from .routers import users, oauth

app = FastAPI()
add_cors_middleware(app)
app.include_router(users.router)
app.include_router(oauth.router)
app.include_router(router)