from app.db import engine

with engine.connect() as conn:
    print("Connected to Supabase successfully")