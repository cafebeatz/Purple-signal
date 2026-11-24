 ---------------------------------------------------------------------------
# Servir o HTML
# ---------------------------------------------------------------------------

app.include_router(api)


@app.get("/", include_in_schema=False)
async def serve_frontend():
    candidates = (
        ROOT_DIR / "template" / "index.html",
        ROOT_DIR / "templates" / "index.html",
        ROOT_DIR / "index.html",
        ROOT_DIR / "Purple_Signal_robo_interface.html",
    )
    for html_file in candidates:
        if html_file.is_file():
            return FileResponse(html_file)

    raise HTTPException(
        status_code=404,
        detail="Interface HTML não encontrada na raiz do projeto.",
    )


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Índices e migração única do histórico antigo
# ---------------------------------------------------------------------------

INDEXES: dict[str, list[IndexModel]] = {
    "users": [
        IndexModel([("email", ASCENDING)], unique=True, name="email_unique"),
        IndexModel([("id", ASCENDING)], unique=True, name="id_unique"),
    ],
    "signals": [
        IndexModel([("user_id", ASCENDING), ("ts", DESCENDING)], name="user_ts"),
        IndexModel(
            [("user_id", ASCENDING), ("session_id", ASCENDING), ("ts", DESCENDING)],
            name="user_session_ts",
        ),
        IndexModel(
            [("user_id", ASCENDING), ("result", ASCENDING), ("asset", ASCENDING), ("ts", DESCENDING)],
            name="archive_filters",
        ),
    ],
    "strategy_brain_events": [
        IndexModel([("signal_id", ASCENDING)], unique=True, name="brain_signal_unique"),
        IndexModel([("user_id", ASCENDING), ("pattern_key", ASCENDING)], name="brain_user_pattern"),
    ],
}


async def seed_admin() -> None:
    email = os.environ.get("ADMIN_EMAIL", "admin@sinais.com").lower()
    password = os.environ.get("ADMIN_PASSWORD", "admin123")

    existing = await db.users.find_one({"email": email})
    if existing:
        return

    await db.users.insert_one(
        {
            "id": str(uuid.uuid4()),
            "email": email,
            "name": "Administrador",
            "password_hash": hash_password(password),
            "role": "admin",
            "created_at": iso_now(),
        }
    )
    logger.info("Admin seed criado: %s", email)


async def clear_legacy_signals_once() -> None:
    """Compatibilidade com o startup antigo; preserva integralmente o histórico."""
    logger.info("Migração de histórico ignorada: sinais existentes foram preservados.")


@app.on_event("startup")
async def startup() -> None:
    for collection, models in INDEXES.items():
        for model in models:
            with suppress(Exception):
                await db[collection].create_indexes([model])

    # Mantém compatibilidade com a inicialização sem apagar o histórico.
    await clear_legacy_signals_once()
    await seed_admin()


@app.on_event("shutdown")
async def shutdown() -> None:
    for task in list(_SCAN_TASKS.values()):
        task.cancel()
    _SCAN_TASKS.clear()

    for session in list(_BROKER_SESSIONS.values()):
        with suppress(Exception):
            disconnect_broker(session)

    client.close()
