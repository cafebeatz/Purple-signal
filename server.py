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
    email = os.environ.get("ADMIN_EMAIL", "").strip().lower()
    password = os.environ.get("ADMIN_PASSWORD", "")
    if not email or not password:
        logger.warning(
            "Administrador inicial não criado: configure ADMIN_EMAIL e ADMIN_PASSWORD no ambiente."
        )
        return
    if len(password) < 12:
        logger.warning(
            "Administrador inicial não criado: ADMIN_PASSWORD deve ter pelo menos 12 caracteres."
        )
        return
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
    """Limpa dados operacionais legados uma única vez na migração para a V3.

    A marca é gravada somente depois das exclusões. Se o processo cair durante
    a limpeza, a próxima inicialização pode repetir as exclusões com segurança.
    Usuários, credenciais, configurações e demais dados de conta não são tocados.
    """
    migration_id = "purple_signal_v3_operational_history_reset"
    migrations = db["schema_migrations"]

    if await migrations.find_one({"_id": migration_id}):
        logger.info("Migração V3 já aplicada; histórico operacional preservado.")
        return

    signals_result = await db.signals.delete_many({})
    brain_result = await db.strategy_brain_events.delete_many({})

    try:
        await migrations.insert_one({
            "_id": migration_id,
            "applied_at": iso_now(),
            "deleted_signals": signals_result.deleted_count,
            "deleted_brain_events": brain_result.deleted_count,
        })
    except Exception:
        # Em inicializações concorrentes, outra instância pode gravar a mesma
        # marca após executar as mesmas exclusões. Só ignoramos se ela existir.
        if not await migrations.find_one({"_id": migration_id}):
            raise

    logger.warning(
        "Migração V3 aplicada: %s sinais/operações/sessões e %s eventos de aprendizagem removidos.",
        signals_result.deleted_count,
        brain_result.deleted_count,
    )
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
