
async def _recv_connect(conn, proc, timeout: float = 45.0) -> dict:
    """Aguarda a resposta final da conexão, ignorando mensagens de andamento."""
    loop = asyncio.get_running_loop()
    deadline = time.monotonic() + timeout

    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("Timeout na conexão com a corretora")

        ready = await loop.run_in_executor(
            None, conn.poll, min(0.5, remaining)
        )

        if ready:
            try:
                message = conn.recv()
            except EOFError as exc:
                raise ConnectionError(
                    "O processo da corretora encerrou a comunicação."
                ) from exc

            if message.get("type") == "progress":
                logger.info(
                    "Conexão %s: %s",
                    message.get("broker", "corretora"),
                    message.get("stage", "em andamento"),
                )
                continue

            return message

        if not proc.is_alive():
            raise ConnectionError(
                f"O processo da corretora encerrou inesperadamente "
                f"(código {proc.exitcode})."
            )
