PURPLE SIGNAL ANALYST — pacote de teste

ESTRUTURA
- index.html: interface web.
- server.py: API FastAPI (inclui seleção de ativos no scanner).
- broker.py / broker_worker.py: camada de conexão com corretoras.
- strategy.py: estratégia recebida, mantida sem alteração.
- requirements.txt: dependências.
- render.yaml: configuração inicial do serviço de API no Render.
- .env.example: variáveis necessárias.

TESTE LOCAL
1. Instale Python 3.11+ e MongoDB (local ou Atlas).
2. Crie um ambiente virtual e instale: pip install -r requirements.txt
3. Copie .env.example para .env e preencha MONGO_URL, DB_NAME e JWT_SECRET.
4. Execute: uvicorn server:app --host 0.0.0.0 --port 8000
5. Abra index.html no navegador e informe http://127.0.0.1:8000 como API.
   Se o navegador bloquear chamadas locais por CORS/file://, sirva a pasta:
   python -m http.server 5500
   e abra http://127.0.0.1:5500

PUBLICAÇÃO
- API: publique a pasta no Render e configure as variáveis de ambiente.
- Frontend: publique index.html no Netlify e informe a URL da API na tela de login.
- MongoDB Atlas: crie banco e usuário, permita a conexão do serviço Render.
- Nunca publique .env nem credenciais.

LIMITES IMPORTANTES
- O modo simulado gera candles artificiais e não representa mercado real.
- A integração Polarium é uma adaptação de endpoints da biblioteca IQ Option; não está validada por conexão real.
- A biblioteca iqoptionapi é de terceiros e pode quebrar quando a corretora alterar seus protocolos.
- O backend original não implementa stop diário de ganho/perda nem limites completos de risco.
- Não habilite modo automático em conta REAL antes de testes DEMO, validação jurídica/técnica e controles de risco.
- A estratégia retorna uma pontuação heurística de confluência, não probabilidade estatística de acerto.
- Os quatro arquivos Python recebidos compilam sintaticamente, mas isso não equivale a teste de integração com corretoras.
