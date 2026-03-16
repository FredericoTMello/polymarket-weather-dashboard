# Local Run

## Objetivo

Este projeto roda como um dashboard local com um servidor Python simples e um runtime em JavaScript modular.

O baseline atual e heuristico. Ele nao implementa integralmente o `SPEC.md`.

## Requisitos

- Python 3
- Node.js
- acesso a internet para Open-Meteo e Polymarket

## Como rodar

Na raiz do projeto:

```powershell
python server.py
```

Depois abra:

```text
http://127.0.0.1:8090/dashboard.html
```

Se quiser bind e porta explicitos:

```powershell
python server.py 127.0.0.1 8090
```

## Validacao minima

Checagem de sintaxe:

```powershell
node --check dashboard-app.js
python -m py_compile server.py
```

Testes do nucleo analitico:

```powershell
node --test tests/analytics-core.test.mjs
```

Testes do parser de mercado no backend:

```powershell
python -m unittest discover -s tests -p "*_test.py"
```

Payload de mercado atual no backend:

- `question`
- `city`
- `city_key`
- `date`
- `parse_status`
- `parse_notes`
- `rule_confidence`
- `outcomes`
- `volume`
- `liquidity`

Endpoint de listagem para o ciclo market-first:

```text
/api/polymarket/markets?q=<texto>&city=<cidade>&limit=<n>
```

Exemplo:

```text
http://127.0.0.1:8090/api/polymarket/markets?q=london&limit=5
```

Smoke test manual:

1. Subir `server.py`
2. Abrir `dashboard.html`
3. Buscar e selecionar um mercado do Polymarket na busca principal
4. Confirmar que a pagina carrega:
   - mercado selecionado
   - modelos e historico
   - limites da leitura
5. Como fallback temporario, repetir o fluxo usando a busca por cidade

## Limitacoes conhecidas

- leitura de mercado ainda e heuristica
- data contratual ainda e inferida a partir do titulo do mercado
- timezone continua automatico no baseline atual
- nao ha Monte Carlo para horizontes mais longos
- nao ha backtesting nem position sizing
- scraping do Polymarket pode quebrar se a pagina mudar
- o HTML ainda contem um bloco legado inerte para referencia de transicao

## Arquivos principais

- `server.py`: servidor local e proxy do Polymarket
- `dashboard.html`: pagina principal
- `dashboard-app.js`: runtime da interface
- `analytics-core.mjs`: utilitarios puros testaveis
- `tests/analytics-core.test.mjs`: testes de regressao do nucleo atual
