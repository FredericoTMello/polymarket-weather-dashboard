# Market Reliability Plan

## Objetivo

Planejar um ciclo enxuto para tornar a entidade mercado mais confiavel, sem expandir para features de SaaS e sem tentar implementar o `SPEC.md` por inteiro.

Este plano parte do estado atual do projeto e foca em pequenas mudancas com ganho real de confiabilidade.

## Status

Ciclo concluido.

Entregas concluidas neste ciclo:

- Fase A - parser canonico de evento no backend
- Fase B - testes de parsing de mercado
- Fase C - UI honesta sobre status de mercado

## Estado Atual Da Interpretacao De Mercado

### Backend

Em [`server.py`](C:/Codex/Projetos/polymarket-weather-dashboard/server.py), a interpretacao do mercado hoje e baseada em regras simples:

- `parse_event_title()` extrai apenas `question`, `city` e `date_label`
- a extracao depende de um unico padrao textual: `in <city> on <date>`
- `normalize_city_query()` transforma a cidade em `city_key` por alias
- o payload canonico atual inclui:
  - `question`
  - `city`
  - `city_key`
  - `date`
  - `outcomes`
  - `volume`
  - `liquidity`

O backend ainda nao explicita:

- se o parsing foi confiavel ou parcial
- qual metrica do mercado foi inferida
- qual fonte oficial de resolucao e esperada
- qual timezone contratual deveria valer
- se a data extraida e semanticamente suficiente

### Frontend

Em [`dashboard-app.js`](C:/Codex/Projetos/polymarket-weather-dashboard/dashboard-app.js), a leitura de mercado depende de heuristicas adicionais:

- `buildMarketAnalysis()` assume que `event.date` pode ser resolvido por `resolveTargetDate()`
- `marketRuleConfidence` hoje nao mede confianca semantica do contrato
- `marketRuleConfidence` sobe para `HIGH` quando existe data futura resolvida e previsao para o indice daquela data
- `marketRuleConfidence` cai para `MEDIUM` ou `LOW` por falta de previsao ou data resolvivel

Na pratica, a UI usa `HIGH | MEDIUM | LOW` como um proxy misto de:

- parse de data
- disponibilidade do horizonte meteorologico
- existencia de valores de modelo

Isso mistura duas camadas que deveriam ser separadas:

1. confianca de interpretacao do contrato
2. capacidade de estimar meteorologia para esse contrato

## Principais Riscos

### Risco 1: parsing semantico estreito demais

O parser atual depende de um unico formato de titulo. Se o Polymarket variar redacao, cidade composta, estacao ou texto auxiliar, o payload pode continuar existindo com baixa qualidade sem explicitar isso.

### Risco 2: `marketRuleConfidence` esta semantica e operacionalmente misturado

Hoje a UI trata a confianca como se refletisse regra de mercado, mas ela e fortemente influenciada por disponibilidade de forecast. Isso pode induzir interpretacao errada do que realmente foi validado.

### Risco 3: falta de status explicito de interpretacao

O payload nao informa ao frontend se o mercado foi interpretado como:

- valido
- parcial
- desconhecido

Sem isso, a UI precisa deduzir demais.

### Risco 4: data contratual continua parcial

`resolveTargetDate()` ainda assume ano corrente ou seguinte. Isso e aceitavel para o baseline atual, mas precisa ser tratado como limite explicito da interpretacao, nao como dado confiavel do contrato.

### Risco 5: testes cobrem utilitarios, mas nao o parsing dos eventos do Polymarket

Ja existem testes para ranges, datas e probabilidade, mas ainda nao ha testes que validem:

- titulo -> `question/city/date`
- classificacao de interpretacao
- comportamento esperado para eventos ambiguos

## Melhoria Minima Recomendada

A menor evolucao com maior retorno agora e:

### 1. Consolidar interpretacao do mercado no backend

Mover para o backend a decisao sobre o status de interpretacao do evento.

Adicionar ao payload algo como:

```json
{
  "question": "...",
  "city": "London",
  "city_key": "london",
  "date": "March 16",
  "parse_status": "valid",
  "parse_notes": [],
  "rule_confidence": "HIGH"
}
```

Sem exagero de modelagem. O minimo util e:

- `parse_status = valid | partial | unknown`
- `parse_notes = []`
- `rule_confidence = HIGH | MEDIUM | LOW`

### 2. Separar confianca contratual de disponibilidade meteorologica

No frontend:

- `rule_confidence` deve vir do backend
- disponibilidade de previsao deve virar outra leitura, por exemplo `forecast_support`

Assim a UI deixa de fingir que confianca semantica e a mesma coisa que cobertura de forecast.

### 3. Tratar data inferida como data inferida

Enquanto ano/timezone nao forem fortes:

- o backend deve marcar isso em `parse_notes`
- a UI deve apenas refletir o status, sem promover o mercado como semanticamente validado

### 4. Adicionar testes de parsing de evento

Criar testes pequenos cobrindo:

- evento bem formatado
- evento com cidade reconhecivel e data reconhecivel
- evento com titulo parcial
- evento sem data parseavel
- evento sem cidade confiavel

## O Que Entra Agora

Entram neste ciclo:

- consolidar parsing de titulo, cidade e data no backend
- classificar cada evento em `valid`, `partial` ou `unknown`
- explicitar `rule_confidence` no payload
- adicionar `parse_notes` simples
- ajustar a UI para refletir esse status sem prometer certeza
- adicionar testes para parsing de eventos do Polymarket

## O Que Fica Para Depois

Fica para o ciclo seguinte:

- fluxo market-first completo
- selecao direta de contrato antes da cidade
- parser completo de metrica, fonte oficial e timezone contratual
- endurecimento total contra todos os formatos possiveis do Polymarket
- qualquer camada de persistencia, contas ou SaaS

## O Que Ainda Depende De Validacao Futura

Ainda depende de validacao futura:

- timezone oficial por contrato
- ano contratual explicito
- fonte oficial de resolucao
- correspondencia completa da metrica do mercado
- politica de bloqueio operacional equivalente ao `SPEC.md`

## Roadmap Por Fases

### Fase A - Parser Canonico De Evento

Objetivo:
transformar a interpretacao do evento em uma responsabilidade explicita do backend.

Status:
concluida

Tasks:

- criar uma funcao de interpretacao de evento mais explicita em `server.py`
- retornar `parse_status`, `parse_notes` e `rule_confidence`
- manter o payload atual e apenas acrescentar campos

Concluido quando:

- cada evento retornado pelo backend tem status de interpretacao
- o frontend nao precisa mais inventar confianca semantica sozinho

### Fase B - Testes De Parsing De Mercado

Objetivo:
garantir repetibilidade no comportamento do parser de evento.

Status:
concluida

Tasks:

- adicionar testes de unidade para o parser do backend
- cobrir eventos validos, parciais e desconhecidos
- validar normalizacao de cidade e data extraida

Concluido quando:

- o parser tem cobertura minima para os casos principais
- mudancas futuras no backend podem ser revisadas com menos risco

### Fase C - UI Honesta Sobre Status De Mercado

Objetivo:
refletir o novo status na interface sem aumentar escopo de produto.

Status:
concluida

Tasks:

- mostrar `rule_confidence` vindo do backend
- mostrar notas simples quando o parsing for parcial
- separar status de interpretacao do status de forecast

Concluido quando:

- a UI comunica claramente se o mercado foi interpretado de forma valida, parcial ou desconhecida
- a pagina nao mistura mais confianca de contrato com cobertura meteorologica

## Criterio De Conclusao Do Ciclo

Este ciclo estara concluido quando:

- a entidade mercado tiver status explicito de interpretacao
- backend e frontend concordarem sobre esse status
- testes cobrirem parsing de eventos do Polymarket
- a UI comunicar limites do contrato sem sugerir certeza inexistente

Resultado:

- a entidade mercado passou a ter status explicito de interpretacao
- backend e frontend passaram a refletir esse status
- testes cobrem parsing de eventos do Polymarket
- a UI separa interpretacao do contrato de suporte meteorologico

## Recomendacao Final

O melhor proximo passo de implementacao e um ciclo curto em 3 fases:

1. parser canonico no backend
2. testes de parsing
3. reflexo honesto desse status na UI

Isso aumenta confiabilidade real sem reabrir UX ampla, sem puxar SaaS e sem prometer que o `SPEC.md` ja esta sendo cumprido.
