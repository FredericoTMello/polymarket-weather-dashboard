# Market-First MVP Plan

## Problem

O projeto atual ja saiu da fase de estabilizacao e confiabilidade basica, mas o fluxo principal ainda entra por cidade. Isso cria um desvio entre:

- o produto desejado: analise de um mercado do Polymarket por vez
- a UX atual: buscar cidade, depois tentar encaixar os mercados dessa cidade

O ponto ainda vago nao e mais "como rodar" ou "como interpretar minimamente o mercado". O ponto agora e como migrar para um fluxo market-first sem reescrever a aplicacao inteira.

## Status

Proximo ciclo ativo.

## Goal

Fazer o usuario entrar pelo mercado do Polymarket, mantendo o baseline atual e a UX de analise unica.

Sucesso neste ciclo significa:

- o contrato do Polymarket passa a ser a entidade principal da tela
- cidade, data e outcomes passam a aparecer como contexto do contrato escolhido
- o fluxo atual por cidade pode existir temporariamente, mas deixa de ser a entrada principal
- a analise meteorologica continua subordinada ao mercado selecionado

## Assumptions

- nao vamos implementar parser completo do `SPEC.md` agora
- nao vamos adicionar contas, persistencia ou camadas de SaaS
- o backend atual em `server.py` pode crescer um pouco para expor mercados de forma mais direta
- a UX atual de pagina unica deve ser preservada o maximo possivel
- compatibilidade temporaria com o fluxo city-first e aceitavel se isso reduzir risco

## Estado Atual Do Fluxo

### Entrada principal

Hoje a entrada principal e city-first:

- o usuario busca uma cidade via `WeatherProvider.searchCities()`
- a selecao chama `setActiveCity()`
- a pagina carrega clima e historico da cidade primeiro
- so depois busca mercados por cidade via `MarketProvider.fetchByCity()`

Isso aparece em [`dashboard-app.js`](C:/Codex/Projetos/polymarket-weather-dashboard/dashboard-app.js):

- `DOM.input`, `DOM.dropdown`, `DOM.addButton`
- `fetchCities()`
- `setActiveCity()`
- `loadCity()`
- `MarketProvider.fetchByCity()`

### Consequencia pratica

O mercado ainda e um anexo da cidade, nao o objeto principal da experiencia.

O usuario nao escolhe:

- qual contrato especifico quer analisar
- qual mercado entre varios da mesma cidade
- qual regra do mercado esta priorizando

Em vez disso, ele escolhe uma cidade e o app adapta os mercados que encontrar.

## Principal Limitacao Para O MVP

A principal limitacao para o MVP nao e mais tecnica de baseline. E de produto:

- enquanto a entrada continuar sendo cidade, o app continua parecendo uma ferramenta meteorologica com dados do Polymarket acoplados
- o MVP desejado e o inverso: uma ferramenta de leitura de mercado do Polymarket com camada meteorologica subordinada

Em outras palavras:

- hoje: `cidade -> mercados`
- MVP desejado: `mercado -> cidade/data/outcomes -> analise`

## Menor Evolucao Recomendada

A menor evolucao com maior ganho agora e:

### 1. Introduzir listagem simples de mercados como ponto de entrada

Sem remover o fluxo atual de imediato, adicionar no backend uma forma de listar mercados relevantes ja normalizados.

O minimo util:

- endpoint para listar mercados
- possibilidade de escolher um mercado especifico pelo frontend
- usar o payload ja existente como base

### 2. Tratar o mercado selecionado como entidade principal da tela

No frontend:

- o usuario seleciona um contrato primeiro
- o app deriva cidade e data desse contrato
- a analise meteorologica e carregada para sustentar esse contrato

### 3. Manter o fluxo city-first como compatibilidade temporaria

Em vez de apagar o fluxo atual agora:

- deixar busca por cidade como fallback temporario
- introduzir a nova entrada market-first acima dela
- migrar a tela principal para sempre renderizar em torno de `selectedMarket`

Isso reduz risco e evita reescrita ampla.

## Plan

### Fase 1 - Backend Para Lista De Mercados

Objetivo:
expor mercados do Polymarket como lista utilizavel pelo frontend.

Tasks:

- adicionar no backend um endpoint simples de listagem de mercados
- permitir filtro leve por cidade ou texto, se necessario
- manter o payload canonico atual
- incluir `parse_status`, `parse_notes` e `rule_confidence` na listagem

Definicao de pronto:

- o frontend consegue obter uma lista de contratos sem depender de uma cidade escolhida antes
- o payload continua pequeno e consistente

### Fase 2 - Selecionar Um Mercado No Frontend

Objetivo:
criar a menor entrada market-first sem quebrar a UX atual.

Tasks:

- adicionar uma area de selecao de mercado no topo da pagina
- permitir escolher um contrato especifico
- ao selecionar um mercado:
  - derivar cidade
  - derivar data
  - carregar analise para esse contrato

Compatibilidade temporaria:

- manter busca por cidade ainda disponivel
- tratar city-first como fallback, nao como fluxo principal

Definicao de pronto:

- o usuario consegue abrir a analise a partir de um contrato especifico
- a tela continua unica e compreensivel

### Fase 3 - Reorganizar A Tela Em Torno Do Contrato

Objetivo:
fazer a tela parecer produto de mercado, nao produto de cidade.

Tasks:

- promover regra, data, cidade e outcomes do contrato no topo da analise
- manter clima, historico e sinais como suporte ao contrato
- ajustar textos da interface para refletir que o mercado e o objeto principal

Definicao de pronto:

- a hierarquia visual deixa claro qual contrato esta sendo analisado
- a analise meteorologica aparece como evidência para o contrato

## O Que Entra No Primeiro Ciclo Market-First

Entra agora:

- endpoint simples para listar mercados
- selecao de um contrato especifico
- uso do contrato como entrada principal
- compatibilidade temporaria com city-first
- reorganizacao leve da tela para centralizar o contrato

## O Que Fica Como Compatibilidade Temporaria

Fica temporariamente:

- busca de cidade
- `setActiveCity()` como caminho auxiliar
- fetch por cidade como fallback
- estrutura geral da pagina unica

## O Que Deve Ficar Para Depois

Fica para depois:

- remocao completa do fluxo city-first
- parser mais forte de contrato no nivel do `SPEC.md`
- filtros ricos de mercado
- persistencia de mercados acompanhados
- autenticação, historico salvo e qualquer camada de SaaS

## Risks

- listar mercados direto do Polymarket aumenta a exposicao a scraping fragil
- manter dois fluxos por um tempo aumenta complexidade de transicao
- se a selecao de mercado for feita sem hierarquia clara, a tela pode ficar ambigua
- se o contrato continuar parcialmente interpretado, market-first melhora UX, mas nao resolve sozinho a confiabilidade semantica

## Alternatives

Alternativa mais simples:

- nao introduzir listagem geral de mercados ainda
- apenas permitir escolher entre os contratos retornados depois de uma cidade

Isso e mais barato, mas entrega menos ganho de produto. Continua city-first no essencial, entao eu nao recomendaria como caminho principal para o MVP.

## Criterio De Conclusao Do Ciclo

Este ciclo pode ser considerado concluido quando:

- o usuario entra no fluxo principal escolhendo um mercado
- a tela deixa claro qual contrato esta sendo analisado
- clima, historico e sinais aparecem subordinados ao contrato
- city-first deixa de ser a narrativa principal do produto

## Quando O Projeto Entra Em Modo MVP De Verdade

O projeto passa a parecer um MVP real quando estas condicoes estiverem juntas:

- entrada principal por contrato do Polymarket
- contrato selecionado visivel e compreensivel
- leitura meteorologica sustentando esse contrato
- limites da interpretacao ainda claros
- fluxo local simples e repetivel

Antes disso, ele continua sendo uma base forte de prototipo. Depois disso, passa a ter forma clara de produto.
