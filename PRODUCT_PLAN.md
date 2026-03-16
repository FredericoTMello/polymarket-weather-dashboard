# Product Plan - polymarket-weather-dashboard

## Visao

Transformar o projeto atual em um produto de analise climatica para mercados do Polymarket.

O foco deixa de ser comparar varias cidades ao mesmo tempo.

O foco passa a ser:

1. o usuario escolhe um mercado/cidade do Polymarket
2. o sistema coleta dados de multiplas fontes e modelos meteorologicos
3. o sistema consolida os sinais mais relevantes
4. o usuario recebe uma leitura clara, util e explicitamente heuristica

## Problema

Hoje, quem analisa mercados climaticos do Polymarket precisa juntar manualmente:

- regras do mercado
- data e local corretos
- previsoes de varios modelos
- historico comparavel
- divergencia entre fontes
- limites da leitura

Esse trabalho e fragmentado, facil de interpretar errado e dificil de repetir com consistencia.

## Proposta de Valor

Entregar uma pagina de analise por mercado que responda de forma objetiva:

- qual e a previsao consolidada para essa cidade/data
- como os modelos diferem entre si
- quais sao os limites e sinais da leitura atual
- como isso se compara ao historico
- qual e a leitura analitica do mercado, sem prometer automacao de trade

## Usuario Inicial

Usuario alvo inicial:

- pessoa que acompanha mercados climaticos do Polymarket
- quer profundidade analitica em um mercado por vez
- valoriza clareza, contexto e confiabilidade mais do que volume de widgets

## MVP

O MVP deve fazer bem apenas o essencial:

### Entrada

- selecionar um mercado/cidade do Polymarket
- carregar a data e a regra do mercado

### Analise

- buscar previsoes de multiplos modelos meteorologicos
- buscar historico comparavel
- calcular media, dispersao e sinais basicos
- mostrar leitura consolidada da cidade/data
- exibir limites da leitura e observacoes relevantes

### Saida

- resumo principal da analise
- tabela ou bloco com modelos/fontes
- historico comparavel
- leitura de risco e limites
- visao do mercado selecionado

## O Que Reaproveitar

Vale reaproveitar do projeto atual:

- proxy e scraping base do Polymarket em `server.py`
- integracao base com Open-Meteo
- aliases de cidades
- parsing de ranges de temperatura
- calculo inicial de probabilidade
- calculo inicial de erro historico
- estrutura conceitual do `SPEC.md` como referencia de rigor futuro

## O Que Nao E MVP

Fica fora do MVP:

- comparacao simultanea de varias cidades
- grid de cards como centro do produto
- favoritos como funcionalidade principal
- sizing de posicao
- automacao de operacoes
- backtesting completo
- billing, times, contas e multi-tenant
- promessas estatisticas avancadas sem validacao suficiente

## Decisoes de Produto

### Decisao 1: profundidade em vez de largura

Melhor uma analise profunda de um mercado por vez do que um dashboard superficial de varias cidades.

### Decisao 2: consolidacao antes de sofisticacao

Antes de aumentar o numero de fontes ou regras, o sistema precisa ser legivel, consistente e honesto sobre seus limites.

### Decisao 3: leitura analitica antes de automacao

O produto deve primeiro ajudar o usuario a entender o mercado.
Automacao de trade so faz sentido depois de validar o nucleo analitico.

## Arquitetura de Produto Sugerida

Camadas recomendadas:

1. Market adapter
   Responsavel por listar e normalizar mercados do Polymarket.

2. Weather providers
   Responsavel por consultar previsoes e historico em multiplas fontes.

3. Analysis engine
   Responsavel por normalizar sinais, comparar modelos e produzir a leitura consolidada.

4. Presentation layer
   Responsavel por exibir uma pagina clara para um mercado/cidade por vez.

## Riscos Principais

- scraping do Polymarket e fragil
- timezone e data do contrato podem ser interpretados errado
- multiplas fontes aumentam complexidade de normalizacao
- o projeto atual concentra responsabilidades demais no frontend
- o `SPEC.md` descreve um sistema mais avancado do que o codigo atual suporta

## Baseline Analitico Atual

O baseline atual deve ser entendido como leitura heuristica, nao como motor validado de elegibilidade operacional.

Hoje o projeto consegue:

- agregar previsoes de multiplos modelos
- mostrar historico comparavel simples
- calcular dispersao entre modelos
- fazer parsing basico de ranges de temperatura
- produzir uma leitura heuristica de mercado com base em probabilidade estimada

Hoje o projeto ainda nao garante:

- validacao completa de timezone do contrato
- validacao explicita de ano do contrato
- correspondencia semantica completa das regras do mercado
- Monte Carlo para horizontes mais longos
- sizing, backtesting ou validacao estatistica formal
- baseline protegido por testes unitarios

## Itens Explicitamente Adiados Em Relacao Ao SPEC

Ficam adiados ate haver base confiavel:

- parser completo de regra de mercado
- validacao forte de timezone e data contratual
- Monte Carlo para horizontes > D+2
- criterio operacional de trade
- Kelly / position sizing
- backtesting obrigatorio
- claims fortes de confianca ou elegibilidade

## Principios Para Evolucao

- manter o escopo pequeno
- priorizar baseline confiavel
- nao vender certeza que o sistema nao consegue sustentar
- separar ingestao, analise e apresentacao
- evoluir por fases curtas e verificaveis

## Definicao de Sucesso do MVP

O MVP sera bem sucedido quando:

- o usuario conseguir abrir um mercado especifico
- a aplicacao produzir uma leitura consolidada e compreensivel
- a analise tiver fontes e regras explicitas
- os limites da leitura estiverem claros
- o sistema rodar localmente com fluxo previsivel
