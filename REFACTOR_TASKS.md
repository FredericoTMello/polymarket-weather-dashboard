# Refactor Tasks - single-market analysis SaaS

## Objetivo

Reposicionar o projeto atual para uma aplicacao de analise profunda de um mercado/cidade do Polymarket por vez.

Este arquivo e o checklist operacional da evolucao.

Use junto com `PRODUCT_PLAN.md`.

## Regras de Execucao

- manter mudancas pequenas e revisaveis
- nao misturar reparo de baseline com expansao de escopo
- validar cada fase antes de marcar como concluida
- nao assumir que o `SPEC.md` ja esta implementado

## Fase 0 - Alinhar direcao do produto

Objetivo: garantir que a implementacao siga a nova visao de produto.

- [x] Revisar `PRODUCT_PLAN.md` e confirmar o MVP real.
- [x] Definir a saida principal da pagina de analise.
- [x] Confirmar o que fica fora do MVP inicial.
- [x] Registrar quais partes do projeto atual serao reaproveitadas.

Concluido quando:

- [x] existe um MVP claro de mercado unico
- [x] existe separacao entre MVP, fase seguinte e itens adiados

## Fase 1 - Restaurar baseline tecnico

Objetivo: fazer o projeto atual rodar com fluxo minimo confiavel.

- [x] Corrigir blockers de parse/runtime no frontend.
- [x] Garantir que `server.py` rode no repositorio atual.
- [x] Remover suposicoes operacionais quebradas, como caminhos antigos de execucao.
- [x] Validar o fluxo minimo: mercado -> weather -> analise -> UI.

Concluido quando:

- [x] o frontend carrega sem erro fatal imediato
- [x] o backend sobe localmente com comando simples
- [x] as rotas principais respondem
- [x] existe um procedimento minimo de execucao local

## Fase 2 - Fixar contrato de dados

Objetivo: tornar explicita a relacao entre backend, Polymarket e frontend.

- [x] Definir o payload canonico de mercado retornado pelo backend.
- [x] Alinhar frontend e backend aos mesmos nomes de campos.
- [x] Normalizar aliases de cidade em um unico lugar ou regra consistente.
- [x] Explicitar como data, local e outcomes sao extraidos.

Concluido quando:

- [x] o contrato de mercado e consistente
- [x] frontend e backend nao dependem de campos ambiguos
- [x] a analise nao quebra por suposicoes ocultas do payload

## Fase 3 - Reorientar UX para mercado unico

Objetivo: abandonar o modelo de dashboard multi-cidade como centro do produto.

- [x] Remover dependencia conceitual de varias cidades simultaneas.
- [x] Definir a tela principal como analise de um mercado/cidade por vez.
- [x] Reavaliar favoritos, cards e autoload de estacoes.
- [x] Estruturar a interface em torno de resumo, modelos, historico e confianca.

Concluido quando:

- [x] a UX faz sentido com um mercado por vez
- [x] o usuario entende rapidamente qual mercado esta sendo analisado
- [x] a pagina principal prioriza leitura analitica, nao comparacao em grid

## Fase 4 - Separar responsabilidades tecnicas

Objetivo: reduzir acoplamento sem reescrever tudo de uma vez.

- [x] Separar ingestao de mercados do Polymarket.
- [x] Separar consultas meteorologicas por provider/modelo.
- [x] Separar motor de analise do codigo de interface.
- [x] Isolar parsing de ranges, datas e confianca.
- [x] Remover trechos duplicados, mortos ou contraditorios.

Concluido quando:

- [x] ingestao, analise e apresentacao ficam distinguiveis
- [x] mudancas na logica analitica exigem menos impacto na UI
- [x] o projeto fica mais facil de revisar e testar

## Fase 5 - Definir baseline analitico confiavel

Objetivo: deixar explicito o que o sistema realmente consegue afirmar.

- [x] Comparar implementacao atual com o `SPEC.md` no caminho critico.
- [x] Manter apenas regras que possam ser sustentadas com evidencias no codigo.
- [x] Marcar itens avancados como adiados quando ainda nao houver base confiavel.
- [x] Revisar claims de confianca, edge e elegibilidade.

Concluido quando:

- [x] o produto nao promete mais do que entrega
- [x] os limites da analise estao claros
- [x] existe um baseline analitico honesto e consistente

## Fase 6 - Adicionar safety nets

Objetivo: tornar o nucleo analitico menos fragil a mudancas futuras.

- [x] Adicionar testes para parsing de range.
- [x] Adicionar testes para datas e contratos.
- [x] Adicionar testes para utilitarios de probabilidade.
- [x] Documentar execucao local e limitacoes principais.

Concluido quando:

- [x] o nucleo mais critico tem validacao repetivel
- [x] outra pessoa consegue rodar o projeto com instrucoes simples
- [x] riscos conhecidos estao documentados

## Ordem Recomendada

1. Fase 0
2. Fase 1
3. Fase 2
4. Fase 3
5. Fase 4
6. Fase 5
7. Fase 6

## Proximo Marco Recomendado

Iniciar o ciclo market-first descrito em `MARKET_FIRST_MVP_PLAN.md`, executando apenas o necessario para:

- listar mercados como entrada principal
- manter compatibilidade temporaria com o fluxo por cidade
- aproximar o produto de um MVP real sem abrir escopo de SaaS

## Regra de acompanhamento

- [x] marcar task somente depois de validada
