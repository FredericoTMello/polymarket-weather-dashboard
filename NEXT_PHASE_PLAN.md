# Next Phase Plan

## Estado Atual

O projeto saiu de um dashboard local fragil e passou a ter um baseline executavel, com uma leitura unica por cidade/mercado, contrato de dados mais claro, separacao tecnica minima e testes no nucleo analitico.

Ao fim do ultimo ciclo, o projeto tambem passou por um passe forte de UX/UI:

- hero e entrada principal mais claros
- fluxo market-first mais legivel para exploracao
- card principal mais apresentavel
- hierarquia visual mais coerente com o MVP atual

Hoje o repositorio tem:

- `server.py` como servidor local e proxy simples do Polymarket
- `dashboard.html` como pagina principal
- `dashboard-app.js` como runtime modular da interface
- `analytics-core.mjs` com utilitarios puros testaveis
- `tests/analytics-core.test.mjs` cobrindo parsing, datas e probabilidade
- `LOCAL_RUN.md` com execucao local e limitacoes principais

## O Que Melhorou

Comparado ao estado inicial, houve melhora real:

- o frontend deixou de ter blocker imediato de parse/runtime
- o servidor passou a rodar no repositorio atual sem depender de ambiente antigo
- o payload de mercado ficou mais explicito e consistente
- a UX deixou de ser um grid multi-cidade e passou a focar em uma analise por vez
- a logica principal deixou de ficar toda embutida no HTML ativo
- o produto passou a comunicar melhor seus limites heuristico-analiticos
- agora existe validacao repetivel do nucleo mais critico

Conclusao objetiva: sim, o projeto melhorou de forma material. Ele ainda nao e um SaaS pronto nem um motor analitico confiavel no sentido forte do `SPEC.md`, mas deixou de ser apenas um prototipo fragil e passou a ter uma base mais revisavel.

## O Que Ainda Falta

Os principais gaps atuais sao:

- interpretacao contratual ainda e parcial
- timezone e ano contratual ainda sao inferidos
- scraping do Polymarket continua fragil
- a selecao de mercado ainda e indireta via cidade, nao mercado primeiro
- ainda nao existe persistencia, conta, historico salvo ou camadas reais de SaaS
- backend e integracoes externas seguem sem testes proprios

## Opcoes De Evolucao

### Opcao A - Confiabilidade Analitica

Foco: reduzir risco de interpretacao errada.

Incluir:

- parser mais forte de titulo/regra do mercado
- tratamento mais rigoroso de data contratual
- timezone explicito onde for possivel
- separacao mais clara entre mercado valido, parcial e indefinido
- testes adicionais para backend e parsing de eventos do Polymarket

Melhor quando:

- a prioridade e confiar mais na leitura antes de sofisticar produto

### Opcao B - UX Market-First

Foco: aproximar o fluxo do uso real do Polymarket.

Incluir:

- escolher primeiro o mercado, nao apenas a cidade
- exibir regra, data, cidade e outcomes como objeto principal da tela
- tratar mercado selecionado como entidade central da pagina
- deixar a leitura meteorologica subordinada ao contrato escolhido

Melhor quando:

- a prioridade e tornar o produto mais util para o usuario final sem ainda virar SaaS completo

### Opcao C - Produto/SaaS

Foco: transformar a base atual em aplicacao persistente.

Incluir:

- historico salvo de analises
- usuarios e autenticacao
- armazenamento de mercados monitorados
- camada operacional e de deploy

Melhor quando:

- o nucleo analitico e o fluxo principal ja estiverem mais estaveis

## Recomendacao

A ordem mais racional daqui para frente e:

1. Opcao A em versao enxuta
2. Opcao B como refinamento de produto
3. Opcao C somente depois

Razao:

- hoje o maior risco ainda esta na interpretacao do mercado, nao na ausencia de features de SaaS
- se o contrato do mercado continuar parcialmente inferido, qualquer camada de produto em cima disso amplifica erro
- depois que a leitura do mercado estiver mais solida, a UX market-first fica muito mais natural

Observacao:

- o refinamento visual necessario para explorar o MVP ja foi feito em nivel suficiente
- o proximo ganho real vem mais de confiabilidade semantica do que de mais polish visual

## Proximo Ciclo Sugerido

### Ciclo 1 - Mercado Mais Confiavel

Objetivo:
melhorar a confiabilidade da entidade mercado sem aumentar muito o escopo.

Tasks sugeridas:

- consolidar parsing de titulo, cidade e data do mercado no backend
- classificar eventos em `valid`, `partial` e `unknown`
- explicitar no payload o nivel de certeza da interpretacao
- adicionar testes para parsing de eventos do Polymarket
- refletir esse status na UI sem prometer certeza

### Ciclo 2 - Fluxo Market-First

Objetivo:
fazer o usuario entrar pelo mercado, nao pela cidade.

Tasks sugeridas:

- listar mercados relevantes no backend
- permitir selecao direta de um contrato
- reorganizar a tela em torno do mercado escolhido
- manter cidade, modelos e historico como contexto de suporte

### Ciclo 3 - Preparacao de Produto

Objetivo:
decidir se o projeto vai mesmo virar SaaS e qual o menor passo util.

Tasks sugeridas:

- definir persistencia minima
- definir se havera contas ja no MVP seguinte
- definir se o valor principal e analise sob demanda ou acompanhamento recorrente

## Decisao Recomendada Agora

Se a meta e continuar evoluindo com qualidade, o melhor proximo prompt deve abrir um plano novo focado em confiabilidade de mercado, nao em features de plataforma.
