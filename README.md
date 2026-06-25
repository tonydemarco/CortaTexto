# CortaTexto

App desktop (Tkinter) para **macOS** que encurta um texto até um **limite de
caracteres** definido por você, usando um modelo de IA **local** (via **Ollama**)
— **sem chave de API** e, depois da primeira vez, **100% offline**.

O modelo não conta caracteres com precisão, então **quem conta é o Python**
(`len()`, ignorando quebras de linha, é a fonte de verdade). **O resultado nunca
passa do limite** e fica o mais próximo possível dele.

## Baixar (para usar)

Para a maioria das pessoas, é só baixar o app pronto:

➡️ **[Releases](https://github.com/tonydemarco/CortaTexto/releases)** — baixe o
`.zip`, descompacte e leia o `LEIA-ME PRIMEIRO.txt` que vem junto.

- **Requisito:** Mac com **chip Apple** (M1 ou mais novo). *Não roda em Macs Intel.*
- Como o app não é assinado pela Apple, na **primeira vez** abra com
  **clique-direito (ou Control+clique) → Abrir → Abrir**.
- No **primeiro uso**, o app baixa sozinho o motor de IA (Ollama ~178 MB +
  modelo qwen3 ~5,2 GB). É uma vez só; depois funciona sem internet.

## Como usar

1. Cole o texto a encurtar (um contador mostra o tamanho original).
2. Em **"Reduza para"**, informe o limite em caracteres.
3. Clique **Resumir** (ou `Ctrl/Cmd+Enter`). Dá para **Cancelar** a qualquer
   momento (`Esc`) — o melhor resumo parcial que já cabe é preservado.
4. O resultado é **editável**, com recontagem ao vivo. Use **Copiar resultado**
   quando estiver pronto.
5. O botão **Como funciona** abre uma explicação curta dentro do app.

A interface expõe **apenas o limite** — os demais parâmetros (tolerância,
tentativas, modelo) são constantes no código (ver [Configuração](#configuração)).
As quebras de linha (Enter) **não contam** para o limite. Se o texto original já
couber no limite, ele é devolvido sem alterações.

## Como o tamanho é garantido

O Python é dono da contagem; o LLM só escreve. O roteamento depende de quão
grande é o corte (`r = (original − limite) / original`):

- **Cortes até ~50%** → modo **frase a frase**: cada frase é encurtada
  isoladamente (nunca funde frases nem inventa dados) e o app monta
  deterministicamente o conjunto que mais se aproxima do limite. Resultado
  **fiel** ao original.
- **Cortes maiores que 50%** → **resumo livre**: o modelo reescreve com palavras
  próprias, mantendo a ordem dos assuntos, num laço com tolerância progressiva
  até caber.

Em todos os casos, um aparo final em fronteira de palavra garante que o
resultado **não ultrapasse o limite** (sem reticências).

## Rodar do código-fonte (desenvolvedores)

```bash
# 1. deixe o Ollama rodando e baixe o modelo (uma vez):
ollama pull qwen3        # https://ollama.com  (ou: brew install ollama)

# 2. abra o app:
python3 CortaTexto.py
```

Não há dependências Python de runtime — a chamada ao Ollama usa só a biblioteca
padrão (`urllib`). O Tkinter já vem no Python oficial do macOS. *(Se o Ollama ou
o modelo não estiverem presentes, o próprio app oferece baixá-los na primeira
abertura.)*

## Fonte

A interface usa exclusivamente a fonte **Garoa Light**, **embutida** no próprio
`CortaTexto.py` (binário `zlib`+`base64`) e registrada em runtime via CoreText,
**sem instalar nada no sistema**. Se o registro falhar, o app abre com a fonte
padrão e mostra um aviso discreto.

A Garoa Light é da **Just in Type** e está licenciada sob a **SIL Open Font
License 1.1** (ver [`OFL.txt`](OFL.txt)), com nome reservado "Garoa Light" — pode
ser embutida e redistribuída livremente junto com o software.

## Testes

Os testes não chamam o Ollama real (usam um `chamar_llm` mockado e respostas de
exemplo) e não exigem nada além do Python:

```bash
pip install -r requirements-dev.txt   # pytest
pytest -q
```

Cobrem, entre outros: contagem com acentos/espaços e ignorando quebras de linha,
aceitação imediata quando já cabe, o modo frase a frase (encurtamento fiel,
preservação de parágrafos, cortes sutis determinísticos), o resumo livre (ordem
preservada, remoção de números inventados, nunca usar reticências), corte de
segurança, cancelamento preservando o melhor parcial, e o parsing da resposta do
Ollama (remoção de `<think>…</think>`, truncamento, resposta vazia).

## Configuração

Constantes no topo de `CortaTexto.py` (fáceis de trocar, **só no código** — não
aparecem na interface): `MODELO_OLLAMA` (`qwen3`), `OLLAMA_URL`,
`TOLERANCIA_RODADA_1` (6) / `TOLERANCIA_RODADA_2` (20), `MAX_TENTATIVAS_PADRAO`
(6), `LIMIAR_FRASE_A_FRASE` (0.50), além de temperatura e timeout.

## Licença

- **Código**: [MIT](LICENSE) — © 2026 Tony de Marco.
- **Fonte Garoa Light**: [SIL Open Font License 1.1](OFL.txt) — © 2026 Tony de
  Marco (Just in Type), nome reservado "Garoa Light".

As dependências de runtime (não incluídas no repositório) têm licenças próprias:
**Ollama** (MIT) e o modelo **qwen3** (Apache 2.0).
