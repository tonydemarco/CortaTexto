# CortaTexto

App desktop (Tkinter) para **macOS** que resume um texto usando um modelo de IA
**local** (via **Ollama**), **sem ultrapassar um limite de caracteres** definido
por você — e **sem chave de API**, rodando 100% na sua máquina.

O modelo não conta caracteres com precisão, então **quem conta é o Python**
(`len()` puro é a fonte de verdade). Quando o resumo não cabe, o app pede
correções ao modelo num loop, com **tolerância progressiva** em duas rodadas, até
o texto caber na faixa aceitável. **O resultado nunca passa do limite.**

## Requisitos

- **macOS** com Python 3 (o Tkinter já vem no Python oficial do macOS).
- **Ollama** instalado e rodando: https://ollama.com (ou `brew install ollama`).
- Um **modelo** baixado, por exemplo: `ollama pull qwen3`.

Não há dependências Python de runtime — a chamada ao Ollama usa só a biblioteca
padrão (`urllib`).

## Uso

```bash
# 1. deixe o Ollama rodando e baixe um modelo (uma vez):
ollama pull qwen3

# 2. abra o app:
python3 CortaTexto.py
```

Na janela:

1. Cole o texto a resumir (há um contador do tamanho original).
2. Defina **Limite (caracteres)**, **Tolerância 1** (rodada 1), **Tolerância 2**
   (rodada 2) e **Máx. tentativas**.
3. Em **Modelo (Ollama)**, informe o modelo que você baixou (padrão: `qwen3`).
4. Clique **Resumir** (ou `Ctrl/Cmd+Enter`). Você pode **Cancelar** a qualquer
   momento (`Esc`) — o melhor resumo parcial que já cabe é preservado.
5. O resultado é **editável** com recontagem ao vivo; o contador fica vermelho
   se você ultrapassar o limite. Use **Copiar resultado** quando estiver pronto.

### Faixa de aceitação

A faixa aceita é `[limite − tolerância, limite]`. A rodada 1 usa a Tolerância 1;
se não convergir, a rodada 2 relaxa para a Tolerância 2. Ficar um pouco abaixo
do limite (dentro da tolerância) é desejável — evita pedir correções eternas só
para bater o número exato.

Se o texto **original já couber no limite**, ele é devolvido sem alterações.

## Fonte

A interface usa exclusivamente a fonte **Garoa Light**, **embutida** no próprio
`CortaTexto.py` (binário `zlib`+`base64`) e registrada em runtime via CoreText,
**sem instalar nada no sistema**. Se o registro falhar, o app abre com a fonte
padrão e mostra um aviso discreto.

## Testes

Os testes não chamam o Ollama real (usam um `chamar_llm` mockado e respostas
de exemplo) e não exigem nada além do Python:

```bash
pip install -r requirements-dev.txt   # pytest
pytest -q
```

Cobrem: contagem com acentos/espaços, aceitação imediata quando já cabe, loop de
encurtamento, expansão quando curto demais, corte de segurança, resposta
vazia/`None` falhando, validação de tolerância negativa, cancelamento
preservando o melhor parcial, e o parsing da resposta do Ollama (remoção de
`<think>…</think>`, truncamento, resposta vazia).

## Configuração

Constantes no topo de `CortaTexto.py` (fáceis de trocar): `MODELO_OLLAMA`
(`qwen3`), `OLLAMA_URL` (`http://localhost:11434/api/chat`),
`MAX_TOKENS_MIN`/`MAX_TOKENS_TETO`, `TEMPERATURA`, `TIMEOUT_API`, tolerâncias e
tentativas padrão.
