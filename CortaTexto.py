#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CortaTexto
==========

Aplicativo desktop (Tkinter) para **macOS** que resume um texto usando um modelo
de linguagem LOCAL, via Ollama (sem chave de API; roda na sua maquina),
preservando suas caracteristicas (sentido essencial, fatos, tom e estilo), sem
ultrapassar um limite de caracteres definido pelo usuario.

Como a LLM nao conta caracteres com precisao, quem conta e o Python: a contagem
e exata (len() puro -> letras, espacos, pontuacao, quebras de linha, acentos
como code points). Quando o resumo nao cabe, o programa pede correcoes a LLM
num loop ate o texto caber na faixa aceitavel.

Tolerancia progressiva (faixa = [limite - tolerancia, limite]):
    - Rodada 1: tolerancia da rodada 1 (tolerancia_1, padrao 10), editavel.
    - Rodada 2: se a rodada 1 nao convergir, relaxa para a tolerancia da
      rodada 2 (tolerancia_2, padrao 20), tambem editavel.
    O resultado NUNCA passa do limite, em nenhuma rodada.

Fonte:
    A interface usa exclusivamente a fonte "Garoa Light", que vem EMBUTIDA neste
    arquivo (binario zlib+base64 na constante _GAROA_OTF_B64) e e registrada em
    runtime, sem instalar nada no sistema. No macOS isso usa o CoreText
    (CTFontManagerRegisterGraphicsFont). Se o registro falhar, o app abre na
    fonte padrao e avisa de forma discreta.

    Como o blob foi gerado (a partir de Garoa-Light.otf):
        import base64, zlib
        blob = base64.b64encode(zlib.compress(open("Garoa-Light.otf","rb").read(), 9))

Requisitos (externos, fora do pip -- a UI abre sem eles):
    1. Ollama instalado e rodando:  https://ollama.com
    2. Um modelo baixado, por exemplo:  ollama pull qwen3
    A chamada ao Ollama usa apenas a stdlib (urllib) -- sem dependencias.

Uso:
    python3 CortaTexto.py

Testes (pytest, sem chamar a API real -- ver test_cortatexto.py):
    - contagem exata com acentos/espacos (code points);
    - aceitacao imediata quando o texto ja cabe na faixa (sem chamar a LLM);
    - loop de encurtamento quando acima do limite;
    - expansao quando curto demais;
    - corte de seguranca no ultimo recurso (palavra inteira, SEM "...", e cabe);
    - resposta vazia / None da LLM falham (nunca viram "sucesso");
    - tolerancia negativa levanta ValueError;
    - cancelamento preserva o melhor resumo parcial que ja cabe.
"""

from __future__ import annotations

import base64
import json
import os
import queue
import re
import shutil
import socket
import subprocess
import sys
import threading
import time
import tkinter as tk
import urllib.error
import urllib.request
import zlib
from dataclasses import dataclass, field
from tkinter import font as tkfont
from tkinter import messagebox, ttk
from typing import Callable, List, Optional


# ===========================================================================
# CONFIGURACOES GERAIS (faceis de trocar)
# ===========================================================================
# O app usa um modelo LOCAL via Ollama (sem chave de API; roda na sua maquina).
MODELO_OLLAMA = "qwen3"          # modelo do Ollama -- editavel na interface
OLLAMA_URL = "http://localhost:11434/api/chat"   # endpoint local do Ollama
OLLAMA_BASE = "http://127.0.0.1:11434"           # raiz da API local (autoinstalador)
OLLAMA_DOWNLOAD_URL = "https://ollama.com/download/Ollama-darwin.zip"  # app Mac (zip)
OLLAMA_APP_DIRS = ("~/Applications/Ollama.app", "/Applications/Ollama.app")
ESPACO_MINIMO_BYTES = 8 * 1024 ** 3              # ~8 GB livres p/ Ollama + qwen3
TIMEOUT_DOWNLOAD = 900.0         # s para baixar o Ollama (~178 MB)
MAX_TOKENS_MIN = 2048            # piso de tokens de saida
MAX_TOKENS_TETO = 16384          # teto de tokens de saida (evita custo absurdo)
TEMPERATURA = 0.3                # baixa = mais fiel e estavel
TIMEOUT_API = 60.0               # segundos por requisicao (falha rapido na rede)
TOLERANCIA_RODADA_1 = 6          # caracteres a menos aceitos na 1a rodada (otimo: sweep)
TOLERANCIA_RODADA_2 = 20         # caracteres a menos aceitos na 2a rodada
MAX_TENTATIVAS_PADRAO = 6        # total de tentativas (otimo: sweep -- mais tentativas afunda)
LIMIAR_FRASE_A_FRASE = 0.50      # r=(orig-limite)/orig <= isto -> frase-a-frase (FIEL:
                                 # encurta cada frase e elimina inteiras, sem JAMAIS misturar).
                                 # Cortes DRASTICOS (r > 50%) vao pelo GERATIVO (resumo livre,
                                 # corrido) -- opcao do usuario: acima de 50% o fiel dropava
                                 # frases demais e o texto 'pulava de assunto'; o resumo livre
                                 # flui melhor. (Tambem cai no gerativo se a montagem=None.)
MIN_FRASE_ALVO = 16              # piso de caracteres ao encurtar uma frase (abaixo disto vira
                                 # fragmento sem sentido); a meta por frase nunca pede menos.


# ===========================================================================
# ERROS
# ===========================================================================
class ErroResumo(Exception):
    """Erro de logica/uso durante o resumo (mensagem ja amigavel em PT)."""


class RespostaVaziaError(ErroResumo):
    """A LLM nao devolveu texto utilizavel."""


class ErroOllamaIndisponivel(ErroResumo):
    """Falha de CONFIGURACAO do Ollama (servico fora do ar ou modelo ausente):
    o usuario precisa agir. Diferente de erros transitorios (timeout/truncamento
    /resposta ruim), nao deve ser silenciosamente contornado pelo fallback."""


# ===========================================================================
# LOGICA (independente da interface, testavel com mock)
# ===========================================================================
def contar(texto: str) -> int:
    """Contagem exata de caracteres, IGNORANDO quebras de linha (Enter): elas sao
    formatacao, nao entram no limite (pedido do usuario). Fonte de verdade do
    programa -- a LLM nunca decide o tamanho."""
    return len(texto) - texto.count("\n") - texto.count("\r")


def _milhar(n: int) -> str:
    """Formata inteiro com ponto de milhar (pt-BR): 1782 -> '1.782'."""
    return f"{n:,}".replace(",", ".")


@dataclass
class Resultado:
    texto: str
    caracteres: int
    limite: int
    tentativas: int
    sucesso: bool                 # True se ficou (0 < n <= limite) e nao cancelado
    cortado: bool = False         # True se houve corte de seguranca
    cancelado: bool = False       # True se o usuario cancelou
    historico: List[int] = field(default_factory=list)


_SISTEMA = (
    "Voce e um redator que RESUME com FLUIDEZ. Escreva um texto NOVO, claro e "
    "CORRIDO, com SUAS proprias palavras (NAO precisa preservar a redacao nem a "
    "gramatica do original). MAS SIGA A ORDEM em que os assuntos aparecem no "
    "texto: pode reescrever e fundir frases VIZINHAS, porem NAO embaralhe os fatos "
    "nem traga para o inicio coisas que estavam no fim -- mantenha a mesma "
    "sequencia de assuntos do original. Aproveite QUASE TODO o limite de "
    "caracteres, ficando o mais PROXIMO possivel dele SEM NUNCA ultrapassa-lo, e "
    "mantendo o MAXIMO de informacao (fatos, nomes e numeros) que couber. REGRA "
    "FIRME: use SOMENTE fatos presentes no texto; NUNCA acrescente numeros, idades, "
    "datas ou detalhes que nao estejam la (na duvida, omita). Mantenha o idioma e "
    "um tom natural. Responda APENAS com o texto, sem comentarios, aspas ou rotulos."
)


_SISTEMA_FRASES = (
    "Voce e um editor que ENCURTA frases, UMA POR VEZ, sem nunca fundir nem "
    "reordenar frases. Cada frase vem com uma META de caracteres entre "
    "parenteses; comprima a frase para chegar PERTO dessa meta, cortando "
    "adjetivos, adverbios, oracoes acessorias, exemplos e detalhes secundarios. "
    "REGRAS: nao junte duas frases; nao crie frases novas; mantenha EXATAMENTE "
    "todas as palavras com inicial maiuscula (nomes proprios) e todos os "
    "numeros; o resultado de cada frase deve continuar gramatical. Responda UMA "
    "frase por linha, no formato 'N: frase', usando o MESMO numero e a MESMA "
    "ordem da entrada."
)


def _alvos_encurtamento(frases: List[str], limite: int) -> List[int]:
    """Meta de caracteres por frase, PROPORCIONAL ao limite, para que a soma das
    frases encurtadas caiba (descontando os espacos de juncao) e POUCAS ou
    NENHUMA frase inteira precise ser eliminada depois -- o que e o que deixa o
    texto 'truncado' (referencias penduradas, saltos). Frases curtas nao descem
    abaixo de MIN_FRASE_ALVO. Se ja cabe, devolve os tamanhos atuais (sem cortar)."""
    n = len(frases)
    tamanhos = [contar(f) for f in frases]
    total = sum(tamanhos)
    disp = max(1, limite - (n - 1))            # espaco util (tira as juncoes)
    if total <= disp:
        return tamanhos
    ratio = disp / total
    return [max(MIN_FRASE_ALVO, round(t * ratio)) for t in tamanhos]


def _prompt_encurtar_frases(frases: List[str], alvos: List[int]) -> str:
    numeradas = "\n".join(
        f"{i + 1}: (~{a} caracteres) {f}" for i, (f, a) in enumerate(zip(frases, alvos))
    )
    return (
        "Encurte CADA frase abaixo para chegar PERTO da meta de caracteres "
        "indicada entre parenteses (uma por linha, formato 'N: frase', mesma "
        "numeracao e ordem). Corte adjetivos, oracoes acessorias e detalhes "
        "secundarios. NAO junte frases, NAO reordene, NAO invente. Mantenha "
        "EXATAMENTE todas as palavras com inicial maiuscula e todos os numeros.\n\n"
        + numeradas
    )


def _palavras_alvo(limite: int) -> int:
    """Aproxima quantas palavras cabem em `limite` (~6 chars/palavra com espaco).
    O qwen3 controla contagem de PALAVRAS muito melhor que de caracteres, entao
    damos essa meta -- decisivo em cortes drasticos (limite pequeno)."""
    return max(1, round(limite / 6))


def _prompt_inicial(texto: str, limite: int, minimo: int) -> str:
    orig = len(texto)
    remover = max(0, orig - limite)
    palavras = _palavras_alvo(limite)
    return (
        f"O texto original abaixo tem {orig} caracteres e o limite e {limite} "
        f"(cerca de {palavras} palavras). Escreva um RESUMO novo e CORRIDO, com "
        f"suas proprias palavras, entre {minimo} e {limite} caracteres -- aprox. "
        f"{palavras} palavras --, o mais PROXIMO possivel de {limite} sem JAMAIS "
        f"ultrapassar. Condense com suas proprias palavras SEGUINDO A ORDEM dos "
        f"assuntos do original (nao embaralhe nem antecipe o que vem depois; pode "
        f"reescrever e fundir frases vizinhas). Mantenha o MAXIMO de fatos, nomes e "
        f"numeros que couber e NAO invente nada. Responda apenas com o texto.\n\n"
        f"TEXTO:\n{texto}"
    )


def _prompt_encurtar(resumo: str, atual: int, limite: int, minimo: int) -> str:
    excesso = atual - limite
    palavras = _palavras_alvo(limite)
    return (
        f"O texto abaixo tem {atual} caracteres e ultrapassou o limite em "
        f"{excesso}. Reescreva-o mais ENXUTO, entre {minimo} e {limite} caracteres "
        f"(cerca de {palavras} palavras), o mais PROXIMO possivel de {limite} sem "
        f"passar. Pode reformular e fundir frases vizinhas, MAS mantenha a ORDEM "
        f"dos assuntos (nao embaralhe); preserve o maximo de fatos, nomes e numeros "
        f"e NAO invente. Responda apenas com o novo texto.\n\nTEXTO ATUAL:\n{resumo}"
    )


def _prompt_expandir(texto: str, resumo: str, atual: int, limite: int,
                     minimo: int) -> str:
    faltam = limite - atual
    return (
        f"Sua versao atual do resumo tem apenas {atual} caracteres: faltam cerca "
        f"de {faltam} para chegar perto do limite de {limite}. Reescreva-a MAIS "
        f"COMPLETA, entre {minimo} e {limite} caracteres, o mais PROXIMO possivel "
        f"de {limite} sem ultrapassar, reincorporando fatos, nomes e numeros do "
        f"TEXTO ORIGINAL abaixo que ficaram de fora -- com suas proprias palavras, "
        f"NA MESMA ORDEM dos assuntos do original, sem inventar nada. Responda "
        f"apenas com o novo texto.\n\nTEXTO ORIGINAL:\n{texto}\n\nSUA VERSAO "
        f"ATUAL:\n{resumo}"
    )


def _cortar(texto: str, limite: int) -> str:
    """Corte de seguranca (ultimo recurso) preservando PALAVRAS INTEIRAS e SEM
    reticencias -- o resultado NUNCA termina em '...'. Recua ate a ultima palavra
    inteira que cabe e tira pontuacao/espaco solto do fim. Se nem a 1a palavra
    couber, faz um fatiamento cru. A invariante 'nunca passa do limite' e sempre
    respeitada."""
    if contar(texto) <= limite:
        return texto
    cortado = texto[:limite]
    if " " in cortado:
        cortado = cortado[: cortado.rfind(" ")]      # termina em palavra inteira
    cortado = cortado.rstrip(" ,;:-—\t\n")            # sem pausa/espaco solto no fim
    return (cortado or texto[:limite])[:limite]


# Aberturas: posicoes onde uma aspa dupla deve ser “ (U+201C) em vez de ” (U+201D).
_ABRE_ASPA = set(" \t\n\r([{“‘—–«")


def _curvar_aspas(s: str) -> str:
    """Troca toda aspa dupla reta U+0022 (\") por aspa curva tipografica:
    “ (U+201C) em posicao de ABERTURA e ” (U+201D) em posicao de FECHAMENTO.
    Necessario porque a fonte Garoa Light desenha um COPO no glifo U+0022 -- ele
    nunca deve aparecer na interface. A troca e 1:1 em code points, entao nao
    altera a contagem de caracteres."""
    if '"' not in s:
        return s
    res: List[str] = []
    for ch in s:
        if ch == '"':
            ant = res[-1] if res else ""
            res.append("“" if (ant == "" or ant in _ABRE_ASPA) else "”")
        else:
            res.append(ch)
    return "".join(res)


# Texto-placeholder do Manual (so para validar a janela; o texto real vira depois).
_MANUAL = """O CortaTexto encurta um texto para o tamanho que você quiser. Usa Inteligência Artificial, mas nenhuma informação sai do seu computador. Tecnicamente, é uma orquestração em torno do LLM qwen3 rodando localmente, via Ollama: um prompt de sistema com regras de edição, somado a guardrails em Python que contam os caracteres e garantem que o resultado nunca ultrapasse o limite.

COMO USAR
Cole o texto no campo de cima.
Em "Reduza para", informe o limite em caracteres.
Clique em Resumir (ou tecle Enter).
Acompanhe o processo no rodapé; no fim aparece o número de caracteres e tentativas.
Edite o resultado se quiser e clique em Copiar resultado.

BOM SABER
Cortes leves a moderados: o CortaTexto encurta as frases e, quando precisa, elimina frases inteiras — sem misturar nem inventar.
Cortes drásticos: viram um resumo livre.
Cortes muito drásticos: podem conter imprecisões.
"""


# ---------------------------------------------------------------------------
# APARAR PARA CABER (corte pequeno): o LLM micro-edita o texto (sinonimos
# curtos, tira artigos/palavras superfluas, sem apagar frases) e costuma ficar
# um pouco ACIMA do limite; o PYTHON entao apara o excesso removendo o minimo de
# trechos acessorios (heuristica pura), sem decapitar a 1a frase nem o fecho.
# Assim a qualidade vem do LLM e a contagem fica garantida pelo Python.
# ---------------------------------------------------------------------------
def _ocupa(i: int, j: int, usados: List[tuple]) -> bool:
    """True se o intervalo [i, j) se sobrepoe a algum ja aceito."""
    return any(i < y and x < j for (x, y) in usados)


def _e_subsequencia(curta: str, original: str) -> bool:
    """True se `curta` e o `original` APENAS com palavras apagadas (mesma ordem,
    nada reescrito) -- compara por palavra, ignorando caixa/pontuacao/espacos.
    Prova de DELECAO PURA (fidelidade): se falhar, houve reescrita/reordenacao."""
    pal = lambda s: re.findall(r"\w+", s.lower())
    o, c = pal(original), pal(curta)
    i = 0
    for w in c:
        while i < len(o) and o[i] != w:
            i += 1
        if i >= len(o):
            return False
        i += 1
    return True


# Marcadores de DISCURSO (abertura/transicao) dispensaveis: removiveis SEM mudar
# o sentido quando isolados por virgula. Lista FECHADA (seguro estender).
_DISCURSO = (
    "tecnicamente", "basicamente", "essencialmente", "na verdade", "de fato",
    "em resumo", "em suma", "por fim", "alem disso", "além disso", "inclusive",
    "ou seja", "no entanto", "em geral", "alias", "aliás", "portanto",
    "na pratica", "na prática",
)


def _spans_finos(texto: str) -> List[tuple]:
    """Trechos FINOS (poucos caracteres), gramaticalmente SEGUROS de apagar, para
    o subset-sum encostar no limite em cortes SUTIS. So delecao pura; categorias
    CURADAS (NAO inclui pron-redundante/artigo/coordenada, que distorcem mesmo
    sendo subsequencia)."""
    spans: List[tuple] = []
    baixo = texto.lower()
    # (a) marcador de discurso ABRINDO frase, seguido de virgula -> "Marcador, "
    for m in re.finditer(r"(?:^|[.!?…]\s+)([A-Za-zÀ-ÿ][A-Za-zÀ-ÿ ]*?),\s+", texto):
        if m.group(1).strip().lower() in _DISCURSO:
            spans.append((m.start(1), m.end()))
    # (b) marcador de discurso ENTRE virgulas -> ", marcador"
    for loc in _DISCURSO:
        for m in re.finditer(r",\s+" + re.escape(loc) + r"(?=,)", baixo):
            spans.append(m.span())
    # (c) "em torno" redundante, PRESERVANDO o artigo seguinte (do/da/dos/das)
    for m in re.finditer(r"\s+em torno(?=\s+d[oae]s?\b)", baixo):
        spans.append(m.span())
    # (d) "voce" antes de verbo de desejo -> "que quiser"
    for m in re.finditer(r"\s+voc[eê](?=\s+(?:quiser|quiseres|desejar|precisar|gostar))",
                         baixo):
        spans.append(m.span())
    # (e) artigo no INICIO do texto antes de NOME PROPRIO: "O CortaTexto"->"CortaTexto"
    m = re.match(r"(?:O|A|Os|As|Um|Uma)\s+(?=[A-ZÀ-Ý])", texto)
    if m:
        spans.append(m.span())
    return spans


def _spans_heuristicos(texto: str) -> List[tuple]:
    """Fallback deterministico (sem LLM): aponta trechos acessorios em nivel de
    CLAUSULA -- apartes entre parenteses/colchetes/travessoes, oracoes entre
    virgulas e, por ultimo, sentencas inteiras do meio. A granularidade fina
    permite o subset-sum cair perto do topo do limite mesmo sem ajuda do modelo.
    Ordenados do MENOS para o MAIS essencial."""
    usados: List[tuple] = []
    spans: List[tuple] = []

    def add(i: int, j: int) -> None:
        if j > i and not _ocupa(i, j, usados):
            usados.append((i, j))
            spans.append((i, j))

    # 0) trechos FINOS seguros (marcador de abertura, "em torno", "voce"...) -- PRIMEIRO,
    # para a granularidade fina vencer a grossa na sobreposicao (subset-sum encosta mais).
    for i, j in _spans_finos(texto):
        add(i, j)
    # 1) apartes entre parenteses/colchetes/travessoes (sempre dispensaveis)
    for m in re.finditer(r"\s*\([^)]*\)|\s*\[[^\]]*\]|\s+—[^—]*—", texto):
        add(*m.span())
    # 2) apostos/oracoes entre virgulas no meio da frase (", ... ,")
    for m in re.finditer(r",[^,;.!?\n]+(?=,)", texto):
        add(*m.span())
    # 3) oracoes apos virgula ate o fim da sentenca (", ... .")
    for m in re.finditer(r",[^,;.!?\n]+(?=[.!?\n])", texto):
        add(*m.span())
    # 4) sentencas do meio (mais grosseiro; so entra onde nao houve clausula)
    sent = list(re.finditer(r"[^.!?\n]+[.!?]+", texto))
    if len(sent) > 2:
        for m in sent[1:-1]:                 # exclui 1a e ultima sentenca
            add(*m.span())
    return spans


def _mapa_subconjuntos(comprimentos: List[int]) -> dict:
    """Subset-sum 0/1: mapeia cada SOMA alcancavel -> um conjunto de indices que
    a produz. Usado para escolher quais trechos remover de modo que o resultado
    fique o mais perto possivel do limite (por cima)."""
    alcancavel = {0: []}
    for i, c in enumerate(comprimentos):
        novos = {}
        for s, idxs in alcancavel.items():
            ns = s + c
            if ns not in alcancavel and ns not in novos:
                novos[ns] = idxs + [i]
        alcancavel.update(novos)
    return alcancavel


def _corrigir_maiusculas(s: str) -> str:
    """Garante inicial maiuscula no comeco do texto e apos fim de frase. Util
    quando uma remocao expoe um novo inicio de frase que estava em minuscula no
    meio do periodo original (ex.: cair de '..., as series' para 'As series')."""
    def subir(m: "re.Match") -> str:
        return m.group(1) + m.group(2).upper()
    s = re.sub(r"(^\s*)([^\W\d_])", subir, s)                 # inicio do texto
    s = re.sub(r"([.!?][\"'»”’)\]]?\s+)([^\W\d_])", subir, s)  # apos fim de frase
    return s


def _remover_e_costurar(texto: str, ranges: List[tuple]) -> str:
    """Remove os intervalos dados e costura as juncoes para o texto seguir
    natural: colapsa espacos, tira espaco antes de pontuacao, conserta os
    artefatos tipicos de remover uma oracao do meio de um periodo -- pontuacao
    de pausa duplicada (', ,' -> ','), virgula encostada no fim de frase
    (', .' -> '.') e pontuacao orfa no inicio -- e corrige a capitalizacao
    de inicio de frase exposta pela remocao."""
    if not ranges:
        return texto
    partes: List[str] = []
    pos = 0
    for a, b in sorted(ranges):
        if a < pos:                 # seguranca contra sobreposicao
            a = pos
        if b <= a:
            continue
        partes.append(texto[pos:a])
        pos = b
    partes.append(texto[pos:])
    s = "".join(partes)
    s = re.sub(r"[ \t]{2,}", " ", s)                 # espacos duplicados nas juntas
    s = re.sub(r"\s+([,.;:!?…)\]])", r"\1", s)   # espaco/quebra antes de pontuacao
    s = re.sub(r"([,;:])(?:\s*[,;:])+", r"\1", s)     # pausa repetida -> uma so (',,'->',')
    s = re.sub(r"[,;:]+(\s*[.!?])", r"\1", s)         # virgula/; antes de fim de frase
    s = re.sub(r"([(\[])\s*[,;:.!?]+\s*", r"\1", s)   # pontuacao logo apos abre-parentese
    s = re.sub(r"([([]) +", r"\1", s)                # espaco apos abre-parentese
    s = re.sub(r"(^|[\n.!?]\s*)[,;:]+\s*", r"\1", s)  # pontuacao orfa no inicio/apos frase
    s = re.sub(r" +\n", "\n", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return _corrigir_maiusculas(s.strip())


def _span_acessorio_do_numero(texto: str, i: int, j: int) -> tuple:
    """Intervalo a remover em volta de um numero (em [i, j)) que foi inventado: a
    clausula acessoria que o contem (entre virgulas/pausas ou fronteiras de
    frase), se for curta; senao, so um trecho justo (conector + numero + unidade,
    ex.: 'de 18 anos')."""
    esq, dir = set(",;:.!?\n(["), set(",;:.!?\n)]")
    a = i
    while a > 0 and texto[a - 1] not in esq:
        a -= 1
    b = j
    while b < len(texto) and texto[b] not in dir:
        b += 1
    if b - a <= 30:
        return (a, b)                                # clausula curta: remove inteira
    s = i                                            # clausula longa: corte justo
    m = re.search(r"\b(?:cerca de|aos|de|com|a|h[áa]|por)\s+$", texto[max(0, i - 12):i])
    if m:
        s = i - len(m.group(0))
    e = j
    u = re.match(r"\s*(?:anos?|meses|dias?|semanas?|horas?|minutos?|reais|mil|"
                 r"milh(?:oes|ões)|por cento|%)\b", texto[j:], re.IGNORECASE)
    if u:
        e = j + u.end()
    return (s, e)


def _remover_numeros_inventados(texto: str, original: str) -> str:
    """Remove do resumo numeros que NAO existem no `original` -- o fluxo GERATIVO
    livre as vezes 'inventa' idades/datas (ex.: 'de 18 anos'). Tira o numero com
    um pequeno trecho acessorio em volta e costura a pontuacao. Numeros REAIS
    (presentes no original: 68, 1985...) ficam intactos. Numeros por extenso nao
    sao detectados. No fluxo fiel (frase-a-frase) e no-op: todo numero ja vem do
    original."""
    nums = set(re.findall(r"\d+", original))
    inventados = [m for m in re.finditer(r"\d+", texto) if m.group(0) not in nums]
    if not inventados:
        return texto
    ranges = [_span_acessorio_do_numero(texto, m.start(), m.end()) for m in inventados]
    return _remover_e_costurar(texto, ranges)


def _limites_abertura_fecho(texto: str) -> tuple:
    """Devolve (fim_da_1a_frase, inicio_da_ultima_frase) para proteger o lead e
    o desfecho na deleção. Se houver 0-1 frases, nao protege nada (0, len)."""
    sent = list(re.finditer(r"[^.!?\n]*[.!?]+(?:\s|$)", texto))
    if len(sent) < 3:
        return (0, len(texto))
    return (sent[0].end(), sent[-1].start())


def _aparar_para_caber(texto: str, limite: int, faixa: int,
                       fiel: bool = False) -> Optional[str]:
    """Apara `texto` para caber em `limite` removendo o MINIMO de trechos
    acessorios (heuristica pura, SEM LLM), o mais perto possivel do topo, sem
    decapitar a 1a frase nem cortar o fecho. Devolve o texto aparado (sempre <=
    limite) ou None se nao der para caber so removendo acessorios.
    Com `fiel=True` (caminho frase-a-frase): NUNCA remove trecho com NOME PROPRIO
    ou NUMERO e so aceita candidato que seja DELECAO PURA (subsequencia) -- assim
    o corte SUTIL encosta no limite sem distorcer. Sem `fiel` (caminho gerativo,
    `_encolher_limpo`) mantem o comportamento antigo."""
    excesso = len(texto) - limite
    if excesso <= 0:
        return texto                                # ja cabe
    spans = _spans_heuristicos(texto)
    if fiel:
        # GUARDRAIL: nunca remover trecho com NOME PROPRIO ou NUMERO (ex.: impede
        # apagar ', via Ollama'); so sobra o que e' realmente acessorio.
        nomes = _nomes_proprios(texto)
        def _seguro(i: int, j: int) -> bool:
            trecho = texto[i:j]
            if any(ch.isdigit() for ch in trecho):
                return False
            return not any(nome in trecho for nome in nomes)
        spans = [(a, b) for (a, b) in spans if _seguro(a, b)]
    # QUALIDADE: nao decapitar a abertura nem cortar o fecho. Apara do corpo; so
    # mexe na 1a/ultima frase se nao houver material suficiente no meio. EXCECAO:
    # os trechos FINOS (marcador de abertura, "em torno", artigo inicial...) sao
    # seguros mesmo na 1a/ultima frase (nao decapitam) -> nao os excluir, senao
    # um corte sutil e' forcado a remover uma clausula grande do meio.
    fim_1a, ini_ult = _limites_abertura_fecho(texto)
    exemptos = set(_spans_finos(texto)) if fiel else set()
    corpo = [(a, b) for (a, b) in spans
             if (a >= fim_1a and b <= ini_ult) or (a, b) in exemptos]
    if sum(b - a for a, b in corpo) >= excesso:
        spans = corpo
    spans = spans[:80]                              # limita o custo do subset-sum
    comprimentos = [b - a for a, b in spans]
    if sum(comprimentos) < excesso:
        return None                                 # nao da para aparar so com acessorios
    alcancavel = _mapa_subconjuntos(comprimentos)
    # Janela de somas plausiveis: remover ~`excesso` (ficar no topo). As costuras
    # encurtam ~1 char por junta, entao admitimos somas um pouco abaixo tambem; a
    # medicao real por len() decide (so aceita candidato <= limite).
    margem = len(spans) + 2
    somas = sorted(s for s in alcancavel
                   if excesso - margem <= s <= excesso + faixa + margem)
    if not somas:
        somas = sorted(s for s in alcancavel if s >= excesso)[:1]
    melhor_cand: Optional[str] = None
    for s in somas:
        ranges = [spans[i] for i in alcancavel[s]]
        cand = _remover_e_costurar(texto, ranges)
        L = len(cand)
        # no modo fiel, so aceita DELECAO PURA (subsequencia) -- nunca distorce.
        if fiel and not _e_subsequencia(cand, texto):
            continue
        if 0 < L <= limite and (melhor_cand is None or L > len(melhor_cand)):
            melhor_cand = cand                      # o maior que CABE (mais no topo)
    return melhor_cand


# Palavras funcionais (em minuscula) que, mesmo com inicial maiuscula por
# abrirem frase, podem ser descartadas ao encurtar -- nao sao nomes proprios.
_PALAVRAS_COMUNS = {
    "a", "o", "as", "os", "um", "uma", "uns", "umas", "e", "de", "da", "do",
    "das", "dos", "em", "na", "no", "nas", "nos", "ao", "aos", "à", "às",
}


def _dividir_frases(texto: str) -> List[str]:
    """Divide o texto em FRASES (texto + pontuacao final), preservando a ordem.
    A montagem depois junta com um espaco. O fragmento final sem pontuacao
    tambem vira uma frase."""
    frases: List[str] = []
    pos = 0
    for m in re.finditer(r"[.!?…]+(?=\s|$)", texto):
        trecho = texto[pos:m.end()].strip()
        if trecho:
            frases.append(trecho)
        pos = m.end()
    resto = texto[pos:].strip()
    if resto:
        frases.append(resto)
    return frases


def _fatiar(texto: str) -> tuple:
    """Como _dividir_frases, mas tambem devolve `quebras`: para cada frase,
    quantas QUEBRAS DE LINHA vinham logo depois dela no original. Serve para
    reconstruir os PARAGRAFOS na montagem (preservar o '\\n' do autor)."""
    frases, spans = [], []
    pos = 0
    cortes = [m.end() for m in re.finditer(r"[.!?…]+(?=\s|$)", texto)] + [len(texto)]
    for c in cortes:
        bruto = texto[pos:c]
        s = bruto.strip()
        if s:
            ini = pos + (len(bruto) - len(bruto.lstrip()))
            frases.append(s)
            spans.append((ini, ini + len(s)))
        pos = c
    quebras = []
    for i, (_, fim) in enumerate(spans):
        prox = spans[i + 1][0] if i + 1 < len(spans) else len(texto)
        quebras.append(texto[fim:prox].count("\n"))
    return frases, quebras


def _tokens_chave(frase: str) -> tuple:
    """(palavras com inicial MAIUSCULA que NAO sao funcionais, numeros) da frase
    -- o que NAO pode sumir ao encurta-la. Inclui nomes proprios mesmo no inicio
    da frase; exclui artigos/preposicoes comuns (ver _PALAVRAS_COMUNS)."""
    nums = set(re.findall(r"\d+", frase))
    maius = set()
    for tok in re.finditer(r"\S+", frase):
        w = tok.group(0).strip(".,;:!?()[]{}\"'«»“”‘’…-—/")
        if len(w) >= 2 and w[:1].isupper() and w.lower() not in _PALAVRAS_COMUNS:
            maius.add(w)
    return maius, nums


def _nomes_proprios(texto: str) -> set:
    """Nomes proprios do texto: palavras com inicial maiuscula que aparecem em
    posicao NAO-inicial de frase em ALGUM lugar -- logo, nomes de verdade (Kimi,
    Mercedes, Bolonha...), e NAO 'Bom'/'Quando'/'Mas', que so estao maiusculas
    por abrirem a frase. Serve de trava de fidelidade GLOBAL ao encurtar: permite
    o modelo reescrever o inicio das frases (enxugar mais) sem reverter a frase
    toda, mas exige que todo NOME continue presente."""
    nomes = set()
    for m in re.finditer(r"\S+", texto):
        w = m.group(0).strip(".,;:!?()[]{}\"'«»“”‘’…-—/")
        if len(w) >= 2 and w[:1].isupper() and w.lower() not in _PALAVRAS_COMUNS:
            j = m.start() - 1
            while j >= 0 and texto[j].isspace():
                j -= 1
            if j >= 0 and texto[j] not in ".!?…":     # nao e inicio de frase
                nomes.add(w)
    return nomes


def _frase_fiel(original: str, curta: str, nomes: Optional[set] = None) -> bool:
    """True se `curta` preserva os NOMES e numeros de `original`. Com `nomes`
    (conjunto global de nomes proprios do texto, ver `_nomes_proprios`), exige so
    os nomes REAIS presentes na frase -- liberando o modelo a enxugar inicios de
    frase. Sem `nomes`, cai no comportamento por-frase (toda maiuscula conta)."""
    if set(re.findall(r"\d+", original)) - set(re.findall(r"\d+", curta)):
        return False
    if nomes is None:
        req = _tokens_chave(original)[0]
    else:
        toks = {t.strip(".,;:!?()[]{}\"'«»“”‘’…-—/") for t in original.split()}
        req = {n for n in nomes if n in toks}
    return all(w in curta for w in req)


def _parse_frases_numeradas(resp: str, n: int) -> dict:
    """Le a resposta no formato 'N: frase' (uma por linha) -> {indice0: frase}.
    Linhas fora do padrao ou do intervalo sao ignoradas (a frase volta ao
    original na montagem)."""
    d = {}
    for linha in (resp or "").splitlines():
        m = re.match(r"\s*(\d+)\s*[:.)\-]\s*(.+)$", linha)
        if m:
            idx = int(m.group(1)) - 1
            if 0 <= idx < n and m.group(2).strip():
                d[idx] = m.group(2).strip()
    return d


def _encurtar_frases(frases: List[str], chamar_llm: Callable[[str, str], str],
                     limite: int) -> List[str]:
    """UMA chamada ao LLM para encurtar CADA frase ao seu alvo proporcional ao
    `limite` (`_alvos_encurtamento`) -- assim quase tudo cabe encurtando e quase
    nada precisa ser dropado inteiro (o que truncaria o texto). Cada frase e
    tratada ISOLADA (nunca funde). Usa a versao encurtada SO se ela preserva
    nomes e numeros daquela frase; senao mantem a frase ORIGINAL inteira. Se o
    LLM falhar (transitorio), todas voltam ao original; Ollama fora propaga."""
    alvos = _alvos_encurtamento(frases, limite)
    nomes = _nomes_proprios(" ".join(frases))   # trava de fidelidade GLOBAL
    try:
        resp = chamar_llm(_SISTEMA_FRASES, _prompt_encurtar_frases(frases, alvos))
    except ErroOllamaIndisponivel:
        raise
    except ErroResumo:
        resp = ""
    curtas = _parse_frases_numeradas(resp, len(frases))
    saida = []
    for i, original in enumerate(frases):
        c = curtas.get(i)
        if c:  # o modelo as vezes ECOA a meta "(~N caracteres)" no inicio: remove
            c = re.sub(r"^\s*\(\s*~?\s*\d+[^)]*\)\s*", "", c).strip()
        # so adota a versao curta se for FIEL (mantem nomes/numeros) e REALMENTE
        # menor; senao mantem a original (sem reescrever a toa).
        usar = c and _frase_fiel(original, c, nomes) and contar(c) < contar(original)
        saida.append(c if usar else original)
    return saida


def _montar_frases(originais: List[str], curtas: List[str], limite: int,
                   quebras: Optional[List[int]] = None) -> Optional[str]:
    """Monta as frases NA ORDEM, FIEL (nunca mistura/reordena), o mais perto
    possivel do limite COM O MINIMO DE EDICAO. Preferencia, nesta ordem:
      (1) tudo ORIGINAL, se ja couber -> nao mexe em nada;
      (2) encurtar o MENOR numero de frases que faca caber (subset-sum nas
          ECONOMIAS de cada encurtamento), mantendo TODAS as frases e o resto
          no original -- evita aparar 'um pouquinho de cada' e despencar bem
          abaixo do limite num corte leve;
      (3) so se nem encurtando tudo couber, ELIMINA frases INTEIRAS do miolo
          (preservando 1a e ultima), usando as versoes encurtadas.
    `quebras` (de `_fatiar`): nº de quebras de linha apos cada frase no original;
    sao REINSERIDAS entre as frases mantidas (preserva paragrafos, mesmo dropando
    o miolo). Devolve None se nem a 1a+ultima (encurtadas) cabem."""
    if not originais:
        return None
    n = len(originais)

    def sep(a: int, b: int) -> str:               # separador entre mantidas a e b
        q = max(quebras[a:b]) if (quebras and b > a) else 0
        return "\n\n" if q >= 2 else ("\n" if q == 1 else " ")

    def juntar(idxs, escolha):
        idxs = sorted(idxs)
        if not idxs:
            return ""
        ped = [escolha(idxs[0])]
        for k in range(1, len(idxs)):
            ped.append(sep(idxs[k - 1], idxs[k]))
            ped.append(escolha(idxs[k]))
        return "".join(ped)

    todas = list(range(n))
    orig = lambda i: originais[i]
    full = juntar(todas, orig)
    if contar(full) <= limite:
        return full                               # (1) cabe sem encurtar nada

    # (2) encurtar o MINIMO de frases. Duas opcoes FIEIS, fica-se com a que
    # encosta MAIS no limite (importante em corte SUTIL: o modelo as vezes
    # comprime uma frase demais e o resultado despenca; o aparo de clausula da
    # uma opcao mais fina):
    #   (2a) subset-sum nas ECONOMIAS: encurta o menor conjunto de frases >= excesso;
    #   (2b) APARO de clausulas acessorias do original (`_aparar_para_caber`).
    economia = [contar(originais[i]) - contar(curtas[i]) for i in range(n)]
    excesso = contar(full) - limite
    candidatos = []
    idx_red = [i for i in range(n) if economia[i] > 0]
    if idx_red:
        comps = [economia[i] for i in idx_red]
        mapa = _mapa_subconjuntos(comps)
        viaveis = sorted(s for s in mapa if s >= excesso)
        if viaveis:
            encurtar = {idx_red[k] for k in mapa[viaveis[0]]}
            candidatos.append(juntar(
                todas, lambda i: curtas[i] if i in encurtar else originais[i]))
    aparado = _aparar_para_caber(full, limite, max(20, limite // 4), fiel=True)
    if aparado and 0 < contar(aparado) <= limite:
        candidatos.append(aparado)
    if candidatos:
        return max(candidatos, key=contar)        # o mais perto do limite (por baixo)

    # (3) nem encurtando tudo cabe -> elimina frases INTEIRAS (usa as encurtadas)
    curta = lambda i: curtas[i]
    cand = juntar(todas, curta)
    if contar(cand) <= limite:
        return cand                               # tudo encurtado cabe
    protegidas = {0, n - 1}
    if contar(juntar(sorted(protegidas), curta)) > limite:
        return None                               # nem abertura+fecho cabem
    miolo = [i for i in todas if i not in protegidas]
    comps = [contar(curtas[i]) + 1 for i in miolo]   # +1 pela juncao
    excesso2 = contar(cand) - limite
    alcancavel = _mapa_subconjuntos(comps)
    melhor = None
    for s in sorted(x for x in alcancavel if x >= excesso2)[:10]:
        dropar = {miolo[i] for i in alcancavel[s]}
        c = juntar([i for i in todas if i not in dropar], curta)
        L = contar(c)
        if 0 < L <= limite and (melhor is None or L > contar(melhor)):
            melhor = c                            # o que mantem MAIS conteudo
    return melhor


def _encolher_limpo(texto: str, limite: int, faixa: int) -> tuple:
    """Reduz `texto` para <= limite da forma MAIS limpa possivel, devolvendo
    (resultado, mecanico). Tenta, em ordem: (1) aparar clausulas acessorias
    (`_aparar_para_caber`); (2) terminar numa fronteira de FRASE <= limite; (3)
    so em ultimo caso, corte por PALAVRA INTEIRA (`_cortar`, mecanico=True, SEM
    '...'). Nenhuma das opcoes deixa reticencias."""
    if contar(texto) <= limite:
        return texto, False
    piso = max(1, limite // 2)                    # evita devolver fragmento minusculo
    ap = _aparar_para_caber(texto, limite, faixa)
    if ap is not None and piso <= contar(ap) <= limite:
        return ap, False
    melhor = ""                                   # maior prefixo terminando em frase
    for m in re.finditer(r"[.!?…]+(?=\s|$)", texto):
        if m.end() <= limite:
            melhor = texto[:m.end()]
        else:
            break
    if len(melhor) >= piso:
        return melhor, False
    return _cortar(texto, limite), True


def resumir(
    texto: str,
    limite: int,
    *,
    tolerancia_1: int = TOLERANCIA_RODADA_1,
    tolerancia_2: int = TOLERANCIA_RODADA_2,
    max_tentativas: int = MAX_TENTATIVAS_PADRAO,
    chamar_llm: Callable[[str, str], str],
    deve_cancelar: Optional[Callable[[], bool]] = None,
    relatar: Optional[Callable[[int, int], None]] = None,
) -> Resultado:
    """Resume `texto` para caber em `limite` caracteres usando tolerancia
    progressiva (rodada 1 -> tolerancia_1; rodada 2 -> tolerancia_2).

    `chamar_llm(sistema, usuario)` e injetada para permitir mock em testes.
    `deve_cancelar()` (opcional) e consultada entre tentativas; quando True, o
    loop aborta cedo e devolve o melhor resumo parcial que ja cabe no limite.
    `relatar(tentativa, caracteres)` (opcional) e chamada a cada tentativa, para
    a interface mostrar o progresso ao vivo.
    """
    # --- Validacao do contrato publico (defesa em profundidade) ---
    if limite < 1:
        raise ValueError("O limite deve ser um inteiro positivo.")
    if tolerancia_1 < 0 or tolerancia_2 < 0:
        raise ValueError("As tolerancias nao podem ser negativas.")
    if max_tentativas < 1:
        raise ValueError("O numero de tentativas deve ser >= 1.")

    def cancelado() -> bool:
        return deve_cancelar is not None and deve_cancelar()

    # O texto original ja cabe no limite? Entao nao ha o que resumir: devolve
    # como esta, sem chamar a LLM. (Nao faz sentido pedir a LLM para EXPANDIR
    # o original que ja servia -- isso so arriscaria inventar conteudo.)
    if contar(texto) <= limite:
        n = contar(texto)
        return Resultado(texto, n, limite, 0, True, False, False, [n])

    historico: List[int] = []
    melhor: Optional[str] = None  # melhor candidato que CABE (mais perto por baixo)
    melhor_acima: Optional[str] = None  # menor candidato ACIMA do limite (p/ aparar)

    def anota(caracteres: int) -> None:
        """Registra uma tentativa no historico e a reporta ao vivo (se houver
        callback `relatar`), para a interface mostrar o progresso."""
        historico.append(caracteres)
        if relatar is not None:
            relatar(len(historico), caracteres)

    def registra_melhor(cand: str) -> None:
        nonlocal melhor, melhor_acima
        if not cand:
            return
        c = contar(cand)
        if c <= limite:                              # cabe: guarda o mais longo
            if melhor is None or c > contar(melhor):
                melhor = cand
        elif melhor_acima is None or c < contar(melhor_acima):
            melhor_acima = cand                      # acima: guarda o MENOR (mais perto)

    def validar(cand: object) -> str:
        """A LLM precisa devolver texto nao vazio. '' ou None sao erro de
        contrato -- nunca devem virar um resumo 'de sucesso'."""
        if not isinstance(cand, str) or not cand.strip():
            raise RespostaVaziaError(
                "A LLM retornou um texto vazio; nao foi possivel resumir."
            )
        return cand

    def finalizar(atual: Optional[str], cancelado_flag: bool) -> Resultado:
        # Escolhe, entre os candidatos que CABEM, o mais proximo do limite
        # (melhor qualidade). Se nada coube e nao foi cancelado, corte limpo.
        candidatos = [c for c in (atual, melhor) if c and contar(c) <= limite]
        # O loop gerativo as vezes DERRAPA bem abaixo do limite (encurta demais) e
        # nao recupera (expandir nao cresce). Entao APROVEITA o melhor candidato
        # que ficou ACIMA do limite, aparado de forma LIMPA (clausula/fronteira de
        # frase, sem '...'): costuma encostar no limite, bem melhor que o de baixo.
        if not cancelado_flag and melhor_acima is not None:
            aparado, mec = _encolher_limpo(
                melhor_acima, limite, max(tolerancia_1, tolerancia_2))
            if not mec and aparado and contar(aparado) <= limite:
                candidatos.append(aparado)
        cortado = False
        if candidatos:
            final = max(candidatos, key=contar)
        elif cancelado_flag:
            final = ""  # cancelou antes de obter qualquer parcial valido
        else:
            # nada coube: reduz da forma mais LIMPA possivel (apara clausulas ou
            # termina numa frase); so corta por palavra (SEM '...') em ultimo caso.
            final, cortado = _encolher_limpo(
                atual or "", limite, max(tolerancia_1, tolerancia_2))
        if final:  # remove numeros/idades que o gerativo livre tenha INVENTADO
            final = _remover_numeros_inventados(final, texto)  # (no fiel: no-op)
        if contar(final) > limite:  # ultima garantia da invariante
            final = _cortar(final, limite)
            cortado = True
        nf = contar(final)
        # "sucesso" = ha um resumo nao vazio DENTRO do limite, sem corte
        # mecanico de seguranca e sem cancelamento. Atencao: o resultado pode
        # ficar ABAIXO da faixa [limite - tolerancia, limite] e ainda assim ser
        # sucesso -- caber no limite e o que importa; a faixa e so um alvo de
        # qualidade que o loop tenta atingir.
        sucesso = (0 < nf <= limite) and not cortado and not cancelado_flag
        return Resultado(final, nf, limite, len(historico), sucesso, cortado,
                         cancelado_flag, historico)

    if cancelado():
        return finalizar("", True)

    # ===== ROTEAMENTO POR REGIME ===========================================
    # Estrategia PRIMARIA: FRASE-A-FRASE (NUNCA mistura frases). O Python fatia
    # em frases; UMA chamada ao LLM encurta CADA frase isolada (mantendo
    # nomes/numeros, sem fundir); o Python monta na ORDEM e, se ainda passar,
    # ELIMINA frases INTEIRAS do miolo (preservando 1a e ultima) ate caber, o
    # mais perto possivel do limite. Vale ate cortes de 50% (ver
    # LIMIAR_FRASE_A_FRASE): e fiel e encosta no limite. Cortes DRASTICOS (r > 50%),
    # ou quando nem abertura+fecho cabem (montagem devolve None), seguem pelo fluxo
    # GERATIVO (resumo livre, corrido, que parafraseia/mistura) abaixo.
    atual: Optional[str] = None
    remover = contar(texto) - limite          # > 0 aqui (ja passou o atalho)
    if remover > 0 and remover / contar(texto) <= LIMIAR_FRASE_A_FRASE:
        frases, quebras = _fatiar(texto)          # `quebras`: preserva paragrafos
        if len(frases) >= 2:
            curtas = _encurtar_frases(frases, chamar_llm, limite)  # 1 chamada (ou fallback)
            montado = _montar_frases(frases, curtas, limite, quebras)
            if montado is not None and 0 < contar(montado) <= limite:
                montado = _corrigir_maiusculas(montado)     # inicial de frase
                anota(contar(montado))
                registra_melhor(montado)
                # Resultado FIEL (sem mistura) -> entrega sempre. Mesmo que fique
                # abaixo do limite (texto de poucas frases grandes), NAO recorremos
                # ao gerativo: por opcao do usuario, fiel-porem-curto > misturar.
                return finalizar(montado, cancelado())
        # 1 frase so, ou nem abertura+fecho cabem: cai no fluxo gerativo abaixo

    # Resumo inicial gerativo (corte GRANDE). Conta como tentativa. Mira perto do
    # topo da faixa da rodada 1 para nascer proximo do limite (e nao curto demais).
    if atual is None:
        if cancelado():                         # cancelou apos a deleção falhar
            return finalizar("", True)
        if len(historico) >= max_tentativas:    # orcamento gasto na deleção
            return finalizar(texto, False)       # corte de seguranca do original
        atual = validar(chamar_llm(
            _SISTEMA, _prompt_inicial(texto, limite, max(0, limite - tolerancia_1))))
        anota(contar(atual))
        registra_melhor(atual)
    n = contar(atual)

    # Distribui as tentativas RESTANTES (descontando as ja gastas) entre as duas
    # rodadas. Se nao sobrou orcamento (ex.: max_tentativas == 1), encerra sem
    # disparar nenhuma chamada extra (o usuario nao deve pagar a mais).
    restantes = max(0, max_tentativas - len(historico))
    if restantes == 0:
        return finalizar(atual, False)
    tent_r1 = max(1, restantes // 2)
    rodadas = [(tolerancia_1, tent_r1), (tolerancia_2, restantes - tent_r1)]

    for tolerancia, n_tent in rodadas:
        minimo = max(0, limite - tolerancia)
        for _ in range(n_tent):
            if minimo <= n <= limite:
                break  # caiu na faixa vigente
            if cancelado():
                return finalizar(atual, True)
            if n > limite:
                atual = validar(
                    chamar_llm(_SISTEMA, _prompt_encurtar(atual, n, limite, minimo))
                )
            else:  # n < minimo -> curto demais (expande contra o ORIGINAL)
                atual = validar(
                    chamar_llm(_SISTEMA,
                               _prompt_expandir(texto, atual, n, limite, minimo))
                )
            n = contar(atual)
            anota(n)
            registra_melhor(atual)
            # PARADA ANTECIPADA: se o modelo estacionou (3 tamanhos iguais
            # seguidos) ainda ACIMA do limite, insistir nao adianta -- o limite
            # esta abaixo do "piso" do modelo. Encerra e deixa o finalizar aparar
            # de forma limpa, sem gastar as tentativas restantes.
            if (n > limite and len(historico) >= 3
                    and historico[-1] == historico[-2] == historico[-3]):
                return finalizar(atual, False)
        if minimo <= n <= limite:
            break  # convergiu; nao precisa da proxima rodada

    return finalizar(atual, False)


def _extrair_texto_ollama(corpo: dict) -> str:
    """Extrai e valida o texto da resposta do Ollama: remove blocos de
    'pensamento' (<think>...</think>) de modelos como o Qwen3, detecta
    truncamento por limite de tokens e rejeita resposta vazia."""
    if corpo.get("done_reason") == "length":
        raise ErroResumo(
            "A resposta foi truncada por atingir o limite de tokens. "
            "Tente um limite de caracteres menor."
        )
    texto = (corpo.get("message") or {}).get("content") or ""
    texto = re.sub(r"<think>.*?</think>", "", texto, flags=re.DOTALL).strip()
    if not texto:
        raise RespostaVaziaError(
            "O Ollama nao retornou texto. O modelo esta baixado? "
            "Rode: ollama pull <modelo>."
        )
    return texto


def _ler_corpo_erro(e: urllib.error.HTTPError) -> str:
    try:
        return e.read().decode("utf-8", "ignore")
    except Exception:
        return ""


def _chamar_ollama(modelo: str, sistema: str, usuario: str, max_tokens: int,
                   *, url: str = OLLAMA_URL, timeout: float = TIMEOUT_API) -> str:
    """Faz UMA chamada ao Ollama LOCAL (sem chave de API), pela API nativa
    /api/chat via stdlib (urllib) -- sem dependencias externas.

    Desliga o 'thinking' (think=False): modelos como o Qwen3 gastariam todo o
    orcamento de tokens 'pensando' e a resposta sairia truncada/lenta. Se o
    modelo nao suportar o parametro 'think', refaz a chamada sem ele."""
    base = {
        "model": modelo,
        "stream": False,
        "messages": [{"role": "system", "content": sistema},
                     {"role": "user", "content": usuario}],
        "options": {"temperature": TEMPERATURA, "num_predict": max_tokens},
    }
    payload = {**base, "think": False}
    corpo = None
    for tentativa in (1, 2):
        req = urllib.request.Request(
            url, data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"}, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                corpo = json.loads(resp.read().decode("utf-8"))
            break
        except urllib.error.HTTPError as e:
            detalhe = _ler_corpo_erro(e)
            if tentativa == 1 and "think" in detalhe.lower():
                payload = base   # modelo nao suporta 'think' -> tenta sem ele
                continue
            if e.code == 404 or "not found" in detalhe.lower():
                raise ErroOllamaIndisponivel(
                    f"Modelo '{modelo}' nao encontrado no Ollama. "
                    f"Baixe com: ollama pull {modelo}")
            raise ErroResumo(
                f"Erro do Ollama (HTTP {e.code}). {detalhe[:200]}".strip())
        except socket.timeout:
            # transitorio: o modelo demorou demais (o 'thinking' do Qwen3 varia).
            raise ErroResumo(
                f"O Ollama demorou mais de {int(timeout)}s para responder.")
        except urllib.error.URLError as e:
            motivo = getattr(e, "reason", e)
            if isinstance(motivo, (TimeoutError, socket.timeout)):  # timeout transitorio
                raise ErroResumo(
                    f"O Ollama demorou mais de {int(timeout)}s para responder.")
            raise ErroOllamaIndisponivel(
                "Nao consegui falar com o Ollama em localhost:11434. O Ollama "
                "esta instalado e rodando? (instale em ollama.com e rode "
                f"'ollama serve'). Detalhe: {motivo}")
    if corpo is None:
        raise ErroResumo("Falha ao chamar o Ollama.")
    return _extrair_texto_ollama(corpo)


def traduzir_excecao(e: Exception) -> str:
    """Converte excecoes em mensagens claras em PT. As mensagens do Ollama ja
    vem prontas dentro de ErroResumo; aqui apenas as repassamos e cobrimos os
    casos genericos (ValueError de validacao e erros inesperados)."""
    if isinstance(e, (ErroResumo, ValueError)):
        return str(e)
    return f"Erro inesperado: {e}"


# ===========================================================================
# FONTE EMBUTIDA (Garoa Light) -- binario zlib+base64
# ===========================================================================
# Preenchido na build a partir de Garoa-Light.otf (ver docstring do modulo).
_GAROA_OTF_B64 = """
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
Bx95nHZz9n1BzibN59+KLFv/Ce1WttM=
"""

# Familias possiveis pelas quais o Tk pode reconhecer a fonte (varia por SO).
_FONTE_FAMILIAS_PREFERIDAS = ("Garoa Light", "Garoa")


def _bytes_fonte() -> bytes:
    """Descomprime o binario da fonte embutida."""
    return zlib.decompress(base64.b64decode(_GAROA_OTF_B64))


def registrar_fonte_embutida() -> bool:
    """Registra a fonte embutida para o PROCESSO atual (sem instalar no
    sistema). Implementado para macOS via CoreText, direto da memoria.
    Retorna True se registrou; em outro SO ou falha, retorna False (o app
    usa a fonte padrao)."""
    if sys.platform != "darwin":
        return False
    try:
        import ctypes
        from ctypes import byref, c_char_p, c_long, c_void_p, util

        raw = _bytes_fonte()
        cf = ctypes.cdll.LoadLibrary(util.find_library("CoreFoundation"))
        cg = ctypes.cdll.LoadLibrary(util.find_library("CoreGraphics"))
        ct = ctypes.cdll.LoadLibrary(util.find_library("CoreText"))

        cf.CFDataCreate.restype = c_void_p
        cf.CFDataCreate.argtypes = [c_void_p, c_char_p, c_long]
        cg.CGDataProviderCreateWithCFData.restype = c_void_p
        cg.CGDataProviderCreateWithCFData.argtypes = [c_void_p]
        cg.CGFontCreateWithDataProvider.restype = c_void_p
        cg.CGFontCreateWithDataProvider.argtypes = [c_void_p]
        ct.CTFontManagerRegisterGraphicsFont.restype = ctypes.c_bool
        ct.CTFontManagerRegisterGraphicsFont.argtypes = [c_void_p, c_void_p]

        data = cf.CFDataCreate(None, raw, len(raw))
        provider = cg.CGDataProviderCreateWithCFData(data)
        if not data or not provider:  # falha cedo, motivo claro
            return False
        cgfont = cg.CGFontCreateWithDataProvider(provider)
        if not cgfont:
            return False
        err = c_void_p(0)
        # A registracao retem a fonte; nao liberamos as refs (a fonte vive
        # enquanto o processo existir -- vazamento de ~80KB, irrelevante).
        return bool(ct.CTFontManagerRegisterGraphicsFont(cgfont, byref(err)))
    except Exception:
        return False


def familia_da_fonte(root: tk.Misc) -> Optional[str]:
    """Retorna o nome de familia que o Tk reconhece para a fonte embutida,
    ou None se indisponivel."""
    disponiveis = {f.lower(): f for f in tkfont.families(root)}
    for nome in _FONTE_FAMILIAS_PREFERIDAS:
        if nome.lower() in disponiveis:
            return disponiveis[nome.lower()]
    return None


# ===========================================================================
# INTERFACE (Tkinter)
# ===========================================================================
class _ErroValidacao(Exception):
    """Mensagem de validacao da UI (mostrada como aviso)."""


# Paleta inspirada no app OmniFoto (tema escuro, texto branco, Garoa Light).
# Tema CLARO: fundo branco, letras pretas, cinza nos demais elementos.
COR_FUNDO = "#ffffff"        # fundo da janela
COR_SUPERFICIE = "#f4f4f4"   # campos (cinza bem claro)
COR_SUPERFICIE2 = "#e8e8e8"  # hover
COR_TEXTO = "#1a1a1a"        # texto principal (preto)
COR_TEXTO_DIM = "#777777"    # texto secundario / dicas / contadores (cinza)
COR_BORDA = "#cccccc"        # bordas sutis (cinza)
COR_BORDA_FOCO = "#888888"   # borda em foco
COR_NORMAL = COR_TEXTO       # contagem normal (preto)
COR_ALERTA = COR_TEXTO       # "acima do limite": SEM vermelho -- preto (pedido do usuario)
COR_OK = COR_TEXTO           # feedback "Copiado!": SEM verde -- preto (pedido do usuario)
COR_BTN_OFF_BG = "#f0f0f0"   # botao desabilitado (fundo)
COR_BTN_OFF_FG = "#bbbbbb"   # botao desabilitado (texto)

# Logo (wordmark) embutido como GIF base64 -- branco sobre o fundo escuro.
# Gerado a partir de CortaTexto_logo.png (invertido p/ branco, 133x64).
_LOGO_GIF_B64 = """
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
"""


# ===========================================================================
# AUTOINSTALADOR (1o uso): baixa o Ollama + o modelo qwen3 SEM o usuario abrir o
# Terminal. Tudo por caminho absoluto / urllib / subprocess (so stdlib).
# ===========================================================================
def _ollama_binario() -> Optional[str]:
    """Caminho do binario `ollama`: dentro de uma Ollama.app instalada
    (~/Applications ou /Applications) ou no PATH. None se nao achar."""
    for d in OLLAMA_APP_DIRS:
        b = os.path.join(os.path.expanduser(d), "Contents", "Resources", "ollama")
        if os.path.exists(b):
            return b
    # apps de GUI (abertos pelo Finder) NAO herdam o PATH do shell -> checa os
    # caminhos comuns do Homebrew/manual explicitamente, alem do `which`.
    for p in ("/opt/homebrew/bin/ollama", "/usr/local/bin/ollama"):
        if os.path.exists(p):
            return p
    return shutil.which("ollama")


def _ollama_no_ar(timeout: float = 2.0) -> bool:
    """True se o servidor do Ollama responde em 127.0.0.1:11434."""
    try:
        urllib.request.urlopen(OLLAMA_BASE + "/api/version", timeout=timeout)
        return True
    except Exception:
        return False


def _modelo_instalado(modelo: str = MODELO_OLLAMA, timeout: float = 4.0) -> bool:
    """True se o `modelo` (ex.: qwen3) ja foi baixado (consulta /api/tags)."""
    try:
        r = urllib.request.urlopen(OLLAMA_BASE + "/api/tags", timeout=timeout)
        nomes = [m.get("name", "") for m in json.loads(r.read()).get("models", [])]
        return any(n == modelo or n.startswith(modelo + ":") for n in nomes)
    except Exception:
        return False


def _espaco_livre_ok(minimo: int = ESPACO_MINIMO_BYTES) -> bool:
    try:
        return shutil.disk_usage(os.path.expanduser("~")).free >= minimo
    except Exception:
        return True                                  # na duvida, nao bloqueia


def _instalar_ollama(progresso: Callable[[str, float], None]) -> str:
    """Baixa o Ollama (zip) e extrai em ~/Applications (sem senha de admin).
    Devolve o caminho do binario. Se ja instalado, so devolve o caminho."""
    ja = _ollama_binario()
    if ja:
        return ja
    base = os.path.expanduser("~/Applications")
    os.makedirs(base, exist_ok=True)
    zip_tmp = os.path.join("/tmp", "Ollama-darwin.zip")
    req = urllib.request.Request(OLLAMA_DOWNLOAD_URL,
                                 headers={"User-Agent": "CortaTexto"})
    with urllib.request.urlopen(req, timeout=TIMEOUT_DOWNLOAD) as resp:
        total = int(resp.headers.get("Content-Length") or 0)
        baixado = 0
        with open(zip_tmp, "wb") as f:
            while True:
                pedaco = resp.read(1 << 16)
                if not pedaco:
                    break
                f.write(pedaco)
                baixado += len(pedaco)
                progresso("Baixando o Ollama (a IA local)",
                          (baixado / total) if total else 0.0)
    subprocess.run(["ditto", "-x", "-k", zip_tmp, base], check=True)   # preserva o bundle
    try:
        os.remove(zip_tmp)
    except OSError:
        pass
    b = _ollama_binario()
    if not b:
        raise ErroResumo("Nao consegui instalar o Ollama (extracao falhou).")
    return b


def _iniciar_ollama(binario: Optional[str] = None) -> None:
    """Inicia o servidor: abre a Ollama.app (sobe a API sozinha) ou, em ultimo
    caso, roda o binario com `serve` em segundo plano."""
    for d in OLLAMA_APP_DIRS:
        app = os.path.expanduser(d)
        if os.path.exists(app):
            subprocess.run(["open", app], check=False)
            return
    if binario:
        subprocess.Popen([binario, "serve"], stdout=subprocess.DEVNULL,
                         stderr=subprocess.DEVNULL)


def _esperar_ollama(timeout: int = 60) -> bool:
    for _ in range(max(1, timeout)):
        if _ollama_no_ar():
            return True
        time.sleep(1)
    return False


def _baixar_modelo(binario: str, modelo: str,
                   progresso: Callable[[str, float], None]) -> None:
    """Roda `ollama pull <modelo>` e reporta o progresso (le os '%' da saida)."""
    proc = subprocess.Popen([binario, "pull", modelo], stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True, bufsize=1)
    for linha in proc.stdout:
        m = re.search(r"(\d{1,3})%", linha)
        if m:
            progresso("Baixando o modelo qwen3 (~5,2 GB)",
                      min(100, int(m.group(1))) / 100.0)
    proc.wait()
    if proc.returncode != 0:
        raise ErroResumo("Nao consegui baixar o modelo qwen3. Verifique a internet.")


class App(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("CortaTexto")
        self.minsize(640, 560)
        self.config(bg=COR_FUNDO)

        # Logo (carregado uma vez; guardamos a referencia p/ nao ser coletado).
        self._logo_img = self._carregar_logo()

        self._fila: "queue.Queue[tuple]" = queue.Queue()
        self._processando = False
        self._cancelar_evt = threading.Event()
        self._after_id: Optional[str] = None
        self._copia_after: Optional[str] = None
        # Identifica o job atual. Cancelar (ou iniciar outro) incrementa o id, o
        # que invalida o resultado de qualquer thread anterior ainda em voo: a
        # fila so aceita resultados do job vigente -> cancelamento e imediato e
        # nao exibe nada do trabalho cancelado.
        self._job_id = 0

        self._aplicar_fonte()
        self._construir()
        self._ajustar_janela()

        self.protocol("WM_DELETE_WINDOW", self._ao_fechar)
        # Enter aciona Resumir -- EXCETO quando o foco esta numa caixa de texto
        # multilinha (entrada/saida editavel), onde Enter insere quebra de linha.
        # Ctrl/Command+Enter aciona Resumir sempre (inclusive dentro das caixas).
        self.bind("<Return>", self._on_return)
        self.bind("<KP_Enter>", self._on_return)
        self.bind("<Control-Return>", lambda e: self._on_resumir())
        self.bind("<Command-Return>", lambda e: self._on_resumir())
        self.bind("<Escape>", lambda e: self._cancelar())
        self.txt_entrada.focus_set()

        # Garante que a janela apareca NA FRENTE ao abrir (apps Tk empacotados
        # no macOS as vezes sobem atras das outras janelas). Pinamos no topo
        # por um instante, damos foco e soltamos o topo logo em seguida.
        self.lift()
        self.attributes("-topmost", True)
        self.after(600, lambda: self.attributes("-topmost", False))
        self.focus_force()

        if not self._fonte_ok:
            self.lbl_status.config(
                text="Fonte Garoa Light indisponivel - usando fonte padrao.")

        # Workaround do bug de redraw do Tk no macOS recente: a janela abre em
        # branco ate ser redimensionada. Forcamos um micro-resize (1px e volta)
        # logo apos a janela ser mapeada, o que dispara o desenho de tudo.
        self.after(80, self._forcar_redraw)

    def _forcar_redraw(self, _tentativas: int = 0) -> None:
        try:
            if not self.winfo_exists():
                return
            self.update_idletasks()
            w, h = self.winfo_width(), self.winfo_height()
            if w <= 1 and _tentativas < 20:  # ainda nao mapeada; tenta de novo
                self.after(50, lambda: self._forcar_redraw(_tentativas + 1))
                return
            self.geometry(f"{w + 1}x{h + 1}")
            self.update_idletasks()
            self.geometry(f"{w}x{h}")
        except tk.TclError:
            pass

    def _ajustar_janela(self) -> None:
        """Abre a janela mostrando TODA a interface: dimensiona pelo tamanho
        requisitado pelos widgets (sem cortar nada) e centraliza, sem exceder
        a tela."""
        self.update_idletasks()
        w = int(self.winfo_reqwidth() * 1.3)   # 30% mais largo que o necessario
        h = self.winfo_reqheight()             # altura inalterada (como ja estava)
        sw, sh = self.winfo_screenwidth(), self.winfo_screenheight()
        w = min(w, sw - 80)
        h = min(h, sh - 120)
        x = max(0, (sw - w) // 2)
        y = max(24, (sh - h) // 3)
        self.geometry(f"{w}x{h}+{x}+{y}")

    # ---- logo e botoes ----------------------------------------------------
    def _carregar_logo(self) -> Optional[tk.PhotoImage]:
        """Carrega o logo embutido (GIF base64). Retorna None se falhar."""
        try:
            return tk.PhotoImage(data="".join(_LOGO_GIF_B64.split()))
        except Exception:
            return None

    def _criar_botao(self, parent: tk.Misc, texto: str,
                     comando: Callable[[], None], primario: bool = False):
        """Botao no estilo do OmniFoto: PILULA (cantos totalmente arredondados,
        como border-radius: 999px), borda fina, texto preto e hover que escurece
        levemente. Desenhado num Canvas porque o Tkinter nao tem cantos
        arredondados nativos. Tema claro: borda/fundo em cinza."""
        if primario:  # botao principal (Resumir): um pouco mais marcado
            cores = {"borda": "#999999", "fill": "#f0f0f0",
                     "borda_h": "#555555", "fill_h": "#e2e2e2"}
        else:         # secundarios: fundo branco, borda cinza clara
            cores = {"borda": "#cccccc", "fill": COR_FUNDO,
                     "borda_h": "#888888", "fill_h": "#f0f0f0"}
        m = tkfont.Font(font=self._f_base)
        w = m.measure(texto) + 44
        h = m.metrics("linespace") + 22
        cv = tk.Canvas(parent, width=w, height=h, bg=COR_FUNDO,
                       highlightthickness=0, bd=0, cursor="hand2")
        cv._cores = cores           # type: ignore[attr-defined]
        cv._habilitado = True       # type: ignore[attr-defined]
        cv._comando = comando       # type: ignore[attr-defined]
        cv._texto = texto           # type: ignore[attr-defined]
        cv._wh = (w, h)             # type: ignore[attr-defined]
        self._desenhar_botao(cv, hover=False)
        cv.bind("<Enter>", lambda e, c=cv: self._desenhar_botao(c, hover=True)
                if c._habilitado else None)
        cv.bind("<Leave>", lambda e, c=cv: self._desenhar_botao(c, hover=False)
                if c._habilitado else None)
        cv.bind("<Button-1>", lambda e, c=cv: c._comando()
                if c._habilitado else None)
        return cv

    def _rrect_fill(self, cv: tk.Canvas, x1, y1, x2, y2, r, color):
        """Retangulo arredondado PREENCHIDO (2 retangulos + 4 ovais de canto,
        todos solidos e sobrepostos) -- sem arcos/linhas, entao nao ha juncoes
        e nao aparecem falhas de pixel nos cantos."""
        r = max(0, min(r, (x2 - x1) / 2, (y2 - y1) / 2))
        d = 2 * r
        cv.create_rectangle(x1 + r, y1, x2 - r, y2, fill=color, outline=color)
        cv.create_rectangle(x1, y1 + r, x2, y2 - r, fill=color, outline=color)
        cv.create_oval(x1, y1, x1 + d, y1 + d, fill=color, outline=color)
        cv.create_oval(x2 - d, y1, x2, y1 + d, fill=color, outline=color)
        cv.create_oval(x1, y2 - d, x1 + d, y2, fill=color, outline=color)
        cv.create_oval(x2 - d, y2 - d, x2, y2, fill=color, outline=color)

    def _rounded(self, cv: tk.Canvas, x1, y1, x2, y2, r, fill, outline, width):
        """Forma arredondada com borda LIMPA (sem falhas): desenha a forma cheia
        na cor da borda e, por cima, a mesma forma recuada `width` px na cor de
        preenchimento -- a borda vira um anel continuo. r = altura/2 produz uma
        pilula (cantos totalmente redondos, como o border-radius:999px)."""
        self._rrect_fill(cv, x1, y1, x2, y2, r, outline)
        self._rrect_fill(cv, x1 + width, y1 + width, x2 - width, y2 - width,
                         max(0, r - width), fill)

    def _desenhar_botao(self, cv: tk.Canvas, hover: bool) -> None:
        cv.delete("all")
        w, h = cv._wh
        c = cv._cores
        if not cv._habilitado:
            borda, fill, fg = "#dddddd", COR_BTN_OFF_BG, COR_BTN_OFF_FG
        else:
            borda = c["borda_h"] if hover else c["borda"]
            fill = c["fill_h"] if hover else c["fill"]
            fg = COR_TEXTO
        self._rounded(cv, 2, 2, w - 2, h - 2, r=(h - 4) / 2,
                      fill=fill, outline=borda, width=2)
        cv.create_text(w // 2, h // 2, text=cv._texto, fill=fg,
                       font=self._f_base)

    def _set_botao(self, cv: tk.Canvas, habilitado: bool) -> None:
        cv._habilitado = habilitado  # type: ignore[attr-defined]
        cv.config(cursor="hand2" if habilitado else "arrow")
        self._desenhar_botao(cv, hover=False)

    def _campo_pilula(self, parent: tk.Misc, textvariable: tk.StringVar,
                      chars: int):
        """Campo de entrada do TAMANHO de um campo normal, so com os cantos
        arredondados (pilula). Tamanho FIXO calculado pela fonte e pelo numero
        de caracteres (nao expande): um Entry sem borda, centrado, sobre um
        Canvas que desenha a pilula branca de borda cinza. Retorna (card, entry)."""
        f = tkfont.Font(font=self._f_base)
        h = f.metrics("linespace") + 14                  # altura ~ 1 linha + folga
        # Largura = texto + folga das pontas arredondadas. Folga enxuta (h//2,
        # ~1 raio) para o campo ficar estreito; nunca menor que a altura, senao a
        # pilula vira circulo.
        w = max(h, f.measure("0") * chars + h // 2)
        card = tk.Frame(parent, bg=COR_FUNDO, width=w, height=h,
                        highlightthickness=0, bd=0)
        card.grid_propagate(False)                       # mantem o tamanho fixo
        cv = tk.Canvas(card, width=w, height=h, bg=COR_FUNDO,
                       highlightthickness=0, bd=0)
        cv.place(x=0, y=0)
        self._rounded(cv, 1, 1, w - 1, h - 1, r=(h - 2) / 2,
                      fill="#ffffff", outline=COR_BORDA, width=2)
        ent = tk.Entry(card, textvariable=textvariable, width=chars, bd=0,
                       relief="flat", highlightthickness=0, bg="#ffffff",
                       fg=COR_TEXTO, insertbackground=COR_TEXTO,
                       justify="center", font=self._f_base)
        ent.place(relx=0.5, rely=0.5, anchor="center")
        return card, ent

    # ---- caixa de texto branca com cantos arredondados --------------------
    def _caixa_texto(self, parent: tk.Misc, **text_kw):
        """Caixa de texto BRANCA com cantos arredondados e barra de scroll no
        visual da interface. Desenhamos um cartao branco arredondado num Canvas
        e embutimos um `tk.Text` por cima, recuado pelo raio. A barra de scroll
        e CUSTOM (Canvas): um trilho de LINHA FINA cinza (`COR_BORDA`, 2px) com
        um CIRCULO (anel) como alca, que reflete e controla a rolagem. Retorna
        (card, texto)."""
        r = 28          # raio dos cantos das caixas
        larg_sb = 22    # largura da faixa da barra de scroll (folga lateral p/ o anel)
        rc = 8          # raio do circulo (alca)
        mc = 2          # margem p/ o contorno do circulo nao ser cortado nas pontas
        card = tk.Frame(parent, bg=COR_FUNDO, highlightthickness=0, bd=0)
        card.grid_rowconfigure(0, weight=1)
        card.grid_columnconfigure(0, weight=1)
        cv = tk.Canvas(card, width=1, height=1, bg=COR_FUNDO,
                       highlightthickness=0, bd=0)
        cv.grid(row=0, column=0, sticky="nsew")
        txt = tk.Text(
            card, bg="#ffffff", fg=COR_TEXTO, insertbackground=COR_TEXTO,
            relief="flat", bd=0, highlightthickness=0, padx=10, pady=8,
            font=self._f_caixa, **text_kw)   # 10 pt: cabe mais linha no mesmo espaco
        # espaco extra a direita (r + larg_sb) reservado para a barra custom
        txt.grid(row=0, column=0, sticky="nsew", padx=(r, r + larg_sb), pady=r)

        # ---- barra de scroll custom: trilho (linha) + alca (circulo) ----
        sb = tk.Canvas(card, width=larg_sb, bg="#ffffff",
                       highlightthickness=0, bd=0)
        sb._frac = (0.0, 1.0)                       # (first, last) da view
        sb.place(relx=1.0, x=-r, y=r, anchor="ne", width=larg_sb,
                 relheight=1.0, height=-2 * r)

        def _desenhar_sb(evento=None):
            try:
                h = sb.winfo_height()
            except tk.TclError:
                return
            sb.delete("all")
            if h < 4:
                return
            cx = larg_sb // 2
            sb.create_line(cx, rc, cx, h - rc, fill=COR_BORDA, width=2)
            first, last = sb._frac
            size = last - first
            if size < 0.999:                        # ha conteudo a rolar
                topo = rc + mc                      # trajeto recuado das pontas
                usable = max(1, (h - rc - mc) - topo)
                denom = 1.0 - size
                p = (first / denom) if denom > 1e-6 else 0.0
                cy = topo + min(1.0, max(0.0, p)) * usable
                sb.create_oval(cx - rc, cy - rc, cx + rc, cy + rc,
                               outline=COR_BORDA, width=2, fill="#ffffff")

        def _sb_set(first, last):                   # chamado pelo yscrollcommand
            sb._frac = (float(first), float(last))
            _desenhar_sb()

        def _sb_arrasta(evento):                    # arrastar/clicar move a view
            h = sb.winfo_height()
            topo = rc + mc
            usable = max(1, (h - rc - mc) - topo)
            first, last = sb._frac
            size = last - first
            p = min(1.0, max(0.0, (evento.y - topo) / usable))
            txt.yview_moveto(p * (1.0 - size))

        txt.config(yscrollcommand=_sb_set)
        sb.bind("<Button-1>", _sb_arrasta)
        sb.bind("<B1-Motion>", _sb_arrasta)
        sb.bind("<Configure>", _desenhar_sb)

        def _redraw(evento=None):
            try:
                w, h = card.winfo_width(), card.winfo_height()
            except tk.TclError:
                return
            if w > 4 and h > 4:
                cv.delete("all")
                self._rounded(cv, 1, 1, w - 1, h - 1, r,
                              fill="#ffffff", outline=COR_BORDA, width=2)
        card.bind("<Configure>", _redraw)
        return card, txt

    # ---- fonte ------------------------------------------------------------
    def _aplicar_fonte(self) -> None:
        """Aplica a fonte Garoa Light a TODA a interface. Usamos widgets tk
        classicos (nao ttk): o tema ttk do macOS recente as vezes nao desenha
        os widgets dentro do app empacotado, enquanto os widgets tk classicos
        pintam de forma confiavel e honram a fonte via option_add."""
        fam = familia_da_fonte(self)
        self._fonte_ok = fam is not None
        if fam is None:  # fallback: fonte padrao do sistema
            fam = tkfont.nametofont("TkDefaultFont").actual("family")
        self._familia = fam

        base = 15  # escala maior, no espirito do OmniFoto (font-size 150%)
        # option_add aplica a fonte a todos os widgets tk criados a partir daqui.
        self.option_add("*Font", (fam, base))

        # Fontes nomeadas (ScrolledText, dialogos e widgets herdam destas).
        for nome in ("TkDefaultFont", "TkTextFont", "TkFixedFont", "TkMenuFont",
                     "TkHeadingFont", "TkCaptionFont", "TkSmallCaptionFont",
                     "TkIconFont", "TkTooltipFont"):
            try:
                tkfont.nametofont(nome).configure(family=fam, size=base)
            except tk.TclError:
                pass

        # Fontes derivadas (hierarquia por TAMANHO; sem negrito -- o arquivo
        # tem apenas o peso Light).
        # Tamanho UNICO para todos os textos da interface (sem hierarquia).
        self._f_titulo = (fam, base)
        self._f_base = (fam, base)
        self._f_pequena = (fam, base)
        self._f_caixa = (fam, 12)   # texto DENTRO das caixas (entrada/saida/manual)

    # ---- construcao da UI -------------------------------------------------
    def _construir(self) -> None:
        # Estilos reutilizaveis (tema claro: texto PRETO, cinza so em bordas/campos).
        est_lbl = {"bg": COR_FUNDO, "fg": COR_TEXTO}
        est_dim = {"bg": COR_FUNDO, "fg": COR_TEXTO, "font": self._f_pequena}
        est_ent = {"bg": COR_SUPERFICIE, "fg": COR_TEXTO,
                   "insertbackground": COR_TEXTO, "relief": "flat",
                   "highlightthickness": 1, "highlightbackground": COR_BORDA,
                   "highlightcolor": COR_BORDA_FOCO}

        # ---- topo: LOGO no canto esquerdo (o botao 'Como funciona' fica embaixo,
        # na linha do Resumir) ----
        topo = tk.Frame(self, bg=COR_FUNDO)
        topo.grid(row=0, column=0, columnspan=4, sticky="ew", padx=14, pady=(12, 2))
        if self._logo_img is not None:
            tk.Label(topo, image=self._logo_img, bg=COR_FUNDO).grid(
                row=0, column=0, sticky="w")
        else:  # fallback textual se o logo nao carregar
            tk.Label(topo, text="CortaTexto", font=(self._familia, 26),
                     **est_lbl).grid(row=0, column=0, sticky="w")

        # ---- cabecalho da entrada: rotulo + contador ----
        cab_ent = tk.Frame(self, bg=COR_FUNDO)
        cab_ent.grid(row=1, column=0, columnspan=4, sticky="ew",
                     padx=14, pady=(6, 0))
        cab_ent.grid_columnconfigure(0, weight=1)
        tk.Label(cab_ent, text="Texto a ser tesourado:", font=self._f_titulo,
                 **est_lbl).grid(row=0, column=0, sticky="w")
        self.lbl_entrada_contagem = tk.Label(
            cab_ent, text="Original: 0 caracteres", **est_dim)
        self.lbl_entrada_contagem.grid(row=0, column=1, sticky="e")

        # ---- entrada de texto (caixa branca arredondada) ----
        card_ent, self.txt_entrada = self._caixa_texto(self, height=20, wrap="word")
        card_ent.grid(row=2, column=0, columnspan=4, sticky="nsew",
                      padx=14, pady=4)
        self.txt_entrada.bind("<KeyRelease>", self._atualizar_contagem_entrada)
        # Aspas curvas: nunca deixar entrar/exibir U+0022 (copo na Garoa Light).
        self.txt_entrada.bind("<Key>", self._aspa_curva)
        self.txt_entrada.bind(
            "<<Paste>>",
            lambda e: self._colar_curvo(e, self._atualizar_contagem_entrada))

        # ---- parametros ----
        # So "Reduza para: [campo] caracteres" aparece. Tolerancia 1/2 e Max.
        # tentativas ficam OCULTOS: as StringVars existem (com os defaults) para
        # o resumir() le-las, mas nao ha widgets na interface.
        params = tk.Frame(self, bg=COR_FUNDO)
        params.grid(row=3, column=0, columnspan=4, sticky="ew", padx=12, pady=2)
        celp = {"pady": 5}
        tk.Label(params, text="Reduza para:", **est_lbl).grid(
            row=0, column=0, padx=(0, 6), **celp)
        self.var_limite = tk.StringVar(value="280")
        card_lim, self.ent_limite = self._campo_pilula(params, self.var_limite, 10)
        card_lim.grid(row=0, column=1, **celp)
        self.ent_limite.bind("<FocusOut>", self._formatar_limite)  # 1.000 ao sair
        self.ent_limite.bind("<KeyRelease>", self._formatar_limite_vivo)  # ao vivo
        tk.Label(params, text="caracteres", **est_lbl).grid(
            row=0, column=2, padx=(6, 0), **celp)
        self.var_limite.trace_add("write", lambda *a: self._recontar_saida())
        # OCULTOS (usam os defaults internamente; sem campos na interface):
        self.var_tol1 = tk.StringVar(value=str(TOLERANCIA_RODADA_1))
        self.var_tol2 = tk.StringVar(value=str(TOLERANCIA_RODADA_2))
        self.var_tentativas = tk.StringVar(value=str(MAX_TENTATIVAS_PADRAO))

        # (Sem campo de modelo na UI -- usa o MODELO_OLLAMA fixo, local via Ollama.)

        # ---- botoes + status ----
        botoes = tk.Frame(self, bg=COR_FUNDO)
        botoes.grid(row=6, column=0, columnspan=4, sticky="ew", padx=12, pady=6)
        self.btn_resumir = self._criar_botao(botoes, "Resumir",
                                             self._on_resumir, primario=True)
        self.btn_resumir.grid(row=0, column=0, padx=(0, 8))
        self.btn_cancelar = self._criar_botao(botoes, "Cancelar", self._cancelar)
        self.btn_cancelar.grid(row=0, column=1, padx=8)
        self.btn_copiar = self._criar_botao(botoes, "Copiar resultado",
                                            self._copiar)
        self.btn_copiar.grid(row=0, column=2, padx=8)
        self._set_botao(self.btn_cancelar, False)
        self._set_botao(self.btn_copiar, False)
        self.lbl_status = tk.Label(botoes, text="", **est_dim)
        self.lbl_status.grid(row=0, column=3, padx=10)
        self.lbl_copia = tk.Label(botoes, text="", bg=COR_FUNDO, fg=COR_OK,
                                  font=self._f_pequena)
        self.lbl_copia.grid(row=0, column=4, padx=6)
        # botao 'Como funciona' (ex-'Manual') empurrado para a DIREITA, alinhado
        # com o Resumir nesta mesma linha (espacador com peso entre eles).
        botoes.grid_columnconfigure(5, weight=1)
        self.btn_manual = self._criar_botao(botoes, "Como funciona",
                                            self._abrir_manual, primario=True)
        self.btn_manual.grid(row=0, column=6, sticky="e")

        # ---- resultado (editavel, com recontagem ao vivo) ----
        tk.Label(self, text="Veja como ficou e edite se precisar:",
                 font=self._f_titulo,
                 **est_lbl).grid(row=7, column=0, columnspan=4, sticky="w",
                                 padx=14, pady=(4, 0))
        card_sai, self.txt_saida = self._caixa_texto(self, height=20, wrap="word")
        card_sai.grid(row=8, column=0, columnspan=4, sticky="nsew",
                      padx=14, pady=4)
        self.txt_saida.bind("<KeyRelease>", self._recontar_saida)
        self.txt_saida.bind("<Key>", self._aspa_curva)
        self.txt_saida.bind(
            "<<Paste>>", lambda e: self._colar_curvo(e, self._recontar_saida))
        self.lbl_contagem = tk.Label(self, text="", bg=COR_FUNDO,
                                     fg=COR_NORMAL, font=self._f_pequena)
        self.lbl_contagem.grid(row=9, column=0, columnspan=4, sticky="w",
                              padx=14, pady=(0, 12))

        # Expansao: as duas caixas de texto dividem a altura disponivel POR
        # IGUAL (mesmo peso) -- na tela, ambas mostram o mesmo nº de linhas.
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(2, weight=1)
        self.grid_rowconfigure(8, weight=1)

        # No 1o uso: garante Ollama rodando + modelo baixado (autoinstalador).
        self._setup_aberto = False
        self.after(500, self._verificar_ambiente)

    # ---- autoinstalador (Ollama + qwen3) ---------------------------------
    def _verificar_ambiente(self) -> None:
        """Numa thread: se o Ollama esta instalado mas parado, sobe o servidor.
        Se faltar instalar o Ollama ou baixar o modelo, abre a tela de setup."""
        fila: queue.Queue = queue.Queue()

        def checar():
            if not _ollama_no_ar():                    # servidor parado -> tenta subir
                _iniciar_ollama(_ollama_binario())
                _esperar_ollama(15)
            # PRONTO = servidor no ar E modelo baixado. NAO exige achar o binario:
            # app de GUI nao herda o PATH, mas se a API responde e o modelo existe,
            # o app ja funciona (resumir fala direto com a porta 11434).
            fila.put(_ollama_no_ar() and _modelo_instalado())

        threading.Thread(target=checar, daemon=True).start()

        def aguardar():
            try:
                pronto = fila.get_nowait()
            except queue.Empty:
                self.after(200, aguardar)
                return
            if not pronto:
                self._abrir_setup()
        self.after(200, aguardar)

    def _abrir_setup(self) -> None:
        """Tela 'Preparar o CortaTexto': baixa Ollama + qwen3 com progresso, sem
        Terminal. So aparece quando falta instalar algo."""
        if self._setup_aberto:
            return
        self._setup_aberto = True
        top = tk.Toplevel(self)
        top.withdraw()                 # nao mapear ainda: evita o "pisca + pulo"
        top.title("Preparar o CortaTexto")
        top.configure(bg=COR_FUNDO)
        top.transient(self)
        est = {"bg": COR_FUNDO, "fg": COR_TEXTO}

        def fechar():
            self._setup_aberto = False
            top.destroy()

        tk.Label(top, text="Falta preparar a IA (uma vez só)", font=self._f_titulo,
                 **est).grid(row=0, column=0, sticky="w", padx=20, pady=(18, 6))
        msg = ("O CortaTexto usa uma IA que roda no SEU Mac. Na primeira vez é "
               "preciso baixar o Ollama (~178 MB) e o modelo qwen3 (~5,2 GB) — uma "
               "vez só; depois funciona offline. Pode levar alguns minutos.")
        tk.Label(top, text=msg, wraplength=470, justify="left", font=self._f_caixa,
                 **est).grid(row=1, column=0, sticky="w", padx=20, pady=(0, 10))
        self._setup_status = tk.Label(top, text="", wraplength=470, justify="left",
                                      font=self._f_caixa, fg=COR_TEXTO_DIM, bg=COR_FUNDO)
        self._setup_status.grid(row=2, column=0, sticky="w", padx=20, pady=(0, 14))
        linha = tk.Frame(top, bg=COR_FUNDO)
        linha.grid(row=3, column=0, sticky="w", padx=20, pady=(0, 18))
        btn = self._criar_botao(linha, "Instalar agora", lambda: iniciar(), primario=True)
        btn.grid(row=0, column=0, padx=(0, 8))
        self._criar_botao(linha, "Agora não", fechar).grid(row=0, column=1)
        top.grid_columnconfigure(0, weight=1)

        fila: queue.Queue = queue.Queue()

        def progresso(texto, frac):
            fila.put(f"{texto}… {int(frac * 100)}%")

        def trabalho():
            try:
                if not _espaco_livre_ok():
                    raise ErroResumo("Espaco insuficiente: sao necessarios ~8 GB "
                                     "livres no disco.")
                binario = _instalar_ollama(progresso)
                fila.put("Iniciando o Ollama…")
                _iniciar_ollama(binario)
                if not _esperar_ollama(60):
                    raise ErroResumo("O Ollama nao iniciou a tempo. Tente de novo.")
                if not _modelo_instalado():
                    fila.put("Baixando o modelo qwen3 (~5,2 GB). Isso demora…")
                    _baixar_modelo(binario, MODELO_OLLAMA, progresso)
                fila.put(("ok", None))
            except Exception as e:                     # noqa: BLE001
                fila.put(("erro", str(e)))

        def iniciar():
            self._set_botao(btn, False)
            self._setup_status.config(text="Comecando…")
            threading.Thread(target=trabalho, daemon=True).start()
            self.after(150, bombear)

        def bombear():
            try:
                while True:
                    item = fila.get_nowait()
                    if isinstance(item, tuple) and item[0] == "ok":
                        self._setup_status.config(text="Tudo pronto! Ja pode resumir.")
                        self._setup_aberto = False
                        top.after(1000, top.destroy)
                        return
                    if isinstance(item, tuple) and item[0] == "erro":
                        self._setup_status.config(text="Nao deu certo: " + item[1])
                        btn.config(text="Tentar de novo")
                        self._set_botao(btn, True)
                        return
                    self._setup_status.config(text=str(item))
            except queue.Empty:
                pass
            self.after(150, bombear)

        top.update_idletasks()
        w = max(520, top.winfo_reqwidth())
        x = max(0, (self.winfo_screenwidth() - w) // 2)
        top.geometry(f"{w}x{top.winfo_reqheight()}+{x}+{max(40, self.winfo_screenheight() // 4)}")
        top.update_idletasks()         # forca o WM a aplicar a geometria antes (macOS)
        top.protocol("WM_DELETE_WINDOW", fechar)
        top.deiconify()                # so agora exibe, ja na posicao final
        top.lift()
        top.focus_force()

    # ---- helpers de UI ----------------------------------------------------
    def _limite_atual(self) -> Optional[int]:
        try:
            v = int(self.var_limite.get().replace(".", "").strip())  # tolera milhar
            return v if v > 0 else None
        except (ValueError, tk.TclError):
            return None

    def _formatar_limite(self, evento=None) -> None:
        """Ao sair do campo, reescreve o limite com ponto de milhar (1.000)."""
        v = self._limite_atual()
        if v is not None:
            self.var_limite.set(_milhar(v))

    def _formatar_limite_vivo(self, evento=None) -> None:
        """Mostra o ponto de milhar JA enquanto o usuario digita (1.770), sem
        esperar sair do campo. Preserva a posicao do cursor contando os digitos a
        sua esquerda. Campo vazio fica vazio (permite apagar e redigitar)."""
        ent = self.ent_limite
        texto = self.var_limite.get()
        digitos = re.sub(r"\D", "", texto)
        if not digitos:
            if texto:
                self.var_limite.set("")
            return
        formatado = _milhar(int(digitos))
        if formatado == texto:
            return                                   # nada mudou: nao mexe no cursor
        try:
            pos = ent.index("insert")
        except tk.TclError:
            pos = len(texto)
        dig_esq = len(re.sub(r"\D", "", texto[:pos]))
        self.var_limite.set(formatado)
        idx = vistos = 0                             # recoloca o cursor apos os
        while idx < len(formatado) and vistos < dig_esq:  # mesmos N digitos
            if formatado[idx].isdigit():
                vistos += 1
            idx += 1
        ent.icursor(idx)

    def _atualizar_contagem_entrada(self, evento=None) -> None:
        if not self.winfo_exists():  # callback agendado pode rodar pos-fechamento
            return
        n = contar(self.txt_entrada.get("1.0", "end-1c"))
        lim = self._limite_atual()
        extra = "  (ja cabe no limite)" if lim is not None and n <= lim else ""
        self.lbl_entrada_contagem.config(
            text=f"Original: {_milhar(n)} caracteres{extra}")

    def _recontar_saida(self, evento=None) -> None:
        if not hasattr(self, "txt_saida"):
            return
        if self._processando:
            return  # durante o processamento o rotulo mostra o progresso ao vivo
        n = contar(self.txt_saida.get("1.0", "end-1c"))
        lim = self._limite_atual()
        if lim is None:
            self.lbl_contagem.config(text=f"{_milhar(n)} caracteres",
                                     foreground=COR_NORMAL)
            return
        # sem aviso de "acima do limite" (pedido do usuario): so o contador.
        self.lbl_contagem.config(text=f"{_milhar(n)} / {_milhar(lim)} caracteres",
                                 foreground=COR_NORMAL)

    # ---- acao do botao ----------------------------------------------------
    def _on_return(self, event=None):
        """Enter aciona Resumir, MENOS quando o foco esta numa caixa de texto
        multilinha (entrada ou saida editavel) -- ali Enter insere quebra de
        linha normalmente. Use Ctrl/Command+Enter para acionar de dentro delas."""
        if self.focus_get() in (getattr(self, "txt_entrada", None),
                                 getattr(self, "txt_saida", None)):
            return  # deixa o widget inserir a quebra de linha
        self._on_resumir()
        return "break"

    def _on_resumir(self) -> None:
        if self._processando:
            return  # ja ha um job em andamento (barra reentrancia)

        texto = self.txt_entrada.get("1.0", "end-1c")
        try:
            if not texto.strip():
                raise _ErroValidacao("Cole um texto para resumir.")
            limite = self._ler_int(self.var_limite, "Limite", minimo=1)
            tol1 = self._ler_int(self.var_tol1, "Tolerancia 1", minimo=0)
            tol2 = self._ler_int(self.var_tol2, "Tolerancia 2", minimo=0)
            tentativas = self._ler_int(self.var_tentativas, "Max. tentativas",
                                       minimo=1)
        except _ErroValidacao as e:
            messagebox.showwarning("Atencao", str(e))
            return

        # bloqueia UI e dispara thread
        self._processando = True
        self._cancelar_evt.clear()
        self._job_id += 1
        meu_id = self._job_id
        self._set_botao(self.btn_resumir, False)
        self._set_botao(self.btn_cancelar, True)
        self._set_botao(self.btn_copiar, False)
        self.lbl_status.config(text="Processando...")
        self.lbl_copia.config(text="")
        self.lbl_contagem.config(text="", foreground=COR_NORMAL)
        self._set_saida("")

        t = threading.Thread(
            target=self._trabalho,
            args=(meu_id, texto, limite, tol1, tol2, tentativas, MODELO_OLLAMA),
            daemon=True,
        )
        t.start()
        self._after_id = self.after(100, self._checar_fila)

    def _ler_int(self, var: tk.StringVar, nome: str, minimo: int) -> int:
        try:
            v = int(var.get().replace(".", "").strip())   # tolera milhar (1.000)
        except (ValueError, tk.TclError):
            raise _ErroValidacao(f"{nome}: informe um numero inteiro.")
        if v < minimo:
            raise _ErroValidacao(f"{nome}: deve ser >= {minimo}.")
        return v

    def _cancelar(self) -> None:
        if not self._processando:
            return
        # Cancelamento IMEDIATO: invalida o job em voo (qualquer resultado que a
        # thread ainda produza sera descartado pela fila), libera a UI na hora e
        # NAO exibe nenhum resultado. A chamada de rede ja disparada termina
        # sozinha em segundo plano (thread daemon), porem e ignorada.
        self._cancelar_evt.set()
        self._job_id += 1                 # invalida o resultado da thread atual
        if self._after_id is not None:
            try:
                self.after_cancel(self._after_id)
            except tk.TclError:
                pass
            self._after_id = None
        self._finalizar_processamento()
        self._set_saida("")               # nao apresenta nenhum resultado
        self._set_botao(self.btn_copiar, False)
        self.lbl_contagem.config(text="", foreground=COR_NORMAL)
        self.lbl_status.config(text="Cancelado")

    def _trabalho(self, job_id: int, texto: str, limite: int, tol1: int,
                  tol2: int, tentativas: int, modelo: str) -> None:
        try:
            # max_tokens dimensionado pelo limite (em chars) com folga, para
            # nao truncar a saida em limites altos. Teto evita custo absurdo;
            # se ainda assim truncar, _chamar_ollama detecta (done_reason=length).
            max_toks = min(MAX_TOKENS_TETO,
                           max(MAX_TOKENS_MIN, int(limite * 1.5)))

            def chamar(sistema: str, usuario: str) -> str:
                return _chamar_ollama(modelo, sistema, usuario, max_toks)

            def relatar(tentativa: int, caracteres: int) -> None:
                # Roda na thread de trabalho: so enfileira (thread-safe); quem
                # toca a UI e o _checar_fila, na thread principal.
                self._fila.put((job_id, "progresso", (tentativa, caracteres)))

            res = resumir(
                texto, limite,
                tolerancia_1=tol1, tolerancia_2=tol2,
                max_tentativas=tentativas,
                chamar_llm=chamar,
                deve_cancelar=self._cancelar_evt.is_set,
                relatar=relatar,
            )
            self._fila.put((job_id, "ok", res))
        except Exception as e:  # erros de rede/API/logica
            self._fila.put((job_id, "erro", traduzir_excecao(e)))

    def _checar_fila(self) -> None:
        if not self.winfo_exists():  # janela ja foi fechada
            return
        # Drena tudo o que chegou: mensagens de PROGRESSO atualizam o rotulo ao
        # vivo (e seguem drenando); a final ('ok'/'erro') encerra.
        while True:
            try:
                jid, tipo, payload = self._fila.get_nowait()
            except queue.Empty:
                if self._processando:
                    self._after_id = self.after(80, self._checar_fila)
                return

            if jid != self._job_id:
                continue  # job cancelado/obsoleto -> descarta e segue drenando

            if tipo == "progresso":
                tent, chars = payload
                self.lbl_contagem.config(
                    text=f"Tentativa {tent} → {_milhar(chars)} caracteres",
                    foreground=COR_NORMAL)
                continue  # ao vivo: nao encerra ainda

            # tipo == 'ok' ou 'erro': encerra o processamento
            self._finalizar_processamento()
            if tipo == "erro":
                # after_idle evita abrir o dialogo modal de dentro do callback de
                # polling (no aqua do macOS isso pode prender o foco).
                self.after_idle(lambda p=payload: messagebox.showerror(
                    "Erro ao resumir", p))
                return

            res: Resultado = payload
            self._set_saida(res.texto)
            if res.texto.strip():
                self._set_botao(self.btn_copiar, True)

            if res.cancelado:
                sufixo = "  -  Cancelado"
            elif res.cortado:
                sufixo = "  -  (corte automatico aplicado)"
            else:
                sufixo = ""
            cor = COR_ALERTA if res.caracteres > res.limite else COR_NORMAL
            if res.tentativas == 0:                    # original ja cabia
                texto_lbl = f"{_milhar(res.caracteres)} caracteres (ja cabia no limite)"
            else:
                plural = "tentativa" if res.tentativas == 1 else "tentativas"
                texto_lbl = (f"{_milhar(res.caracteres)} caracteres em "
                             f"{res.tentativas} {plural}{sufixo}")
            self.lbl_contagem.config(text=texto_lbl, foreground=cor)
            return

    def _finalizar_processamento(self) -> None:
        self._processando = False
        try:
            self._set_botao(self.btn_resumir, True)
            self._set_botao(self.btn_cancelar, False)
            self.lbl_status.config(text="")
        except tk.TclError:
            pass

    # ---- utilitarios UI ---------------------------------------------------
    def _aspa_curva(self, event):
        """Ao digitar a aspa reta (U+0022), insere a aspa curva apropriada
        (“ na abertura, ” no fechamento) -- o glifo reto vira um copo na Garoa
        Light, entao nunca deve ser inserido."""
        if event.char != '"':
            return None  # qualquer outra tecla segue normalmente
        w = event.widget
        try:
            ant = w.get("insert -1c", "insert")
            w.insert("insert", "“" if (ant == "" or ant in _ABRE_ASPA) else "”")
        except tk.TclError:
            return None  # widget somente-leitura (ex.: janela do Manual)
        return "break"  # impede a insercao do U+0022

    def _colar_curvo(self, event, recontar):
        """Cola convertendo aspas retas em curvas (nunca exibir U+0022)."""
        w = event.widget
        try:
            conteudo = self.clipboard_get()
        except tk.TclError:
            return "break"
        try:
            w.delete("sel.first", "sel.last")   # substitui a selecao, se houver
        except tk.TclError:
            pass
        w.insert("insert", _curvar_aspas(conteudo))
        self.after(10, recontar)
        return "break"  # substitui o colar padrao (que traria U+0022)

    def _set_saida(self, texto: str) -> None:
        self.txt_saida.delete("1.0", "end")
        self.txt_saida.insert("1.0", _curvar_aspas(texto))  # nunca exibir U+0022
        self._recontar_saida()

    def _copiar(self) -> None:
        self.clipboard_clear()
        self.clipboard_append(self.txt_saida.get("1.0", "end-1c"))
        self.lbl_copia.config(text="Copiado!")
        if self._copia_after is not None:
            try:
                self.after_cancel(self._copia_after)
            except tk.TclError:
                pass
        self._copia_after = self.after(
            1500, lambda: self.lbl_copia.config(text=""))

    def _abrir_manual(self) -> None:
        """Abre o Manual numa janela pop-up SIMPLES: somente o texto (sem titulo,
        sem caixa/fio cinza e sem barra de scroll) e o botao Fechar. A altura se
        ajusta ao conteudo para tudo caber sem precisar rolar."""
        top = tk.Toplevel(self)
        top.withdraw()                 # nao mapear ainda: evita o "pisca + pulo"
        top.title("Como funciona")
        top.configure(bg=COR_FUNDO)
        top.transient(self)
        # texto cru sobre o fundo da janela: nada de borda, relevo ou scrollbar
        txt = tk.Text(top, wrap="word", width=60, relief="flat", bd=0,
                      highlightthickness=0, bg=COR_FUNDO, fg=COR_TEXTO,
                      font=self._f_caixa, padx=24, pady=18, cursor="arrow")
        txt.grid(row=0, column=0, sticky="nsew")
        txt.insert("1.0", _curvar_aspas(_MANUAL))
        # altura = linhas EXIBIDAS (medidas na largura estreita -> limite
        # superior; a janela final e mais larga, entao nada fica cortado mesmo
        # sem a barra de scroll).
        top.update_idletasks()
        try:
            dl = txt.count("1.0", "end-1c", "displaylines")
            linhas = dl[0] if isinstance(dl, (tuple, list)) else int(dl)
        except (tk.TclError, TypeError, ValueError):
            linhas = 24
        txt.config(height=max(8, linhas + 1), state="disabled")  # +1 de folga
        btn = self._criar_botao(top, "Fechar", top.destroy)
        btn.grid(row=1, column=0, pady=(4, 16))
        top.grid_rowconfigure(0, weight=1)
        top.grid_columnconfigure(0, weight=1)
        top.update_idletasks()
        w = max(560, top.winfo_reqwidth())
        sh = self.winfo_screenheight()
        h = min(top.winfo_reqheight(), sh - 140)
        x = max(0, (self.winfo_screenwidth() - w) // 2)
        y = max(24, (sh - h) // 3)
        top.geometry(f"{w}x{h}+{x}+{y}")
        top.update_idletasks()         # forca o WM a aplicar a geometria antes (macOS)
        top.bind("<Escape>", lambda e: top.destroy())
        top.deiconify()                # so agora exibe, ja na posicao final
        top.lift()
        top.focus_force()

    def _ao_fechar(self) -> None:
        # Sinaliza cancelamento (a thread daemon encerra logo) e fecha.
        self._cancelar_evt.set()
        if self._after_id is not None:
            try:
                self.after_cancel(self._after_id)
            except tk.TclError:
                pass
        self.destroy()


def main() -> None:
    registrar_fonte_embutida()
    App().mainloop()


if __name__ == "__main__":
    main()
