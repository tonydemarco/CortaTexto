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
NJKn/cFHHXvdmn3fUrOU+eZbyM3WfwLrYG+4
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

# Logo (wordmark) embutido como PNG base64 -- preto sobre o fundo branco.
# Gerado de CortaTexto_logo.png: composto sobre branco, a luminancia vira o
# alfa de uma silhueta preta, redimensionado p/ altura 120 (249x120).
_LOGO_GIF_B64 = """
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
mUqHyv8DZ/hTV4qrTIoAAAAASUVORK5CYII=
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
