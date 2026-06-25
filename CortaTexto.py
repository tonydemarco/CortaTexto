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
PlrbB1qbkO91Xjv0jEZfedx/p9AsI5K8e8PntYc2ksOkfNt/B2ORKy13w065tlzmtZ2S0viF+yPK
fev4FmbEwnJ/zG+UJ+Z8t5vRqptRqZtRqZtRqZtRqZtRqZtRqZtRqZtRqZuRohtLvL63fQuy/uzf
K+ezd9EDPUUulxnE2d6jzJF3N++e8x2G8YXnj4GeO5b7b62+eh1RedflaZhCrbTfK7o/j19gmRd2
VTC4UL+RDG0JR1SBXl3ctbnHzvnPbXWz7ltuzs2A7Tp8mqjQdzbrWt4Re0veg5o6rf98Gmv03uRs
J+VkKKMVy3RaBzlz8pZU6XKljM+XbPvfSNVlS1q7VzbHNkdZscmWo5xXk1+b43ax7/RKl5SsfmMr
p4vJn5bURsi4je5OKb3Dq9GzSs5om0hze/U7Xr9sL0+u79PTzq+lL0+/np/lU8HAJNH9+6GH5O9d
yu2i5kz/NU1+CX35joVeIJ/3/jntt/pR+WvZEHW3/N5Yg/zwXKPapu5Ru9Qe1axa1EHVqg6rI+o5
dVQ9r/Rb7gCp+j32ZvkltAckZRclF0CS8oapUlVCpBujLlJjVZUarqqJWOXqRrYq7vRV6nr5hbMF
1LBHLaG0VrVUvUjZt6mX1Sm90LuaQFn6F7fyaFdUfr8rLr/g5f3yVoLyk2oEdejf3RpNTWPVODVe
Xc51E4kky2nPVtqwXd2vdqhDlK3L1RpV6GXtsaeWMm9gK8SKZZRZR90XYdVadbG8aW+hzUH5a2KM
dm7Glm1sIdr7PT4fYAuRp4Xzh9gC1KB//UzXoq82IMq3AuwMYucIck+QNk1kC2PhZFqma4tJKVEp
JSKlRKSUiPSUrjkgNQek5oDUrGu4mCtj8lcDrYve4tSW4EwJWwh1kqTpmsPSEwYKjeVzHFseSo3n
++VsEbErKnbFxK581Fsuv7t5N59b2QxR0kDL+/ncwWaIvXGxNy72xql5vLS6qL+vPrNjGFuBWFP4
BWs8O7QFIbEgoK5Qk/r1qVAzsK+KLaJm0mMRNZctohbRYxHfyjq2iKpni6gVbBG8aw0WaG3j6i62
fLWeLd9vj1Y06LdK6xr026bVDfot1BoH/Xbq3olJa6PS2qi0Niq/dBeQniiUlgakFQFpRVDsD4n9
eWJ/ntifJ/bnif15YnmeWJ4nlnv+EKaMw37J4bOUrPN8+Vdc5eifOTiqOA==
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
