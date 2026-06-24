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
import socket
import sys
import threading
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


def _aparar_para_caber(texto: str, limite: int, faixa: int) -> Optional[str]:
    """Apara `texto` para caber em `limite` removendo o MINIMO de trechos
    acessorios (heuristica pura, SEM LLM), o mais perto possivel do topo, sem
    decapitar a 1a frase nem cortar o fecho. Usado para aparar o excesso da
    micro-edição do LLM (que costuma ficar um pouco acima do limite). Devolve o
    texto aparado (sempre <= limite) ou None se nao der para caber so removendo
    acessorios (ai o corte de seguranca de `finalizar` assume)."""
    excesso = len(texto) - limite
    if excesso <= 0:
        return texto                                # ja cabe
    spans = _spans_heuristicos(texto)
    # QUALIDADE: nao decapitar a abertura nem cortar o fecho. Apara do corpo; so
    # mexe na 1a/ultima frase se nao houver material suficiente no meio.
    fim_1a, ini_ult = _limites_abertura_fecho(texto)
    corpo = [(a, b) for (a, b) in spans if a >= fim_1a and b <= ini_ult]
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

    # (2) encurtar o MINIMO de frases: economia de cada uma ao trocar orig->curta
    economia = [contar(originais[i]) - contar(curtas[i]) for i in range(n)]
    excesso = contar(full) - limite
    idx_red = [i for i in range(n) if economia[i] > 0]
    if idx_red:
        comps = [economia[i] for i in idx_red]
        mapa = _mapa_subconjuntos(comps)
        viaveis = sorted(s for s in mapa if s >= excesso)
        if viaveis:                               # da pra caber so encurtando
            encurtar = {idx_red[k] for k in mapa[viaveis[0]]}
            return juntar(todas, lambda i: curtas[i] if i in encurtar else originais[i])

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
AAcB0Ac4yvfxp6D++YD1G8CaDVi/gboV/1vyUPcv5aEt5KznaRwJOcfAd9Q96CkEU+H3UoBlAMsB
PABWAKwD3oYCrAcIA1ADhMP9DQCRAFEA0QAxAFi6FfRocoDWABP4vgx7o2fCaXBFCvzAu6sH764e
vLt68O7qwburB++uHrw7sAIAVgAsQFuAdgDtAToAdAToDNCVrz8HHl498KBe4MhzvR68vbqmEUC9
AHrHlhEA9oGz4Nk8gPk8v+oEXwMs4i1QPXh/deD91YP3VwfeXx14f3Xg+dXxbXyAl95akNLrIJHX
wbuqA++qDryrOvCu6sC7qgPvqg68qzrwrurAu6oD76oOvKs68K7qwLuqA++qDryrOvCu6sC7qgPv
qg68qzrwrurAu6oD76oOvKs68K7qwLuqA++qDryrOvCu6sC7qgPvqg68qzrwrurAu6oD76oOvKs6
8K7qwLuqA++qDryrOvCu6sC7qgPvqg68q//T3plAx1GcebxqRrd1jCVDYnPYBmNsIDY+ZTDmigmE
yIQjmITTHOIQ5nBQCGSJOGRgEl4OHK6AuJKNX4iyD3Mlm857q00yYUMS9S7psGkOAW6ONtCQNEkq
gIHZX33dksaHZMEq4Mez+v2nWz3dVdVVv/q+r6p7ZiKiq4joKiK6ioiuIqKriOgqIroyGpaIsAwR
liHCMkRYhqjCCHlLhB0j8cPRElkYfTlrWp1IKiKSioikIvXAwNitFtWhetSAcmhsOqYbfjwXM57r
UHuw3hPthT6BZqCZQvylaha1PqfYphak47xF6EAZ7zm0diTjPThjzNdBi0e0eKSOkXGfQ4s7tLhD
i3eok999W52CTiWt09l3Bsecyfps1IbOQyvI6xL2fw1dh76BvoW+LT4jwj5G2McI+xhhHyPsY4R9
jJStjdtQF7od3YHuFILsDEaEfYywjxH2McI+UnNCVoyNjLCRETbSWpMebGSEjYywkRE2MsJGRtjI
CPo6sZER9tH6lAgSO7GNEbYxwjZG2MYI2xhBpwOdDnQ60OlApwOdDnQ60OlApwOdDnQ60OlApwOd
DnQ60OlApwOdDnQ60OlApwOdDnQ60OlApwOdDnQ60OlApwOdDnQ60OlApwOdDnQ60OlApwOdDnQ6
0OlApwOdDnQ60OlApwOdDnQ60OlApwOdMXQG0BlDZwydMXTGjGk7GNN2MKbtYEzbwZi2gzFtB2Pa
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
fZNw3yjcN8H9iWyfBP1NQv9YoT8n9OeE/hw1d7L0gLHCaC3LWPlViEa2LJflA/ZjCnsskbXCYpmw
qIXFCmGxUigsx/4uYI9lsVZYrBAKK4TCOvhbTIta/rLCX62QV61aWLSQlxXmssJcXUqbJawKO/QF
Smc5q6LUp1Buy1mdcFZVwlmdcFYlnNUJZ1XCWZ1wViWc1eEBlpOapS0hrFEIqxSqKtRl+IB6oapO
XcUyVtiqE7YqhK0KYatC2KrAgtwEnbewNKhbsSMN6m6WBvzAA7xa8mrltzMa1E9ZGtTPWBrwLv9J
G1j7UqN+yZJTBZacepilQSxOjfqN+j3bltoq9ThcVqk3YK5KN+pxqh7mprNtqSoTqqqhai7b8yBG
CzEVerE+VI0RbmqFm6xwUyvcZIWbWuEmK9zUCjdZ4aZWuKkWbqqEmyrhpoq6z0i5tZTSeu+PCTta
6NBCR0a4yErbZqQltbSSlprOSp2VS52VS52VS52VS52VS22VS22VS22VS35lUjflkmuZ1E25XL2W
69Zy3bZsZeJntbXPKoOFbuCKrH0u472jhHg9LPF1QxBfX0J8wwiIz5QQX7MJ8YmtrRHia4T4zCbE
6xLiK4X4zCbEZ0qIz5QQnykhPiPE6xLi61KbOkh8VojPCPGJNc0MSXzCtG2TMdIaY4ZgNDsso/Ul
jOZKGK0pYbSmhNGaEkZrtsio/W2ZbGqpxMIk/ULa37ZIYnG01L+WfLXkqCUvLbno9HduplKrWn7h
pmrgV23K01+uybKn/Izl55ylJl106vJ2NfWii/aepWbwOlvNbT+nfTkRafJbNyo92v6XnF0h+5Iy
5YgzJsHa3hB1YHrUAjlPl12R/F/2fPJ/9S6kY9crk/9rtk/X41Aj6czHrx9Pi1+sXsYnHUa9Lde/
zLRkzszckHk9W5+dnT0x+83s09l12dfLxpXNl3FHrfoD672JcJIo0u75o+x5rGTP/8qeP8keTal8
+b+OXjNF7aXmqv0g/nBqdCmxwGnU2vnqaco6kfefkfVE9aysJ6mA9ST2P896MuvHpe8+IXX0pPxm
0VNSL328TsRXZzj6OV4n46sz5GZ/MWh7ztuFOpuhXklzeVFSCSWVdZLKS5LKy5JKOVdg62dnzv+z
HPkXOTKWI1+XI//K67T0av8mx/y9ZI+RPf+QPTXkvzNs7EVdLLCf+JIrfye9zmR9r3pDznhT8nlL
8lkv+bwtJXpXrqtor0srua6xaA/qdD51aePxI+TnypfRx5cTAV2iOtRKbYnJaJtuVmfF4lnbVqnL
ea3VFbzWYQEz9DDbE5p0Na/jrDVU2+kxvO6ga23uus7mrusl9wabu7ZRwR7YygzjgCy013Neo+TW
JLmNk7KXce3juPZpenuxjbVqR/0x2YIE/XHZqmdrO46ePlBPM7im/el5LcSSn9cTJDbO6B1kndU7
yrpM7yTrSj2RtbXfk2RdSXyVoafvzGuNniyl30XqzpYyp7bXU9L0dpXjdrPHSTmlvfXu6btT5d1p
ksr0jZnQe8iV7ilXupfU6yekXmdIe1vixqkJ5EmbY9EyajcsGu8Qy3GderbU3Rxbd9g46wGrOT7p
GQOE6PmUxNbXAta2tvZlbetqnrRds7TdPtJ2CyW9/SS9RZIePU7vL3Vqtw6QLXuNOdsT9IFS+oOk
9AdL6T850FrJEYvliEPkiE/JEYfKEWlLE/tmGLe+yut9WPIMXjhTbtScfitHDVRyZLXcQ62l9PV4
kBzE2pHMOLUdeXyMmhxPHe3Ade9EndqePpk+uqv8tthUxjPTKPMe0LUXY/UZ+NO9ua7Z5IEHL186
hO3698zBmRMzX8+sy5Zlp2WPzq7M/tFar7Kasr222a73ZbsSMpP85+KhDqRfbrNY2yzWR8BiESPO
Jpo+jdj4LOLgc9UnuYrPqkNoxyPo3fOIw+cTux/G9X1a7QOpp0Pq2cTK59EPFsLwZ2jvRUS+1xHx
2njXziMeQA85iNHHb9Qj6rfqd4w9GPnATSPkJ5ajJaV+OdF1O9TbmHqlyqtVjHTKsGaX0TeupNyX
0h86WX9FXU6MnVX/wnF2fb46Vtkr/KL9TUVKdxzt/Gn1eVL7svoqtWdHAzaGPxydgGx0ehI6mdSX
MD5Yoezs6jJ0KlrNddrXLK8fT+13ZTrqsXa7MbXQiX0etMr2Vx+/K8dfr75PTPkDxmNN1Oqv1S5i
RezvSSp6fBOp2LnXHCnMxfLb8dyeMoK1bd44kKPNbcKAL7D2f/qAxbfR9fXyWuiPkP8PFJ3pUw==
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
        top.bind("<Escape>", lambda e: top.destroy())
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
