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
    - corte de seguranca no ultimo recurso (termina em "..." e cabe);
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
TOLERANCIA_RODADA_1 = 10         # caracteres a menos aceitos na 1a rodada
TOLERANCIA_RODADA_2 = 20         # caracteres a menos aceitos na 2a rodada
MAX_TENTATIVAS_PADRAO = 10       # total de tentativas (dividido entre as rodadas)
LIMIAR_CORTE_PEQUENO = 0.25      # r=(orig-limite)/orig <= isto -> deleta (nao resume)


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
    """Contagem exata: TODOS os caracteres (len puro). Fonte de verdade do
    programa -- a LLM nunca decide o tamanho."""
    return len(texto)


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
    "Voce e um editor que ENCURTA textos com FIDELIDADE ao original. Mantenha as "
    "palavras, as construcoes e a ordem das frases do autor; NAO parafraseie nem "
    "reescreva por reescrever. Para caber no limite, REMOVA o que for menos "
    "essencial (repeticoes, exemplos, detalhes secundarios, oracoes acessorias) "
    "e preserve o restante o mais proximo possivel do texto original, ajustando "
    "apenas o minimo necessario para a frase seguir gramatical e coesa. Nao "
    "invente informacao. Mantenha o idioma, o tom e a voz do autor. Seu objetivo "
    "NAO e so caber: e usar QUASE TODO o orcamento de caracteres, ficando o mais "
    "PROXIMO possivel do limite pedido SEM NUNCA ultrapassa-lo -- preserve o "
    "maximo de conteudo que o limite permitir e so corte o necessario para nao "
    "passar. Responda APENAS com o texto, sem comentarios, aspas ou rotulos."
)


_SISTEMA_MICRO = (
    "Voce e um editor de copy desk que ENXUGA textos com edicoes MINIMAS, mantendo "
    "TODAS as frases, fatos e a ordem. Voce NAO resume nem remove frases inteiras: "
    "aperta a redacao tirando palavras superfluas (artigos, pronomes e adverbios "
    "dispensaveis, repeticoes) e trocando expressoes longas por sinonimos mais "
    "curtos. REGRAS ABSOLUTAS DE FIDELIDADE: (1) use somente palavras INTEIRAS e "
    "reais do portugues; NUNCA abrevie nem corte palavras pela metade. (2) Preserve "
    "EXATAMENTE a grafia de todo nome proprio -- copie cada nome letra por letra "
    "(ex.: 'Monaco' continua 'Monaco', JAMAIS 'Monaca' ou 'Mon'). (3) NAO altere "
    "numeros nem datas. (4) Mantenha TODOS os itens de listas e enumeracoes -- nao "
    "remova nenhum (ex.: 'China, Japao, Miami, Canada e Monaco' deve continuar com "
    "os cinco). Mantenha idioma, tom e sentido. Responda apenas com o texto, sem "
    "comentarios nem rotulos."
)


def _prompt_micro_edicao(texto: str, limite: int, minimo: int) -> str:
    """Pede uma versao MICRO-EDITADA: mais enxuta com edicoes minimas e
    distribuidas, mantendo todas as frases. O modelo costuma ficar um pouco
    acima de `limite`; o Python apara o excesso depois (ver _aparar_para_caber).
    Por isso miramos um pouco abaixo do limite no pedido, para reduzir o aparo."""
    orig = len(texto)
    alvo = max(minimo, limite - 15)
    return (
        f"O texto abaixo tem {orig} caracteres. Reduza para perto de {alvo} "
        f"caracteres (no maximo {limite}) removendo cerca de {orig - alvo}, com "
        f"edicoes MINIMAS e distribuidas ao longo de TODO o texto: corte artigos e "
        f"palavras superfluas, troque palavras longas por sinonimos curtos, enxugue "
        f"as frases. NAO apague frases inteiras nem fatos: o resultado deve conter "
        f"TODAS as frases e informacoes do original, apenas mais enxutas. Preserve "
        f"EXATAMENTE nomes proprios (letra por letra), numeros e TODOS os itens de "
        f"listas. Fique o mais PROXIMO possivel de {limite} sem ultrapassar.\n\n"
        f"TEXTO:\n{texto}"
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
        f"(cerca de {palavras} palavras). Reescreva-o para ter entre {minimo} e "
        f"{limite} caracteres -- aproximadamente {palavras} palavras --, o mais "
        f"PROXIMO possivel de {limite} sem JAMAIS ultrapassar. Mantenha os fatos "
        f"centrais e o maximo de conteudo que couber; remova so o menos essencial. "
        f"Responda apenas com o texto.\n\nTEXTO:\n{texto}"
    )


def _prompt_encurtar(resumo: str, atual: int, limite: int, minimo: int) -> str:
    excesso = atual - limite
    palavras = _palavras_alvo(limite)
    return (
        f"O texto abaixo tem {atual} caracteres e ultrapassou o limite em "
        f"{excesso}. Encurte-o ate ficar entre {minimo} e {limite} caracteres "
        f"(cerca de {palavras} palavras), o mais PROXIMO possivel de {limite} sem "
        f"passar -- corte palavras supérfluas e detalhes secundarios, mantendo os "
        f"fatos centrais e sem parafrasear o que ficar. Responda apenas com o novo "
        f"texto.\n\nTEXTO ATUAL:\n{resumo}"
    )


def _prompt_expandir(texto: str, resumo: str, atual: int, limite: int,
                     minimo: int) -> str:
    faltam = limite - atual
    return (
        f"Sua versao atual do resumo tem apenas {atual} caracteres: faltam cerca "
        f"de {faltam} para chegar perto do limite de {limite}. Aumente-a para "
        f"ficar entre {minimo} e {limite} caracteres, o mais PROXIMO possivel de "
        f"{limite} sem ultrapassar. Reincorpore trechos e detalhes do TEXTO "
        f"ORIGINAL abaixo que voce havia cortado, usando as palavras e "
        f"construcoes do original, sem inventar nada. Responda apenas com o novo "
        f"texto.\n\nTEXTO ORIGINAL:\n{texto}\n\nSUA VERSAO ATUAL:\n{resumo}"
    )


def _cortar(texto: str, limite: int) -> str:
    """Corte de seguranca preservando palavras inteiras e terminando em '...'.

    Observacao: quando o limite e pequeno demais para comportar as reticencias
    (limite < 4), o resultado e apenas um fatiamento cru (sem '...'). A
    invariante 'nunca passa do limite' e sempre respeitada."""
    if contar(texto) <= limite:
        return texto
    reticencias = "..."
    alvo = max(0, limite - len(reticencias))
    cortado = texto[:alvo]
    if " " in cortado:
        cortado = cortado[: cortado.rfind(" ")]
    resultado = (cortado.rstrip() + reticencias) if alvo > 0 else texto[:limite]
    return resultado[:limite]


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
_MANUAL_PLACEHOLDER = """CortaTexto — Manual (rascunho de teste)

O QUE E
O CortaTexto encurta um texto para caber num limite de caracteres que voce
define, usando uma IA LOCAL (Ollama, modelo qwen3). Principio: quem conta os
caracteres e o proprio app — o resultado NUNCA passa do limite.

COMO USAR
1. Cole o texto no campo de cima ("Texto a ser tesourado").
2. Em "Reduza para", informe o limite em caracteres.
3. (Opcional) ajuste Tolerancia 1/2 e Max. tentativas.
4. Clique em Resumir (ou tecle Enter).
5. Acompanhe as tentativas ao vivo; no fim aparece "X caracteres em N tentativas".
6. Edite o resultado se quiser e clique em Copiar resultado.

OS CAMPOS
- Reduza para: o limite; o resultado nunca o ultrapassa.
- Tolerancia 1/2: o quao abaixo do limite e aceitavel (alvo de qualidade).
- Max. tentativas: teto de chamadas a IA.

BOM SABER
- Corte leve: o texto e enxugado mantendo todas as frases.
- Corte grande: vira um resumo de verdade.
- Cortes muito drasticos podem terminar em reticencias.
- Precisa do Ollama rodando e do modelo baixado (ollama pull qwen3).

(Texto de exemplo, apenas para testar esta janela. O manual definitivo sera
fornecido depois.)
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


def _nomes_proprios(texto: str) -> set:
    """Nomes proprios candidatos: tokens iniciados por MAIUSCULA que NAO abrem
    frase (estes seriam maiusculos so por posicao). Heuristica simples usada
    para verificar que a micro-edição nao perdeu nem mutilou um nome."""
    nomes = set()
    inicio = True                               # proxima palavra abre frase?
    for tok in re.finditer(r"\S+", texto):
        bruto = tok.group(0)
        palavra = bruto.strip(".,;:!?()[]{}\"'«»“”‘’…-—/")
        if not inicio and len(palavra) >= 2 and palavra[:1].isupper():
            nomes.add(palavra)
        inicio = bool(re.search(r"[.!?…][\"'»”’)\]]?$", bruto))
    return nomes


def _encolher_limpo(texto: str, limite: int, faixa: int) -> tuple:
    """Reduz `texto` para <= limite da forma MAIS limpa possivel, devolvendo
    (resultado, mecanico). Tenta, em ordem: (1) aparar clausulas acessorias
    (`_aparar_para_caber`); (2) terminar numa fronteira de FRASE <= limite; (3)
    so em ultimo caso, corte por palavra com '...' (`_cortar`, mecanico=True).
    Os dois primeiros nao truncam no meio de uma frase nem deixam '...'."""
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


def _fidelidade_violacoes(original: str, saida: str) -> int:
    """Quantos nomes proprios + numeros do `original` SUMIRAM de `saida`. Defesa
    contra a reescrita gerativa perder/mutilar um nome (ex.: 'Monaco' ->
    'Monaca', sumir 'Japao') ou um numero. 0 = totalmente fiel nesses tokens."""
    nums = set(re.findall(r"\d+", original)) - set(re.findall(r"\d+", saida))
    nomes = sum(1 for nome in _nomes_proprios(original) if nome not in saida)
    return len(nums) + nomes


def _fidelidade_ok(original: str, saida: str) -> bool:
    """True se `saida` preserva TODOS os nomes proprios e numeros do original."""
    return _fidelidade_violacoes(original, saida) == 0


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
    melhor: Optional[str] = None  # melhor candidato que cabe no limite

    def anota(caracteres: int) -> None:
        """Registra uma tentativa no historico e a reporta ao vivo (se houver
        callback `relatar`), para a interface mostrar o progresso."""
        historico.append(caracteres)
        if relatar is not None:
            relatar(len(historico), caracteres)

    def registra_melhor(cand: str) -> None:
        nonlocal melhor
        if cand and contar(cand) <= limite:
            if melhor is None or contar(cand) > contar(melhor):
                melhor = cand

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
        cortado = False
        if candidatos:
            final = max(candidatos, key=contar)
        elif cancelado_flag:
            final = ""  # cancelou antes de obter qualquer parcial valido
        else:
            # nada coube: reduz da forma mais LIMPA possivel (apara clausulas ou
            # termina numa frase); so usa corte com '...' em ultimo caso.
            final, cortado = _encolher_limpo(
                atual or "", limite, max(tolerancia_1, tolerancia_2))
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
    # Corte PEQUENO (limite perto do tamanho original): pedir um RESUMO faz o
    # modelo "desabar" para seu tamanho natural -- bem abaixo do limite. Em vez
    # disso pedimos uma MICRO-EDIÇÃO: o LLM enxuga o texto (tira artigos/palavras
    # superfluas, troca por sinonimos curtos) MANTENDO todas as frases. Ele
    # costuma ficar um pouco ACIMA do limite; o PYTHON entao apara o excesso
    # removendo o minimo de trechos acessorios (perto do topo, sem decapitar a 1a
    # frase nem o fecho). Qualidade do LLM + contagem garantida pelo Python.
    # Corte GRANDE segue pelo fluxo gerativo (resumo) abaixo.
    atual: Optional[str] = None
    remover = contar(texto) - limite          # > 0 aqui (ja passou o atalho)
    if remover > 0 and remover / contar(texto) <= LIMIAR_CORTE_PEQUENO:
        faixa = max(tolerancia_1, tolerancia_2)
        minimo = max(0, limite - tolerancia_1)
        # SALVAGUARDA DE FIDELIDADE: a reescrita gerativa as vezes perde/mutila um
        # nome proprio ou numero. Geramos ate 3 micro-edições e ficamos com a
        # PRIMEIRA que (apos o aparo) cabe E preserva todos os nomes/numeros. Se
        # nenhuma for perfeita, adotamos a MENOS infiel (menor numero de nomes/
        # numeros perdidos) -- e o resultado e editavel.
        editado: Optional[str] = None
        melhor_viol = None
        for _ in range(min(max_tentativas, 3)):
            if cancelado():
                break
            cand = validar(chamar_llm(_SISTEMA_MICRO,
                                      _prompt_micro_edicao(texto, limite, minimo)))
            if contar(cand) > limite:          # passou: apara o excesso (Python)
                aparado = _aparar_para_caber(cand, limite, faixa)
                if aparado is not None:
                    cand = aparado
            anota(contar(cand))                # reporta o tamanho efetivo da tentativa
            viol = _fidelidade_violacoes(texto, cand) if contar(cand) <= limite \
                else 1 + len(texto)            # nao coube: pior que qualquer fiel
            if melhor_viol is None or viol < melhor_viol:
                editado, melhor_viol = cand, viol   # guarda a melhor ate agora
            if contar(cand) <= limite and viol == 0:
                break                          # cabe E fiel: pronto
        if editado is None:                    # cancelou antes de qualquer resultado
            return finalizar("", True)
        registra_melhor(editado)
        return finalizar(editado, cancelado()) # <= limite garantido por finalizar

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
5QP2Ywp7LJG1wmKZsKiFxQphsVIoLMf+LmCPZbFWWKwQCiuEwjr4W0yLWv6ywl+tkFetWli0kJcV
5rLCXF1KmyWsCjv0BUpnOaui1KdQbstZnXBWVcJZnXBWJZzVCWdVwlmdcFYlnNXhAZaTmqUtIaxR
CKsUqirUZfiAeqGqTl3FMlbYqhO2KoStCmGrQtiqwILcBJ23sDSoW7EjDepulgb8wAO8WvJq5bcz
GtRPWRrUz1ga8C7/SRtY+1KjfsmSUwWWnHqYpUEsTo36jfo925baKvU4XFapN2CuSjfqcaoe5qaz
bakqE6qqoWou2/MgRgsxFXqxPlSNEW5qhZuscFMr3GSFm1rhJivc1Ao3WeGmVripFm6qhJsq4aaK
us9IubWU0nrvjwk7WujQQkdGuMhK22akJbW0kpaazkqdlUudlUudlUudlUudlUttlUttlUttlUt+
ZVI35ZJrmdRNuVy9luvWct22bGXiZ7W1zyqDhW7giqx9LuO9o4R4PSzxdUMQX19CfMMIiM+UEF+z
CfGJra0R4muE+MwmxOsS4iuF+MwmxGdKiM+UEJ8pIT4jxOsS4utSmzpIfFaIzwjxiTXNDEl8wrRt
kzHSGmOGYDQ7LKP1JYzmShitKWG0poTRmhJGa7bIqP1tmWxqqcTCJP1C2t+2SGJxtNS/lny15Kgl
Ly256PR3bqZSq1p+4aZq4FdtytNfrsmyp/yM5eecpSZddOrydjX1oov2nqVm8DpbzW0/p305EWny
WzcqPdr+l5xdIfuSMuWIMybB2t4QdWB61AI5T5ddkfxf9nzyf/UupGPXK5P/a7ZP1+NQI+nMx68f
T4tfrF7GJx1GvS3Xv8y0ZM7M3JB5PVufnZ09MfvN7NPZddnXy8aVzZdxR636A+u9iXCSKNLu+aPs
eaxkz//Knj/JHk2pfPm/jl4zRe2l5qr9IP5wanQpscBp1Nr56mnKOpH3n5H1RPWsrCepgPUk9j/P
ejLrx6XvPiF19KT8ZtFTUi99vE7EV2c4+jleJ+OrM+RmfzFoe87bhTqboV5Jc3lRUgkllXWSykuS
ysuSSjlXYOtnZ87/sxz5FzkyliNflyP/yuu09Gr/Jsf8vWSPkT3/kD015L8zbOxFXSywn/iSK38n
vc5kfa96Q854U/J5S/JZL/m8LSV6V66raK9LK7musWgP6nQ+dWnj8SPkl86X0ceXEwFdojrUSm2J
yWibblZnxeJZ21apy3mt1RW81mEBM/Qw2xOadDWv46w1VNvpMbzuoGtt7rrO5q7rJfcGm7u2UcEe
2MoM44AstNdzXqPk1iS5jZOyl3Ht47j2aXp7sY21akf9MdmCBP1x2apnazuOnj5QTzO4pv3peS3E
kp/XEyQ2zugdZJ3VO8q6TO8k60o9kbW135NkXUl8laGn78xrjZ4spd9F6s6WMqe211PS9HaV43az
x0k5pb317um7U+XdaZLK9I2Z0HvIle4pV7qX1OsnpF5nSHtb4sapCeRJm2PRMmo3LBrvEMtxnXq2
1N0cW3fYOOsBqzk+6RkDhOj5lMTW1wLWtrb2ZW3rap60XbO03T7Sdgslvf0kvUWSHj1O7y91arcO
kC17jTnbE/SBUvqDpPQHS+k/OdBayRGL5YhD5IhPyRGHyhFpSxP7Zhi3vsrrfVjyDF44U27UnH4r
Rw1UcmS13EOtpfT1eJAcxNqRzDi1HXl8jJocTx3twHXvRJ3anj6ZPrqr/LbYVMYz0yjzHtC1F2P1
GfjTvbmu2eSBBy9fOoTt+vfMwZkTM1/PrMuWZadlj86uzP7RWq+ymrK9ttmu92W7EjKT/OfioQ6k
X26zWNss1kfAYhEjziaaPo3Y+Czi4HPVJ7mKz6pDaMcj6N3ziMPnE7sfxvV9Wu0DqadD6tnEyufR
DxbC8Gdo70VEvtcR8dp4184jHkAPOYjRx2/UI+q36neMPRj5wE0j5CeWoyWlfjnRdTvU25h6pcqr
VYx0yrBml9E3rqTcl9IfOll/RV1OjJ1V/8Jxdn2+OlbZK/yi/U1FSncc7fxp9XlS+7L6KrVnRwM2
hj8cnYBsdHoSOpnUlzA+WKHs7OoydCpazXXa1yyvH0/td2U66rF2uzG10Il9HrTK9lcfvyvHX6++
T0z5A8ZjTdTqr9UuYkXs70kqenwTqdi51xwpzMXy2/HcnjKCtW3eOJCjzW3CgC+w9n/6gMW30fX1
8lroj5D/D+X9tfs=
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
COR_ALERTA = "#cc3333"       # acima do limite
COR_OK = "#2e7d32"           # feedback "Copiado!"
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
            font=self._f_base, **text_kw)
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

    # ---- construcao da UI -------------------------------------------------
    def _construir(self) -> None:
        # Estilos reutilizaveis (tema claro: texto PRETO, cinza so em bordas/campos).
        est_lbl = {"bg": COR_FUNDO, "fg": COR_TEXTO}
        est_dim = {"bg": COR_FUNDO, "fg": COR_TEXTO, "font": self._f_pequena}
        est_ent = {"bg": COR_SUPERFICIE, "fg": COR_TEXTO,
                   "insertbackground": COR_TEXTO, "relief": "flat",
                   "highlightthickness": 1, "highlightbackground": COR_BORDA,
                   "highlightcolor": COR_BORDA_FOCO}

        # ---- topo: LOGO no canto esquerdo, botao MANUAL no canto direito ----
        topo = tk.Frame(self, bg=COR_FUNDO)
        topo.grid(row=0, column=0, columnspan=4, sticky="ew", padx=14, pady=(12, 2))
        if self._logo_img is not None:
            tk.Label(topo, image=self._logo_img, bg=COR_FUNDO).grid(
                row=0, column=0, sticky="w")
        else:  # fallback textual se o logo nao carregar
            tk.Label(topo, text="CortaTexto", font=(self._familia, 26),
                     **est_lbl).grid(row=0, column=0, sticky="w")
        topo.grid_columnconfigure(1, weight=1)        # espacador empurra p/ direita
        self.btn_manual = self._criar_botao(topo, "Manual", self._abrir_manual,
                                            primario=True)  # cinza, como o Resumir
        self.btn_manual.grid(row=0, column=2, sticky="ne", pady=(55, 0))  # 53px abaixo

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
        params = tk.Frame(self, bg=COR_FUNDO)
        params.grid(row=3, column=0, columnspan=4, sticky="ew", padx=12, pady=2)
        # Quatro grupos (rotulo + campo) distribuidos pela largura toda:
        # grupos nas pontas e gaps iguais no meio ("space-between"). As colunas
        # IMPARES (3, 6, 9) sao espacadoras com peso -- elas absorvem a sobra de
        # largura e empurram os grupos para se distribuirem por igual. Dentro de
        # cada grupo o rotulo fica colado no seu campo (padx pequeno).
        celp = {"pady": 5}
        tk.Label(params, text="Reduza para:", **est_lbl).grid(
            row=0, column=0, padx=(0, 6), **celp)
        self.var_limite = tk.StringVar(value="280")
        self._campo_pilula(params, self.var_limite, 10)[0].grid(
            row=0, column=1, **celp)
        tk.Label(params, text="caracteres", **est_lbl).grid(
            row=0, column=2, padx=(6, 0), **celp)
        self.var_limite.trace_add("write", lambda *a: self._recontar_saida())
        tk.Label(params, text="Tolerância 1:", **est_lbl).grid(
            row=0, column=4, padx=(0, 6), **celp)
        self.var_tol1 = tk.StringVar(value=str(TOLERANCIA_RODADA_1))
        self._campo_pilula(params, self.var_tol1, 3)[0].grid(
            row=0, column=5, **celp)
        tk.Label(params, text="Tolerância 2:", **est_lbl).grid(
            row=0, column=7, padx=(0, 6), **celp)
        self.var_tol2 = tk.StringVar(value=str(TOLERANCIA_RODADA_2))
        self._campo_pilula(params, self.var_tol2, 3)[0].grid(
            row=0, column=8, **celp)
        tk.Label(params, text="Max. tentativas:", **est_lbl).grid(
            row=0, column=10, padx=(0, 6), **celp)
        self.var_tentativas = tk.StringVar(value=str(MAX_TENTATIVAS_PADRAO))
        self._campo_pilula(params, self.var_tentativas, 3)[0].grid(
            row=0, column=11, **celp)
        for _c in (3, 6, 9):                     # espacadores entre os grupos
            params.grid_columnconfigure(_c, weight=1, minsize=24)

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
            v = int(self.var_limite.get())
            return v if v > 0 else None
        except (ValueError, tk.TclError):
            return None

    def _atualizar_contagem_entrada(self, evento=None) -> None:
        if not self.winfo_exists():  # callback agendado pode rodar pos-fechamento
            return
        n = contar(self.txt_entrada.get("1.0", "end-1c"))
        lim = self._limite_atual()
        extra = "  (ja cabe no limite)" if lim is not None and n <= lim else ""
        self.lbl_entrada_contagem.config(text=f"Original: {n} caracteres{extra}")

    def _recontar_saida(self, evento=None) -> None:
        if not hasattr(self, "txt_saida"):
            return
        if self._processando:
            return  # durante o processamento o rotulo mostra o progresso ao vivo
        n = contar(self.txt_saida.get("1.0", "end-1c"))
        lim = self._limite_atual()
        if lim is None:
            self.lbl_contagem.config(text=f"{n} caracteres", foreground=COR_NORMAL)
            return
        if n > lim:
            self.lbl_contagem.config(
                text=f"{n} / {lim} caracteres  -  acima do limite!",
                foreground=COR_ALERTA)
        else:
            self.lbl_contagem.config(text=f"{n} / {lim} caracteres",
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
            v = int(var.get().strip())
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
                    text=f"Tentativa {tent} → {chars} caracteres",
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
                texto_lbl = f"{res.caracteres} caracteres (ja cabia no limite)"
            else:
                plural = "tentativa" if res.tentativas == 1 else "tentativas"
                texto_lbl = (f"{res.caracteres} caracteres em "
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
        """Abre o Manual numa janela pop-up no mesmo visual da interface (fundo
        branco, fonte Garoa, caixa branca arredondada com a barra de scroll
        custom). Texto ainda e placeholder."""
        top = tk.Toplevel(self)
        top.title("Manual")
        top.configure(bg=COR_FUNDO)
        top.transient(self)
        tk.Label(top, text="Manual do CortaTexto", bg=COR_FUNDO, fg=COR_TEXTO,
                 font=self._f_base).grid(row=0, column=0, sticky="w",
                                         padx=16, pady=(14, 6))
        card, txt = self._caixa_texto(top, height=22, wrap="word")
        card.grid(row=1, column=0, sticky="nsew", padx=14, pady=4)
        txt.insert("1.0", _curvar_aspas(_MANUAL_PLACEHOLDER))
        txt.config(state="disabled")               # somente leitura
        btn = self._criar_botao(top, "Fechar", top.destroy)
        btn.grid(row=2, column=0, pady=(6, 14))
        top.grid_rowconfigure(1, weight=1)
        top.grid_columnconfigure(0, weight=1)
        top.minsize(480, 360)
        top.update_idletasks()
        w = max(620, top.winfo_reqwidth())
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
