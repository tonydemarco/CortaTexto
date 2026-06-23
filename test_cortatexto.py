# -*- coding: utf-8 -*-
"""
Testes do CortaTexto com a LLM mockada (sem chamar a API real).

A logica de resumo (`resumir`) recebe `chamar_llm` por injecao de dependencia,
entao basta passar uma funcao fake. Importar o modulo NAO cria janela Tk nem
exige o pacote `anthropic`.

Rode com:  pytest -q
"""

import pytest

import CortaTexto as ct


# --------------------------------------------------------------------------
# Contagem exata (code points: acentos, espacos, pontuacao)
# --------------------------------------------------------------------------
def test_contar_acentos_e_espacos():
    # Acentos contam como UM code point cada (len conta code points Unicode).
    assert ct.contar("ção") == 3        # "cao" com cedilha+til = 3
    assert ct.contar("café") == 4            # "cafe" com agudo = 4
    assert ct.contar("ação e reacão!") == len("ação e reacão!")
    assert ct.contar("ola mundo, tudo bem?") == 20  # espacos e pontuacao contam
    assert ct.contar("ola\nmundo") == 9            # a quebra de linha conta


# --------------------------------------------------------------------------
# Aceitacao imediata: ja cabe na faixa -> nao chama a LLM
# --------------------------------------------------------------------------
def test_ja_cabe_sem_chamar_llm():
    chamadas = []

    def fake(sistema, usuario):
        chamadas.append((sistema, usuario))
        return "nao deveria ser chamado"

    texto = "a" * 275  # dentro de [280-10, 280]
    r = ct.resumir(texto, 280, chamar_llm=fake)
    assert chamadas == []
    assert r.tentativas == 0
    assert r.sucesso and not r.cortado and not r.cancelado
    assert r.texto == texto and r.caracteres == 275


# --------------------------------------------------------------------------
# Loop de encurtamento quando acima do limite
# --------------------------------------------------------------------------
def test_curvar_aspas_abre_e_fecha():
    # U+0022 (copo na Garoa) -> curvas: “ na abertura, ” no fechamento
    assert ct._curvar_aspas('ele disse "oi" e saiu') == 'ele disse “oi” e saiu'
    assert ct._curvar_aspas('"inicio" e "fim"') == '“inicio” e “fim”'
    assert ct._curvar_aspas("sem aspas") == "sem aspas"
    assert '"' not in ct._curvar_aspas('um "teste" aqui')   # nunca sobra U+0022
    # troca 1:1 em code points -> NAO altera a contagem de caracteres
    s = 'a "b c" d'
    assert len(ct._curvar_aspas(s)) == len(s)


def test_relatar_reporta_cada_tentativa():
    # o callback `relatar` recebe (numero_da_tentativa, caracteres) a cada passo
    saidas = iter(["a" * 400, "a" * 350, "a" * 278])
    reportes = []
    r = ct.resumir("orig " * 300, 280,
                   chamar_llm=lambda s, u: next(saidas),
                   relatar=lambda t, c: reportes.append((t, c)))
    assert reportes == [(1, 400), (2, 350), (3, 278)]
    assert r.historico == [400, 350, 278]


def test_encurtamento_converge():
    saidas = iter(["a" * 400, "a" * 350, "a" * 278])

    def fake(sistema, usuario):
        return next(saidas)

    r = ct.resumir("orig " * 300, 280, chamar_llm=fake)
    assert r.caracteres <= 280
    assert r.caracteres == 278
    assert r.sucesso and not r.cortado
    assert r.historico == [400, 350, 278]


# --------------------------------------------------------------------------
# Expansao quando curto demais
# --------------------------------------------------------------------------
def test_expansao_converge():
    saidas = iter(["a" * 100, "a" * 275])

    def fake(sistema, usuario):
        return next(saidas)

    r = ct.resumir("x" * 1000, 280, chamar_llm=fake)
    assert 270 <= r.caracteres <= 280
    assert r.sucesso


# --------------------------------------------------------------------------
# Corte de seguranca (ultimo recurso): nunca passa do limite, termina em "..."
# --------------------------------------------------------------------------
def test_corte_de_seguranca():
    def fake(sistema, usuario):
        return "palavra " * 100  # sempre 800 chars, nunca cabe

    r = ct.resumir("x" * 1000, 50, max_tentativas=3, chamar_llm=fake)
    assert r.caracteres <= 50          # invariante: nunca passa do limite
    assert r.cortado is True
    assert r.texto.endswith("...")
    assert not r.sucesso  # nao convergiu pela LLM


def test_nunca_passa_do_limite_em_varios_limites():
    def teimoso(sistema, usuario):
        return "lorem ipsum dolor sit amet " * 50

    for limite in (1, 2, 3, 5, 10, 40, 137, 280):
        r = ct.resumir("conteudo " * 200, limite, max_tentativas=2,
                       chamar_llm=teimoso)
        assert r.caracteres <= limite, f"estourou no limite {limite}"


# --------------------------------------------------------------------------
# Resposta vazia / None nunca viram "sucesso"
# --------------------------------------------------------------------------
def test_resposta_vazia_falha():
    def fake(sistema, usuario):
        return ""

    with pytest.raises(ct.RespostaVaziaError):
        ct.resumir("x" * 1000, 280, chamar_llm=fake)


def test_resposta_so_espacos_falha():
    def fake(sistema, usuario):
        return "   \n  "

    with pytest.raises(ct.RespostaVaziaError):
        ct.resumir("x" * 1000, 280, chamar_llm=fake)


def test_resposta_none_falha():
    def fake(sistema, usuario):
        return None  # type: ignore[return-value]

    with pytest.raises(ct.RespostaVaziaError):
        ct.resumir("x" * 1000, 280, chamar_llm=fake)


# --------------------------------------------------------------------------
# Validacao do contrato publico
# --------------------------------------------------------------------------
def test_tolerancia_negativa_levanta_valueerror():
    with pytest.raises(ValueError):
        ct.resumir("x" * 1000, 280, tolerancia_1=-1,
                   chamar_llm=lambda s, u: "a" * 200)


def test_limite_invalido_levanta_valueerror():
    with pytest.raises(ValueError):
        ct.resumir("x" * 100, 0, chamar_llm=lambda s, u: "a" * 10)


# --------------------------------------------------------------------------
# Cancelamento preserva o melhor parcial que ja cabe
# --------------------------------------------------------------------------
def test_cancelamento_preserva_melhor_parcial():
    # inicial estoura (400); 1o encurtamento cai para 250 (cabe, mas abaixo da
    # faixa) -> vira "melhor"; cancelamos logo apos esse encurtamento.
    saidas = iter(["a" * 400, "a" * 250, "a" * 999])
    chamadas = {"n": 0}

    def fake(sistema, usuario):
        chamadas["n"] += 1
        return next(saidas)

    def deve_cancelar():
        # cancela depois que a LLM ja produziu o 1o parcial que cabe (2 chamadas)
        return chamadas["n"] >= 2

    r = ct.resumir("x" * 2000, 280, chamar_llm=fake,
                   deve_cancelar=deve_cancelar)
    assert r.cancelado is True
    assert r.caracteres <= 280
    assert r.caracteres == 250  # preservou o parcial que cabia
    assert not r.sucesso


def test_cancelamento_antes_de_qualquer_chamada():
    chamadas = []

    def fake(sistema, usuario):
        chamadas.append(1)
        return "a" * 100

    r = ct.resumir("x" * 1000, 280, chamar_llm=fake,
                   deve_cancelar=lambda: True)
    assert r.cancelado is True
    assert chamadas == []  # cancelou antes de gastar qualquer chamada
    assert r.caracteres <= 280


# --------------------------------------------------------------------------
# _cortar: corte por palavras inteiras, respeitando o limite
# --------------------------------------------------------------------------
def test_cortar_preserva_palavras_e_limite():
    txt = "O rato roeu a roupa do rei de Roma ontem a noite toda"
    out = ct._cortar(txt, 20)
    assert len(out) <= 20
    assert out.endswith("...")
    # o trecho antes do "..." e um prefixo do original (nao corta no meio de
    # palavra: o corte foi feito num espaco e depois aparado)
    assert txt.startswith(out[:-3])
    assert not out[:-3].endswith(" ")


def test_cortar_nao_altera_texto_que_ja_cabe():
    assert ct._cortar("curto", 50) == "curto"


# --------------------------------------------------------------------------
# Fonte embutida: o binario decodifica e tem assinatura de OpenType
# --------------------------------------------------------------------------
def test_bytes_fonte_decodifica():
    raw = ct._bytes_fonte()
    assert isinstance(raw, bytes) and len(raw) > 10000
    # Assinatura OTF/CFF = b"OTTO"; TrueType = 0x00010000
    assert raw[:4] in (b"OTTO", b"\x00\x01\x00\x00", b"true", b"ttcf")


# --------------------------------------------------------------------------
# Texto original que ja cabe no limite e devolvido sem chamar a LLM
# --------------------------------------------------------------------------
def test_original_abaixo_da_faixa_nao_chama_llm():
    chamadas = []

    def fake(sistema, usuario):
        chamadas.append(1)
        return "qualquer coisa"

    texto = "a" * 100  # bem abaixo da faixa, mas <= limite 280
    r = ct.resumir(texto, 280, chamar_llm=fake)
    assert chamadas == []          # nao expande um original que ja servia
    assert r.tentativas == 0
    assert r.texto == texto and r.caracteres == 100
    assert r.sucesso and not r.cortado


# --------------------------------------------------------------------------
# Tolerancia progressiva: aceitacao so na RODADA 2 (mecanismo central da spec)
# --------------------------------------------------------------------------
def test_aceita_na_rodada_2():
    # 265 esta fora da faixa da rodada 1 [270,280], mas dentro da rodada 2
    # [260,280]. Logo a rodada 1 se esgota e a rodada 2 aceita.
    def fake(sistema, usuario):
        return "a" * 265

    r = ct.resumir("x" * 1000, 280, chamar_llm=fake)  # tol1=10, tol2=20, mt=10
    assert r.sucesso
    assert r.caracteres == 265
    # 1 chamada inicial + 4 da rodada 1 (tent_r1 = max(1, 9//2) = 4); a rodada 2
    # aceita ja na verificacao, sem nova chamada.
    assert r.tentativas == 5


def test_divisao_de_tentativas_garante_rodada_2():
    # Com max_tentativas=3 (restantes=2 -> 1 por rodada), a rodada 2 ainda e
    # alcancada e aceita o 265.
    def fake(sistema, usuario):
        return "a" * 265

    r = ct.resumir("x" * 1000, 280, max_tentativas=3, chamar_llm=fake)
    assert r.sucesso and r.caracteres == 265


# --------------------------------------------------------------------------
# max_tentativas=1 dispara EXATAMENTE 1 chamada (sem estourar o orcamento)
# --------------------------------------------------------------------------
def test_max_tentativas_1_faz_uma_chamada():
    chamadas = []

    def fake(sistema, usuario):
        chamadas.append(1)
        return "a" * 500  # estoura; nao havera orcamento para encurtar

    r = ct.resumir("x" * 1000, 280, max_tentativas=1, chamar_llm=fake)
    assert len(chamadas) == 1          # nao paga por chamada extra
    assert r.tentativas == 1
    assert r.caracteres <= 280
    assert r.cortado and not r.sucesso  # caiu no corte de seguranca


# --------------------------------------------------------------------------
# Resultado abaixo da faixa (loop nao converge) ainda e sucesso (cabe no limite)
# --------------------------------------------------------------------------
def test_abaixo_da_faixa_mas_dentro_do_limite_e_sucesso():
    def fake(sistema, usuario):
        return "a" * 250  # cabe em 280, mas abaixo da faixa final [260,280]

    r = ct.resumir("x" * 1000, 280, chamar_llm=fake)
    assert r.caracteres == 250
    assert r.sucesso is True
    assert not r.cortado and not r.cancelado


# --------------------------------------------------------------------------
# Cancelamento que cai de volta no 'melhor' anterior (candidato atual estoura)
# --------------------------------------------------------------------------
def test_cancelamento_cai_no_melhor_anterior():
    # 400 (estoura) -> 260 (cabe, vira melhor) -> 300 (estoura) -> cancela.
    saidas = iter(["a" * 400, "a" * 260, "a" * 300])
    chamadas = {"n": 0}

    def fake(sistema, usuario):
        chamadas["n"] += 1
        return next(saidas)

    def deve_cancelar():
        return chamadas["n"] >= 3  # cancela apos o candidato que estoura (300)

    r = ct.resumir("x" * 2000, 280, chamar_llm=fake,
                   deve_cancelar=deve_cancelar)
    assert r.cancelado is True
    assert r.caracteres == 260  # caiu de volta no melhor parcial que cabia
    assert not r.sucesso


# --------------------------------------------------------------------------
# Exclusividade: sucesso=True implica nao-cortado e nao-cancelado
# --------------------------------------------------------------------------
def test_sucesso_exclui_cortado_e_cancelado():
    r_ok = ct.resumir("x" * 1000, 280, chamar_llm=lambda s, u: "a" * 275)
    r_cut = ct.resumir("x" * 1000, 50, max_tentativas=2,
                       chamar_llm=lambda s, u: "pal " * 100)
    r_cancel = ct.resumir("x" * 1000, 280, chamar_llm=lambda s, u: "a" * 400,
                          deve_cancelar=lambda: True)
    for r in (r_ok, r_cut, r_cancel):
        if r.sucesso:
            assert not r.cortado and not r.cancelado
    assert r_ok.sucesso and not r_ok.cortado and not r_ok.cancelado
    assert r_cut.cortado and not r_cut.sucesso
    assert r_cancel.cancelado and not r_cancel.sucesso


# --------------------------------------------------------------------------
# Expansao realmente aciona o prompt de expansao (nao so chega ao tamanho)
# --------------------------------------------------------------------------
def test_expansao_usa_prompt_de_expansao():
    prompts = []
    saidas = iter(["a" * 100, "a" * 275])

    def fake(sistema, usuario):
        prompts.append(usuario)
        return next(saidas)

    r = ct.resumir("x" * 1000, 280, chamar_llm=fake)
    assert 270 <= r.caracteres <= 280
    assert any("Reincorpore" in p for p in prompts)  # marca do prompt de expandir


# --------------------------------------------------------------------------
# _cortar com limite pequeno (< 4): apenas fatia, sem reticencias
# --------------------------------------------------------------------------
def test_cortar_limite_pequeno_sem_reticencias():
    for lim in (1, 2, 3):
        out = ct._cortar("palavra longa demais", lim)
        assert len(out) <= lim
        assert not out.endswith("...")
    out = ct._cortar("palavra longa demais", 6)
    assert len(out) <= 6 and out.endswith("...")


# --------------------------------------------------------------------------
# Camada Ollama (_extrair_texto_ollama) -- parsing da resposta local
# --------------------------------------------------------------------------
def test_extrair_texto_ollama_basico():
    corpo = {"message": {"role": "assistant", "content": "  Resumo local  "},
             "done_reason": "stop"}
    assert ct._extrair_texto_ollama(corpo) == "Resumo local"


def test_extrair_texto_ollama_remove_think():
    # modelos como o Qwen3 podem emitir blocos <think>...</think>
    corpo = {"message": {"content": "<think>raciocinio</think>Resumo final"}}
    assert ct._extrair_texto_ollama(corpo) == "Resumo final"


def test_extrair_texto_ollama_length_levanta_erro():
    corpo = {"message": {"content": "x" * 50}, "done_reason": "length"}
    with pytest.raises(ct.ErroResumo) as info:
        ct._extrair_texto_ollama(corpo)
    assert "truncada" in str(info.value)


def test_extrair_texto_ollama_vazio():
    for corpo in ({}, {"message": {}}, {"message": {"content": "   "}},
                  {"message": {"content": "<think>so pensamento</think>"}}):
        with pytest.raises(ct.RespostaVaziaError):
            ct._extrair_texto_ollama(corpo)


# --------------------------------------------------------------------------
# CORTE PEQUENO: o LLM micro-edita (qualidade) e o Python apara o excesso
# --------------------------------------------------------------------------
# Texto com 4 sentencas + um aparte entre parenteses (material para o aparador).
_TEXTO_DEL = (
    "A reuniao, que estava marcada para as nove horas da manha, foi adiada "
    "para a tarde por causa de um imprevisto. O diretor (visivelmente cansado) "
    "pediu desculpas a todos os presentes. Ele prometeu, mais uma vez, que isso "
    "nao voltaria a acontecer. A equipe aceitou a explicacao sem reclamar "
    "naquele dia."
)

# Saida realista de uma micro-edição (varias frases) usada nos testes de aparo.
_MICRO_OUT = (
    "A praca, recem-inaugurada, virou ponto de encontro. Tem bancos, arvores "
    "e uma fonte. As criancas, animadas, brincam ali toda tarde. Os idosos, "
    "por sua vez, jogam domino sob as arvores antigas. O lugar agrada a todos."
)


def _subseq_sem_espacos(sub, full):
    """True se cada caractere nao-branco de `sub` aparece, NA ORDEM, em `full`.
    Como o aparo so apaga, normaliza espacos e ajusta maiusculas de inicio de
    frase (nunca inventa caracteres), o resultado fiel e sempre uma subsequencia
    do texto de entrada por esta medida -- comparada SEM diferenciar
    maiuscula/minuscula (a capitalizacao de inicio de frase e ajuste esperado)."""
    alvo = iter("".join(full.split()).lower())
    return all(ch in alvo for ch in "".join(sub.split()).lower())


def test_aparar_ja_cabe_retorna_igual():
    txt = "Um texto curto que ja cabe sem aparar."
    assert ct._aparar_para_caber(txt, 1000, 20) == txt


def test_aparar_apara_excesso_e_e_fiel():
    # texto acima do limite -> apara para <= limite, mantendo subsequencia.
    orig = ct.contar(_TEXTO_DEL)
    limite = orig - 30
    faixa = max(ct.TOLERANCIA_RODADA_1, ct.TOLERANCIA_RODADA_2)
    cand = ct._aparar_para_caber(_TEXTO_DEL, limite, faixa)
    assert cand is not None
    assert ct.contar(cand) <= limite
    assert _subseq_sem_espacos(cand, _TEXTO_DEL)


def test_corte_pequeno_usa_micro_edicao_e_apara():
    # corte pequeno -> 1 chamada de micro-edição (_SISTEMA_MICRO); se a saida do
    # LLM passa do limite, o Python apara ate caber, fiel a essa saida.
    orig_in = "z" * (ct.contar(_MICRO_OUT) + 30)    # entrada -> corte pequeno
    limite = ct.contar(_MICRO_OUT) - 15             # micro-saida fica ACIMA
    chamadas = []

    def fake(sistema, usuario):
        chamadas.append(sistema)
        return _MICRO_OUT

    r = ct.resumir(orig_in, limite, chamar_llm=fake)
    assert len(chamadas) == 1                         # uma unica chamada
    assert chamadas[0] == ct._SISTEMA_MICRO           # caminho de micro-edição
    assert r.caracteres <= limite                     # Python aparou para caber
    assert _subseq_sem_espacos(r.texto, _MICRO_OUT)   # so apara a saida do LLM


def test_corte_pequeno_micro_ja_cabe():
    # micro-edição ja devolve <= limite -> usada como esta (sem aparo), sucesso.
    orig_in = "z" * (ct.contar(_MICRO_OUT) + 40)
    limite = ct.contar(_MICRO_OUT) + 5
    r = ct.resumir(orig_in, limite, chamar_llm=lambda s, u: _MICRO_OUT)
    assert r.texto == _MICRO_OUT
    assert r.sucesso and not r.cortado


def test_corte_pequeno_propaga_ollama_indisponivel():
    # Ollama fora do ar na micro-edição -> propaga (nao mascara silenciosamente).
    def fake(sistema, usuario):
        raise ct.ErroOllamaIndisponivel("Ollama nao esta rodando.")

    with pytest.raises(ct.ErroOllamaIndisponivel):
        ct.resumir("z" * 400, 380, chamar_llm=fake)


# --------------------------------------------------------------------------
# Salvaguarda de fidelidade: nomes proprios e numeros do original preservados
# --------------------------------------------------------------------------
def test_fidelidade_detecta_nome_e_numero():
    base = "O evento em Paris reuniu 19 artistas no centro."
    assert ct._fidelidade_ok(base, base)                       # identico: ok
    assert ct._fidelidade_ok(base, "Evento em Paris com 19 artistas.")  # tem ambos
    assert not ct._fidelidade_ok(base, "Evento em Monaco com 19 artistas.")  # nome mudou
    assert not ct._fidelidade_ok(base, "Evento em Paris com artistas.")      # perdeu 19


def test_nomes_proprios_ignora_inicio_de_frase():
    # 'Hoje' abre frase (maiusculo so por posicao) -> nao e nome; 'Paris' e.
    nomes = ct._nomes_proprios("Hoje fui a Paris. Depois voltei.")
    assert "Paris" in nomes
    assert "Hoje" not in nomes and "Depois" not in nomes


def test_corte_pequeno_refaz_se_perde_nome():
    # 1a micro-edição perde 'Paris' -> salvaguarda refaz; 2a e fiel -> adotada.
    orig = ("O grande evento em Paris durou tres dias inteiros e foi um enorme "
            "sucesso de publico e de critica especializada naquela bela cidade.")
    limite = ct.contar(orig) - 15                      # corte pequeno
    saidas = iter([
        "O evento durou tres dias e foi sucesso de publico e critica na cidade.",
        "O evento em Paris durou tres dias e foi sucesso de publico e critica.",
    ])
    calls = []

    def fake(sistema, usuario):
        calls.append(sistema)
        return next(saidas)

    r = ct.resumir(orig, limite, chamar_llm=fake)
    assert "Paris" in r.texto                           # adotou a versao fiel
    assert len(calls) == 2                              # refez por perda de fidelidade
    assert r.caracteres <= limite


def test_corte_pequeno_fallback_quando_nunca_fiel():
    # micro-edição sempre perde o nome -> usa a 1a como fallback (cabe, editavel).
    orig = ("O grande evento em Paris durou tres dias inteiros e foi um enorme "
            "sucesso de publico e de critica especializada naquela bela cidade.")
    limite = ct.contar(orig) - 15
    calls = []

    def fake(sistema, usuario):
        calls.append(sistema)
        return "O evento durou tres dias e foi um sucesso total de publico."

    r = ct.resumir(orig, limite, max_tentativas=3, chamar_llm=fake)
    assert r.caracteres <= limite                       # invariante mantida
    assert len(calls) == 3                              # tentou o maximo (3) e desistiu


def test_ollama_indisponivel_e_subclasse_de_erroresumo():
    # traduzir_excecao repassa a mensagem (passa pelo ramo ErroResumo)
    assert issubclass(ct.ErroOllamaIndisponivel, ct.ErroResumo)
    assert ct.traduzir_excecao(
        ct.ErroOllamaIndisponivel("Ollama fora do ar")) == "Ollama fora do ar"


def test_corte_grande_ainda_usa_gerativo():
    # r alto (resumo de verdade): NAO entra na micro-edição; usa _prompt_inicial.
    sistemas = []

    def fake(sistema, usuario):
        sistemas.append(sistema)
        return "a" * 275

    r = ct.resumir("x" * 1000, 280, chamar_llm=fake)
    assert r.caracteres <= 280
    assert ct._SISTEMA_MICRO not in sistemas        # nao tomou o atalho de micro-edição


def test_remover_e_costurar_invariante_e_costura():
    txt = "Ola, mundo bonito, tudo bem por ai?"
    # remove ", mundo bonito" -> deve costurar sem espaco duplo nem antes de virgula
    i = txt.find(", mundo bonito")
    out = ct._remover_e_costurar(txt, [(i, i + len(", mundo bonito"))])
    assert "  " not in out
    assert " ," not in out
    assert len(out) < len(txt)


def test_costura_conserta_virgula_dupla():
    # remover a oracao do meio SEM as virgulas que a cercam (como o LLM copia)
    # nao pode deixar virgula dupla nem ', ,' no resultado.
    txt = "A casa, que era velha, foi vendida ontem."
    i = txt.find("que era velha")
    out = ct._remover_e_costurar(txt, [(i, i + len("que era velha"))])
    assert ",," not in out and ", ," not in out
    assert out == "A casa, foi vendida ontem."


def test_encolher_apara_clausula_sem_reticencias():
    # tem cláusula acessória removivel -> aparo limpo, sem "..." (mecanico=False)
    txt = "A equipe venceu o jogo, com grande autoridade, ali na capital."
    out, mecanico = ct._encolher_limpo(txt, 40, 20)
    assert ct.contar(out) <= 40
    assert not out.endswith("...")
    assert mecanico is False
    assert _subseq_sem_espacos(out, txt)


def test_encolher_corta_com_reticencias_sem_estrutura():
    # sem virgula/frase para aparar -> corte mecanico por palavra com "..."
    txt = "lorem ipsum dolor sit amet consectetur adipiscing elit sed"
    out, mecanico = ct._encolher_limpo(txt, 20, 20)
    assert ct.contar(out) <= 20
    assert out.endswith("...")
    assert mecanico is True


def test_palavras_alvo_escala_com_limite():
    assert ct._palavras_alvo(60) >= 8 and ct._palavras_alvo(60) <= 12
    assert ct._palavras_alvo(6) >= 1


def test_corrige_maiuscula_no_inicio():
    # remocao que expoe um novo inicio de frase em minuscula -> deve subir
    txt = "Apesar de tudo, as series de streaming dominaram o ano todo."
    i = txt.find("Apesar de tudo, ")
    out = ct._remover_e_costurar(txt, [(i, i + len("Apesar de tudo, "))])
    assert out[0].isupper()                      # nao comeca minusculo
    assert out.startswith("As series")


def test_aparar_protege_abertura_e_fecho():
    # texto com 5 frases; lead e desfecho fortes que NAO devem ser cortados.
    texto = (
        "A cidade investiu pesado em mobilidade neste ano de muitas mudancas. "
        "O metro, que vivia lotado, ganhou dez novos trens importados. "
        "As ciclovias, antes raras, agora cruzam todos os bairros centrais. "
        "Os onibus, enfim, foram totalmente eletrificados pela prefeitura. "
        "O resultado surpreendeu ate os mais pessimistas."
    )
    limite = ct.contar(texto) - 45               # apara 45 chars
    faixa = max(ct.TOLERANCIA_RODADA_1, ct.TOLERANCIA_RODADA_2)
    cand = ct._aparar_para_caber(texto, limite, faixa)
    assert cand is not None and ct.contar(cand) <= limite
    assert cand.startswith("A cidade investiu pesado em mobilidade")  # lead intacto
    assert cand.rstrip().endswith("os mais pessimistas.")             # fecho intacto
    assert _subseq_sem_espacos(cand, texto)


def test_costura_conserta_virgula_antes_de_ponto():
    txt = "Comprei pao, leite e queijo."
    i = txt.find(" e queijo")
    out = ct._remover_e_costurar(txt, [(i, i + len(" e queijo"))])
    assert ",." not in out and ", ." not in out
    assert out.endswith(".")


# --------------------------------------------------------------------------
# traduzir_excecao: passthrough e fallback
# --------------------------------------------------------------------------
def test_traduzir_excecao_passthrough_e_fallback():
    assert ct.traduzir_excecao(ValueError("limite invalido")) == "limite invalido"
    assert ct.traduzir_excecao(ct.ErroResumo("vazio")) == "vazio"
    assert ct.traduzir_excecao(ct.RespostaVaziaError("sem texto")) == "sem texto"
    assert "Erro inesperado" in ct.traduzir_excecao(RuntimeError("boom"))
